"""Chapter 4 baselines on the same outer folds, inner train and validation as FKG-E.

KGE + kNN (Node2Vec, TransE, DistMult) learn token embeddings from the same cleaned rule base as FKG-E
(conflicting rules removed, every rule kept once; SMOTE rows included before cleaning). Unlike the earlier *Lite versions, which saw 2,000
pairs/triples per epoch, every pair/triple is used in every epoch. Embedding size d and the kNN k are chosen
by validation log loss; the decision threshold by validation balanced accuracy.

Classical ML (logistic regression, random forest, XGBoost) uses the fold-local standardized selected
features before fuzzification, trained on real inner-train images only with class weighting; the
configuration is chosen by validation log loss and the threshold by validation balanced accuracy.
The outer test set is never used for any choice.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from compact_fkge import threshold_on_validation

LEVELS = 5


def _log_loss(y, p1):
    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
    return float(-np.mean(np.where(y == 1, np.log(p1), np.log(1 - p1))))


# ============================================================ KGE + kNN
def token_ids(R):
    """Antecedent token of attribute j at level v -> j * LEVELS + v (the FKG-E node code)."""
    return np.arange(R.shape[1])[None, :] * LEVELS + R


def cooccurrence(R, weights):
    """Weighted co-occurrence mu_uv of tokens of different attributes (formula 3.5), dense V x V."""
    V = R.shape[1] * LEVELS
    H = np.zeros((len(R), V))
    H[np.arange(len(R))[:, None], token_ids(R)] = 1.0
    A = (H.T * weights) @ H / weights.sum()
    np.fill_diagonal(A, 0.0)
    return A


def random_walks(A, walks_per_node, length, rng):
    """Uniform (p = q = 1) Node2Vec walks; transition probability proportional to mu_uv."""
    nodes = np.flatnonzero(A.sum(1) > 0)
    P = np.cumsum(A / np.maximum(A.sum(1, keepdims=True), 1e-300), axis=1)
    walks = np.empty((len(nodes) * walks_per_node, length), dtype=int)
    walks[:, 0] = np.tile(nodes, walks_per_node)
    for t in range(1, length):
        u = rng.random_sample(len(walks))
        walks[:, t] = np.minimum((P[walks[:, t - 1]] < u[:, None]).sum(1), A.shape[0] - 1)
    return walks


def triples(R, y, weights):
    """(head, relation, tail, weight): relation 0 = co_occurs (ordered antecedent pairs), 1 = implies class."""
    V = R.shape[1] * LEVELS
    tok = token_ids(R)
    counts = {}
    for row, label, w in zip(tok, y, weights):
        for i, j in itertools.permutations(row, 2):
            counts[(i, 0, j)] = counts.get((i, 0, j), 0.0) + w
        for i in row:
            key = (i, 1, V + int(label))
            counts[key] = counts.get(key, 0.0) + w
    T = np.array(list(counts), dtype=int)
    return T, np.array(list(counts.values())) / weights.sum()


def train_kge(method, R, y, weights, d, cfg, seed, device):
    import torch
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    V = R.shape[1] * LEVELS + 2
    started = time.time()
    if method == "Node2Vec":
        A = cooccurrence(R, weights)
        walks = random_walks(A, cfg["walks_per_node"], cfg["walk_length"], rng)
        pairs = [np.column_stack([walks[:, t], walks[:, t + o]])
                 for o in range(-cfg["window"], cfg["window"] + 1) if o
                 for t in range(max(0, -o), walks.shape[1] - max(0, o))]
        pairs = torch.as_tensor(np.vstack(pairs), device=device)
        freq = np.bincount(walks.ravel(), minlength=V).astype(float) ** 0.75
        noise = torch.as_tensor(freq / freq.sum(), device=device)
        E = torch.nn.Parameter(0.1 * torch.randn(V, d, dtype=torch.float64, device=device))
        C = torch.nn.Parameter(0.1 * torch.randn(V, d, dtype=torch.float64, device=device))
        opt = torch.optim.Adam([E, C], lr=cfg["lr"])
        for _ in range(cfg["epochs"]):
            for idx in torch.randperm(len(pairs), device=device).split(cfg["batch"]):
                u, v = pairs[idx, 0], pairs[idx, 1]
                neg = torch.multinomial(noise, len(idx) * cfg["negatives"], replacement=True).view(len(idx), -1)
                loss = -(torch.nn.functional.logsigmoid((E[u] * C[v]).sum(1)).mean()
                         + torch.nn.functional.logsigmoid(-(E[u][:, None, :] * C[neg]).sum(2)).sum(1).mean())
                opt.zero_grad()
                loss.backward()
                opt.step()
        return E.detach().cpu().numpy(), time.time() - started
    T, w = triples(R, y, weights)
    T = torch.as_tensor(T, device=device)
    w = torch.as_tensor(w, device=device)
    E = torch.nn.Parameter(0.1 * torch.randn(V, d, dtype=torch.float64, device=device))
    Rel = torch.nn.Parameter(0.1 * torch.randn(2, d, dtype=torch.float64, device=device))
    opt = torch.optim.Adam([E, Rel], lr=cfg["lr"])
    k = cfg["negatives"]
    for _ in range(cfg["epochs"]):
        for idx in torch.randperm(len(T), device=device).split(cfg["batch"]):
            h, r, t, ww = T[idx, 0], T[idx, 1], T[idx, 2], w[idx] / w[idx].sum()
            corrupt = torch.randint(0, V, (len(idx), k), device=device)
            if method == "TransE":
                head_side = torch.rand(len(idx), k, device=device) < 0.5
                h2 = torch.where(head_side, corrupt, h[:, None])
                t2 = torch.where(head_side, t[:, None], corrupt)
                pos = (E[h] + Rel[r] - E[t]).norm(dim=1)
                neg = (E[h2] + Rel[r][:, None, :] - E[t2]).norm(dim=2)
                loss = (ww[:, None] * torch.relu(cfg["margin"] + pos[:, None] - neg)).sum() / k
            else:  # DistMult
                pos = (E[h] * Rel[r] * E[t]).sum(1)
                neg = (E[h][:, None, :] * Rel[r][:, None, :] * E[corrupt]).sum(2)
                loss = -(ww * (torch.nn.functional.logsigmoid(pos)
                               + torch.nn.functional.logsigmoid(-neg).mean(1))).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if method == "TransE":
                with torch.no_grad():
                    E.div_(E.norm(dim=1, keepdim=True).clamp_min(1.0))
    return E.detach().cpu().numpy(), time.time() - started


class WeightedKNN:
    """Rule/query vector = mean of its antecedent token embeddings; cosine top-k over the rule base
    where a compact row counts with its multiplicity. Score = smoothed share of DR-positive votes."""

    def __init__(self, E, R, y, weights, k):
        self.E, self.y, self.weights, self.k = E, y, weights, k
        self.rules = self._vectors(R)

    def _vectors(self, R):
        v = self.E[token_ids(R)].mean(1)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    def scores(self, R, batch=256):
        """Rules tied with the k-th neighbour (equal antecedents give equal cosines) share the remaining
        vote budget in proportion to their weights, so the score does not depend on row order."""
        out = []
        k = min(self.k, self.weights.sum())
        for start in range(0, len(R), batch):
            sim = self._vectors(R[start:start + batch]) @ self.rules.T
            order = np.argsort(-sim, axis=1, kind="stable")
            ranked = np.take_along_axis(sim, order, 1)
            position = (np.cumsum(self.weights[order], 1) < k).sum(1)
            boundary = ranked[np.arange(len(sim)), position][:, None]
            full, tie = sim > boundary, sim == boundary
            remaining = k - (full * self.weights).sum(1, keepdims=True)
            used = full * self.weights + tie * self.weights * remaining / (tie * self.weights).sum(1, keepdims=True)
            out.append(((used * self.y).sum(1) + 1) / (used.sum(1) + 2))
        return np.concatenate(out)


def run_kge(method, prepared, cfg, seed, device):
    R, y, weights = prepared["R"], prepared["y"], prepared["weights"]
    trials = []
    for d in cfg["d_grid"]:
        E, seconds = train_kge(method, R, y, weights, d, cfg, seed, device)
        for k in cfg["k_grid"]:
            knn = WeightedKNN(E, R, y, weights, k)
            p_val = knn.scores(prepared["Rval"])
            trials.append({"d": d, "k": k, "validation_log_loss": _log_loss(prepared["yval"], p_val),
                           "train_seconds": seconds, "model": knn, "p_val": p_val})
    best = min(trials, key=lambda t: t["validation_log_loss"])
    threshold = threshold_on_validation(prepared["yval"], best["p_val"])
    return best, threshold, [{key: t[key] for key in ("d", "k", "validation_log_loss", "train_seconds")}
                             for t in trials]


# ============================================================ classical ML
def ml_models(name, cfg, seed, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    if name == "Logistic regression":
        for C in cfg["lr_C"]:
            yield {"C": C}, LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    elif name == "Random forest":
        for leaf in cfg["rf_min_samples_leaf"]:
            yield {"min_samples_leaf": leaf}, RandomForestClassifier(
                n_estimators=cfg["rf_trees"], min_samples_leaf=leaf, class_weight="balanced_subsample",
                n_jobs=cfg["n_jobs"], random_state=seed)
    elif name == "XGBoost":
        from xgboost import XGBClassifier
        ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
        for depth, rate in itertools.product(cfg["xgb_max_depth"], cfg["xgb_learning_rate"]):
            yield {"max_depth": depth, "learning_rate": rate}, XGBClassifier(
                n_estimators=cfg["xgb_trees"], max_depth=depth, learning_rate=rate, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=ratio, tree_method="hist", n_jobs=cfg["n_jobs"],
                random_state=seed, eval_metric="logloss")
    else:
        raise ValueError(name)


ML_METHODS = ("Logistic regression", "Random forest", "XGBoost")


def run_ml(name, X, y, Xval, yval, cfg, seed):
    trials, best = [], None
    for params, model in ml_models(name, cfg, seed, y):
        started = time.time()
        model.fit(X, y)
        seconds = time.time() - started
        p_val = model.predict_proba(Xval)[:, 1]
        loss = _log_loss(yval, p_val)
        trials.append({**params, "validation_log_loss": loss, "train_seconds": seconds})
        if best is None or loss < best[0]:
            best = (loss, params, model, p_val, seconds)
    loss, params, model, p_val, seconds = best
    return model, params, loss, threshold_on_validation(yval, p_val), seconds, trials
