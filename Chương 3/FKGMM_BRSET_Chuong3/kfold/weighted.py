"""Audit fixes for FISA-GM (pre-registered 02/11/2026, before any test result of these variants was seen).

F1 - triple-weighted FISA-GM. FISA-GM gives every attribute triple the same weight:
     s(x) = (1/T) sum_t s_t(x),  s_t(x) = log(C_1(t)+eps) - log(C_0(t)+eps).
     With many weak attributes (e.g. 7 image features among 16) most triples are noise and dilute the
     informative ones. F1 uses s_w(x) = sum_t w_t s_t(x) / sum_t w_t with w_t = max(AUC_t - 1/2, 0), where
     AUC_t is estimated by 5-part patient-grouped cross-fitting on the ORIGINAL training images only
     (no SMOTE rows, no validation or test rows). The final graph is the usual one on all training rules.
     Prediction stays "s_w > 0" (weighted geometric means compared, as argmax D for FISA-GM).
F2 - binary-aware fuzzification (sensitivity analysis). Columns with two values on the original training
     images keep two linguistic levels: SMOTE interpolations are snapped to the nearer observed value
     instead of FCM placing 5 clusters on a binary variable.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from fkgmm.fisa import FKG
from kfold.core import FCMFuzzifier


class BinaryAwareFuzzifier(FCMFuzzifier):
    def __init__(self, n_clusters=5, n_orig=None, **kw):
        super().__init__(n_clusters, **kw)
        self.n_orig = n_orig

    def fit(self, X):
        super().fit(X)
        Xo = X[: self.n_orig] if self.n_orig else X
        self.binary = {j: np.unique(Xo[:, j]) for j in range(X.shape[1]) if len(np.unique(Xo[:, j])) == 2}
        return self

    def transform(self, X):
        R = super().transform(X)
        for j, v in self.binary.items():
            R[:, j] = (np.abs(X[:, j] - v[1]) < np.abs(X[:, j] - v[0])).astype(int)
        return R


def triple_scores(model, R, eps):
    L = np.log(model._lookup(R) + eps)
    return L[:, 1, :] - L[:, 0, :]


def triple_weights(R_orig, y_orig, patients, eps=1e-9, parts=5, seed=0):
    """Cross-fitted per-triple AUC on original training images; returns (w, auc)."""
    n_t = None
    S = None
    for tr, va in StratifiedGroupKFold(parts, shuffle=True, random_state=seed).split(R_orig, y_orig, patients):
        m = FKG(c_norm="none", aggregator="geomean", eps=eps).fit(R_orig[tr], y_orig[tr])
        s = triple_scores(m, R_orig[va], eps)
        if S is None:
            n_t = s.shape[1]
            S = np.zeros((len(y_orig), n_t))
        S[va] = s
    auc = np.array([roc_auc_score(y_orig, S[:, t]) if np.ptp(S[:, t]) > 0 else 0.5 for t in range(n_t)])
    w = np.maximum(auc - 0.5, 0.0)
    return (w if w.sum() > 0 else np.ones(n_t)), auc


def fisa_gm_variant(D, weighted=False, binary_aware=False, n_clusters=5, eps=1e-9, seed=0):
    """One group/fold of kfold.core.load(); returns (pred, score, info). Uses no test labels."""
    n0 = D["n_orig_train"]
    fz = (BinaryAwareFuzzifier(n_clusters, n_orig=n0) if binary_aware else FCMFuzzifier(n_clusters)).fit(D["Xtr"])
    Rtr, Rte = fz.transform(D["Xtr"]), fz.transform(D["Xte"])
    model = FKG(c_norm="none", aggregator="geomean", eps=eps).fit(Rtr, D["ytr"])
    s = triple_scores(model, Rte, eps)
    info = {"n_triples": s.shape[1]}
    if weighted:
        w, auc = triple_weights(Rtr[:n0], D["ytr"][:n0], D["ids_tr"].patient_id.astype(str).to_numpy(), eps, seed=seed)
        score = s @ w / w.sum()
        info.update(w_nonzero=int((w > 0).sum()), inner_auc_median=float(np.median(auc)))
    else:
        score = s.mean(1)
    return (score > 0).astype(int), score, info
