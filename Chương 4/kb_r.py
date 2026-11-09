"""KB-E6: inference cost of the full FKG-E versus FISA as the number of attributes r grows (constraint R4).

r grows the way multimodal fusion grows it: ANOVA-selected image and clinical features (r = 8, 16, 23 = every
feature), then tensor-fusion features (image x clinical products, ANOVA-selected on inner train; r = 32, 40, 48).
For every (r, outer fold) the preparation is that of run.py (patient-held-out validation, SMOTE inside teacher
parts, FCM 5 levels, cleaned rule base, cross-fitted FISA teacher). One full FKG-E with the fixed parameters of
config_kb_r.json is trained for max_epochs and read at two validation-only stopping rules on the same trajectory:
(a) minimum validation log loss with patience (= CompactFKGE.fit) and (b) maximum validation AUC with patience.

Per (r, fold): |R|; R1 = AUC/AP/BalAcc of FISA-GM and FKG-E; R2 = label agreement with FISA; R3 = Dev
(mean 1/2 ||a_E - a_F||_1); R4 = unit operations per query (FISA sequential, FISA table, FKG-E; one unit per
multiply, add, compare, table read, exp or log), CPU single-thread time per query (batches sized to keep temporaries
cache-resident, on the first 1024 test queries; and one query at a time on 300 queries;
FKG-E with rule embeddings computed once as in Algorithm 4.2, FISA table with per-label contiguous tables) and
model memory. Times are only compared within one run on one machine.

--cost-only measures R4 alone (no teacher, no training): inference cost depends on |R|, d_e and r, not on the
values of the embeddings, so the FKG-E of a cost-only case keeps its seeded initial parameters. The cross-fitted
FISA teacher needs O(C(r,3)·|R|) work per training query, which makes full cases slow for r >= 40.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import hashlib
import math
import shutil
import sys
import tempfile
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from build import TARGET, verify_release
from compact_fkge import CompactFISA, threshold_on_validation
from report import binary_metrics
from run import hardware, prepare_fold, read_json, sha, synthetic_data, write_json
from train_fkge import CompactFKGE, build_teacher, training_data

ROOT = Path(__file__).resolve().parent
LEVELS, N_LABELS = 5, 2


# ============================================================================ training trajectory
def trace_fit(model, Rval, yval, max_epochs):
    """The update loop of CompactFKGE.fit without stopping; returns per-epoch validation values and parameters."""
    loss, gradient = model.value_grad()
    model.B0 = math.sqrt(loss / model.lam_C)
    last_eta, trace, snapshots = 1.0, [], []
    for epoch in range(max_epochs):
        eta = 1.0 if epoch == 0 else min(2 * last_eta, 1.0)
        with torch.no_grad():
            for _ in range(40):
                candidate = [p.detach() - eta * g for p, g in zip(model.params, gradient)]
                norm = math.sqrt(sum(float((p * p).sum().item()) for p in candidate))
                if norm > model.B0:
                    candidate = [p * (model.B0 / norm) for p in candidate]
                mapping_norm2 = sum(float(((p.detach() - q) ** 2).sum().item())
                                    for p, q in zip(model.params, candidate)) / eta ** 2
                if model.value(candidate) <= loss - 0.5 * eta * mapping_norm2:
                    break
                eta *= 0.5
        model.params = [p.detach().requires_grad_(True) for p in candidate]
        last_eta = eta
        loss, gradient = model.value_grad()
        grad_norm = math.sqrt(sum(float((g * g).sum().item()) for g in gradient))
        probability = model.probabilities(Rval)
        score = np.log(probability[:, 1]) - np.log(probability[:, 0])
        trace.append({"epoch": epoch + 1, "loss": loss, "grad_norm": grad_norm,
                      "val_logloss": float(-np.mean(np.log(probability[np.arange(len(yval)), yval]))),
                      "val_auc": float(roc_auc_score(yval, score))})
        snapshots.append([p.detach().cpu().numpy().copy() for p in model.params])
        if grad_norm <= 1e-4:
            break
    return trace, snapshots


def select_epoch(trace, key, larger, patience):
    """Early stopping of CompactFKGE.fit applied to a trajectory: index of the restored epoch."""
    best, best_index, waiting = (-np.inf if larger else np.inf), 0, 0
    for index, row in enumerate(trace):
        better = row[key] > best + 1e-9 if larger else row[key] < best - 1e-9
        if better:
            best, best_index, waiting = row[key], index, 0
        else:
            waiting += 1
        if waiting >= patience:
            break
    return best_index


# ============================================================================ inference implementations for R4
class PrecomputedFKGE:
    """Algorithm 4.2: rule embeddings and their norms computed once; same probabilities as CompactFKGE."""

    def __init__(self, model, params):
        Z, U = params[0], params[2]
        rules = model.alpha * (model.H.cpu().numpy() @ Z) + (1 - model.alpha) * U[model.y_np]
        unit = rules / np.sqrt((rules * rules).sum(1) + model.eps0)[:, None]
        self.unit = [np.ascontiguousarray(unit[model.y_np == label]) for label in (0, 1)]
        self.Z, self.eps0, self.tau, self.tau_c, self.eps, self.r = Z, model.eps0, model.tau, model.tau_c, model.eps, model.r
        self.lookup = np.full(model.r * LEVELS, -1)
        self.lookup[model.node_codes] = np.arange(len(model.node_codes))
        self.offsets = np.arange(model.r) * LEVELS

    def probabilities(self, R, batch=256):
        out = []
        for start in range(0, len(R), batch):
            codes = self.lookup[self.offsets[None, :] + R[start:start + batch]]
            known = codes >= 0
            q = np.where(known[..., None], self.Z[np.where(known, codes, 0)], 0.0).sum(1) / self.r
            norm = np.sqrt((q * q).sum(1) + self.eps0)
            logits = []
            for unit in self.unit:
                z = (q @ unit.T) / norm[:, None] / self.tau_c
                top = z.max(1)
                logits.append(self.tau_c * (top + np.log(np.exp(z - top[:, None]).sum(1))))
            logits = np.stack(logits, 1) / self.tau
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            out.append((1 - self.eps) * p + self.eps / 2)
        return np.concatenate(out)


class TableFISA:
    """FISA-GM by table lookup with one contiguous (node x key) table per label; same values as CompactFISA."""

    def __init__(self, fisa):
        self.fisa = fisa
        self.nodes = np.arange(len(fisa.nodes))[None, :]
        self.W = [np.ascontiguousarray(fisa.W[:, :, label]) for label in range(N_LABELS)]

    def log_values(self, R, batch=256):
        out = []
        for start in range(0, len(R), batch):
            keys = self.fisa._keys(R[start:start + batch])
            out.append(np.logaddexp(np.stack([np.log(W[self.nodes, keys] + 1e-9).mean(1) for W in self.W], 1),
                                    np.log(1e-12)))
        return np.concatenate(out)


def sequential_log_values(fisa, x):
    """Original FISA by sequential matching: every rule at every node (cost O(|R| C(r,3))), geometric mean."""
    values = np.empty((len(fisa.nodes), N_LABELS))
    for p, columns in enumerate(fisa.nodes):
        match = np.ones(len(fisa.R), dtype=bool)
        for j in columns:
            match &= fisa.R[:, j] == x[j]
        values[p] = np.bincount(fisa.y, weights=match * fisa.weights * fisa.B[:, p], minlength=N_LABELS)
    return np.logaddexp(np.log(values + 1e-9).mean(0), np.log(1e-12))


def operation_counts(r, n_rules, d_e):
    """Unit operations per query of Proposition md:3.13 as implemented (see module docstring)."""
    nodes = comb(r, 3)
    return {"nodes": nodes,
            "ops_fisa_sequential": 8 * nodes * n_rules + 2 * N_LABELS * nodes,
            "ops_fisa_table": 6 * nodes + 4 * N_LABELS * nodes,
            "ops_fkge": n_rules * (2 * d_e + 4) + r * d_e + 2 * d_e}


def memory_counts(r, n_rules, d_e, n_nodes_v):
    """Stored numbers needed for inference: FISA table W; FISA rule weights B needed for its explanation a_F;
    FKG-E rule embeddings plus value and label embeddings."""
    return {"mem_fisa_table": comb(r, 3) * LEVELS ** 3 * N_LABELS, "mem_fisa_rule_weights": n_rules * comb(r, 3),
            "mem_fkge": n_rules * d_e + n_nodes_v * d_e + N_LABELS * d_e}


def bench(functions, repeats):
    """Warm-up, then the fastest of `repeats` interleaved runs; ms per query."""
    for f, _ in functions.values():
        f()
    times = {name: [] for name in functions}
    for _ in range(repeats):
        for name, (f, n) in functions.items():
            started = time.perf_counter()
            f()
            times[name].append((time.perf_counter() - started) / n * 1000)
    return {name: float(np.min(v)) for name, v in times.items()}


# ============================================================================ one (r, fold)
def fidelity(model, params, R, threshold, pred_F, aF):
    """R2: share of FISA labels reproduced; R3: mean 1/2 ||a_E - a_F||_1 with a_E = w of the FKG-E label."""
    agree, dev = [], []
    tensors = [model.tensor(p) for p in params]
    with torch.no_grad():
        for start in range(0, len(R), model.block_size):
            H = model.tensor(model._one_hot(R[start:start + model.block_size]) / model.r)
            probability, log_weights = model._scores(tensors, H)
            label = ((torch.log(probability[:, 1]) - torch.log(probability[:, 0])) > threshold).long()
            aE = torch.zeros((len(label), len(model.R)), dtype=model.dtype, device=model.device)
            for value, indices in enumerate(model.class_indices):
                aE[:, indices] = torch.where((label == value)[:, None], torch.exp(log_weights[value]), 0.0)
            target = model.tensor(aF[start:start + len(label)])
            dev.append((0.5 * (aE - target).abs().sum(1)).cpu().numpy())
            agree.append(label.cpu().numpy() == pred_F[start:start + len(label)])
    return float(np.mean(np.concatenate(agree))), float(np.mean(np.concatenate(dev)))


def measure_cost(fisa, model, params, Rtest, Dt, cfg):
    """R4 of one case: unit operations, memory and CPU single-thread time per query (interleaved, fastest run)."""
    pre, table = PrecomputedFKGE(model, params), TableFISA(fisa)
    if not np.allclose(pre.probabilities(Rtest), model.probabilities(Rtest), rtol=0, atol=1e-10):
        raise AssertionError("Algorithm 4.2 inference differs from CompactFKGE.probabilities")
    if not np.allclose(table.log_values(Rtest), Dt, rtol=0, atol=1e-10):
        raise AssertionError("Per-label table FISA differs from CompactFISA")
    sample = np.random.default_rng(0).choice(len(Rtest), min(cfg["timing"]["single_queries"], len(Rtest)), replace=False)
    sequential = sample[:cfg["timing"]["sequential_queries"]]
    for q in sequential:
        if not np.allclose(sequential_log_values(fisa, Rtest[q]), Dt[q], rtol=0, atol=1e-9):
            raise AssertionError("Sequential FISA differs from CompactFISA")
    # Batches keep the temporaries near the size they have at r = 16 (256 queries x 560 nodes, 256 x 8.000 rules), so
    # that neither method is timed out of cache; batch timings use the first 1024 test queries.
    nodes, n_rules = len(fisa.nodes), len(model.R)
    fisa_batch, fkge_batch = max(8, min(256, 2 ** 17 // nodes)), max(8, min(256, 2 ** 21 // n_rules))
    head = Rtest[:1024]
    with threadpool_limits(limits=1):
        timing = bench({"ms_fisa_package_batch": (lambda: fisa.log_values(head), len(head)),
                        "ms_fisa_table_batch": (lambda: table.log_values(head, fisa_batch), len(head)),
                        "ms_fkge_batch": (lambda: pre.probabilities(head, fkge_batch), len(head)),
                        "ms_fisa_table_query": (lambda: [table.log_values(Rtest[q:q + 1]) for q in sample], len(sample)),
                        "ms_fkge_query": (lambda: [pre.probabilities(Rtest[q:q + 1]) for q in sample], len(sample))},
                       cfg["timing"]["repeats"])
        timing.update(bench({"ms_fisa_sequential_query": (lambda: [sequential_log_values(fisa, Rtest[q]) for q in sequential],
                                                          len(sequential))}, 1))
    r, d_e = Rtest.shape[1], model.d_e
    return {**operation_counts(r, n_rules, d_e), **memory_counts(r, n_rules, d_e, model.nV), **timing,
            "timing_batch_fisa": fisa_batch, "timing_batch_fkge": fkge_batch, "timing_batch_queries": len(head)}


def cost_case(data, assignments, spec, fold, output, cfg, seed, reuse):
    """R4 only: prepared fold (reused from a full run when available), FISA on the cleaned rules, untrained FKG-E."""
    directory = output / f"r{spec['r']:02}" / f"fold_{fold:02}"
    if (directory / "cost.json").exists():
        return read_json(directory / "cost.json")
    started = time.time()
    source = reuse / f"r{spec['r']:02}" / f"fold_{fold:02}" if reuse else None
    if source and (source / "prepared.npz").exists():
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("prepared.npz", "preparation.json"):
            shutil.copy2(source / name, directory / name)
    info = prepare_fold(data, assignments, fold, directory, seed,
                        {"k_img": spec["k_img"], "k_tab": spec["k_tab"], "n_tensor": spec["n_tensor"]})
    with np.load(directory / "prepared.npz") as archive:
        prepared = {name: archive[name] for name in archive.files}
    rules, queries, _ = training_data(prepared)
    Rtest = prepared["Rtest"]
    fisa = CompactFISA(rules["R"], rules["y"], rules["weights"])
    Dt, _ = fisa.log_values(Rtest)
    model = CompactFKGE(rules, queries, np.full((len(queries["R"]), 2), 0.5), None, queries["y"], seed=seed,
                        device="cpu", block_size=128, **cfg["fkge"])
    params = [p.detach().numpy() for p in model.params]
    result = {"r_nominal": spec["r"], "r": Rtest.shape[1], "fold": fold, "spec": spec, "cost_only": True,
              "selected_features": info["selected_features"], "n_rules": len(rules["R"]), "d_e": cfg["fkge"]["d_e"],
              **measure_cost(fisa, model, params, Rtest, Dt, cfg), "case_seconds": time.time() - started}
    write_json(directory / "cost.json", result)
    del fisa, model
    gc.collect()
    return result


def run_case(data, assignments, spec, fold, output, cfg, device, block_size, seed, cache_root):
    directory = output / f"r{spec['r']:02}" / f"fold_{fold:02}"
    if (directory / "result.json").exists():
        return read_json(directory / "result.json")
    started = time.time()
    info = prepare_fold(data, assignments, fold, directory, seed,
                        {"k_img": spec["k_img"], "k_tab": spec["k_tab"], "n_tensor": spec["n_tensor"]})
    with np.load(directory / "prepared.npz") as archive:
        prepared = {name: archive[name] for name in archive.files}
    rules, queries, _ = training_data(prepared)
    Rval, yval, Rtest, ytest = prepared["Rval"], prepared["yval"], prepared["Rtest"], prepared["ytest"]
    r = Rtest.shape[1]
    cache = cache_root / f"r{spec['r']:02}" / f"fold_{fold:02}"
    cache.mkdir(parents=True, exist_ok=True)
    fisa = None
    if not (cache / "teacher_complete.json").exists():
        fisa, seconds = build_teacher({**queries, "Rval": Rval, "yval": yval}, cache, rules)
        write_json(cache / "teacher_complete.json", {"seconds": seconds})
    teacher_seconds = read_json(cache / "teacher_complete.json")["seconds"]
    if fisa is None:
        fisa = CompactFISA(rules["R"], rules["y"], rules["weights"])
    # ---- FISA-GM on the same cleaned rule base (threshold on validation, as run.py)
    Dv, mv = fisa.log_values(Rval)
    threshold_F = threshold_on_validation(yval, np.where(mv, Dv[:, 1] - Dv[:, 0], 0.0))
    Dt, mt = fisa.log_values(Rtest)
    score_F = np.where(mt, Dt[:, 1] - Dt[:, 0], 0.0)
    pred_F = np.where(mt, (score_F > threshold_F).astype(int), fisa.fallback)
    aF = fisa.attribution(Rtest, pred_F)
    fisa._attribution_matrices = {}
    gc.collect()
    # ---- full FKG-E, one trajectory, two validation-only stopping rules
    small = dict(np.load(cache / "teacher_small.npz"))
    attribution = np.load(cache / "teacher_attribution.npy", mmap_mode="r")
    model = CompactFKGE(rules, queries, small["pF"], attribution, small["yF"], seed=seed, device=device,
                        block_size=block_size, **cfg["fkge"])
    train_started = time.time()
    trace, snapshots = trace_fit(model, Rval, yval, cfg["max_epochs"])
    train_seconds = time.time() - train_started
    selections = {}
    for rule, key, larger in (("a_val_logloss", "val_logloss", False), ("b_val_auc", "val_auc", True)):
        index = select_epoch(trace, key, larger, cfg["patience"])
        params = snapshots[index]
        model.params = [model.tensor(p) for p in params]
        pv, pt = model.probabilities(Rval), model.probabilities(Rtest)
        sv, st = np.log(pv[:, 1]) - np.log(pv[:, 0]), np.log(pt[:, 1]) - np.log(pt[:, 0])
        threshold = threshold_on_validation(yval, sv)
        prediction = (st > threshold).astype(int)
        agree, dev = fidelity(model, params, Rtest, threshold, pred_F, aF)
        selections[rule] = {"epoch": index + 1, **{k: trace[index][k] for k in ("val_logloss", "val_auc")},
                            "threshold": threshold, "R2_agreement": agree, "R3_dev": dev,
                            **binary_metrics(ytest, prediction, st)}
        pd.DataFrame({"fkge_score": st, "fkge_prediction": prediction}).to_csv(directory / f"fkge_{rule}.csv", index=False)
    pd.DataFrame({"fisa_score": score_F, "fisa_prediction": pred_F, TARGET: ytest}).to_csv(directory / "fisa.csv", index=False)
    del attribution, aF
    # ---- R4: operations, memory, CPU single-thread time (FKG-E parameters of rule (b); time depends on shapes only)
    params = snapshots[selections["b_val_auc"]["epoch"] - 1]
    model.params = [model.tensor(p) for p in params]
    result = {"r_nominal": spec["r"], "r": r, "fold": fold, "spec": spec, "selected_features": info["selected_features"],
              "n_rules": len(rules["R"]), "training_queries": len(queries["R"]), "d_e": cfg["fkge"]["d_e"],
              "fkge_parameters": cfg["fkge"], "teacher_seconds": teacher_seconds, "fkge_train_seconds": train_seconds,
              "epochs_run": len(trace),
              "FISA": {"threshold": threshold_F, "matched_rate": float(mt.mean()), **binary_metrics(ytest, pred_F, score_F)},
              "FKG-E": selections, **measure_cost(fisa, model, params, Rtest, Dt, cfg), "trace": trace}
    result["case_seconds"] = time.time() - started
    write_json(directory / "result.json", result)
    (cache / "teacher_attribution.npy").unlink(missing_ok=True)      # large; the case result is complete
    del model, fisa, snapshots
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


# ============================================================================ report
def render(output, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for path in sorted(output.glob("r*/fold_*/result.json")) + sorted(output.glob("r*/fold_*/cost.json")):
        res = read_json(path)
        row = {k: res.get(k, np.nan) for k in (
            "r_nominal", "r", "fold", "n_rules", "nodes", "ops_fisa_sequential", "ops_fisa_table", "ops_fkge",
            "mem_fisa_table", "mem_fisa_rule_weights", "mem_fkge", "ms_fisa_package_batch", "ms_fisa_table_batch",
            "ms_fkge_batch", "ms_fisa_table_query", "ms_fkge_query", "ms_fisa_sequential_query", "teacher_seconds",
            "fkge_train_seconds")}
        row["cost_only"] = bool(res.get("cost_only", False))
        if "FISA" in res:
            row.update({f"fisa_{m}": res["FISA"][m] for m in ("roc_auc", "average_precision", "balanced_accuracy")})
            for rule, key in (("a_val_logloss", "a"), ("b_val_auc", "b")):
                sel = res["FKG-E"][rule]
                row.update({f"fkge_{key}_{m}": sel[m] for m in ("roc_auc", "average_precision", "balanced_accuracy",
                                                               "R2_agreement", "R3_dev", "epoch")})
        rows.append(row)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    for column in ("fisa_roc_auc", "fkge_a_roc_auc", "fkge_b_roc_auc"):
        if column not in frame:
            frame[column] = np.nan
    frame["ops_ratio_sequential"] = frame.ops_fkge / frame.ops_fisa_sequential
    frame["ops_ratio_table"] = frame.ops_fkge / frame.ops_fisa_table
    frame["time_ratio_table_batch"] = frame.ms_fkge_batch / frame.ms_fisa_table_batch
    frame["time_ratio_table_query"] = frame.ms_fkge_query / frame.ms_fisa_table_query
    frame["time_ratio_sequential_query"] = frame.ms_fkge_query / frame.ms_fisa_sequential_query
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["r_nominal", "fold"]).to_csv(reports / "kb_r_by_fold.csv", index=False)
    grouped = frame.drop(columns="fold").groupby("r_nominal")
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd")).join(
        frame.groupby("r_nominal").fold.nunique().rename("n_folds"))
    summary.to_csv(reports / "kb_r_mean_std.csv")
    m = grouped.mean()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for column, label in (("ops_fisa_sequential", "FISA tuần tự"), ("ops_fisa_table", "FISA bảng tra"), ("ops_fkge", "FKG-E đầy đủ")):
        axes[0].plot(m.index, m[column], marker="o", label=label)
    axes[0].set_yscale("log"); axes[0].set_title("Số phép tính / truy vấn")
    for column, label in (("ms_fisa_table_batch", "FISA bảng tra (lô)"), ("ms_fkge_batch", "FKG-E (lô)"),
                          ("ms_fisa_table_query", "FISA bảng tra (từng truy vấn)"), ("ms_fkge_query", "FKG-E (từng truy vấn)")):
        axes[1].plot(m.index, m[column], marker="o", label=label)
    axes[1].set_yscale("log"); axes[1].set_title("Thời gian CPU 1 luồng (ms / truy vấn)")
    for column, label in (("fisa_roc_auc", "FISA-GM"), ("fkge_a_roc_auc", "FKG-E, dừng (a)"), ("fkge_b_roc_auc", "FKG-E, dừng (b)")):
        axes[2].errorbar(m.index, m[column], yerr=grouped.std(ddof=1)[column], marker="o", capsize=3, label=label)
    axes[2].set_title("AUC outer test")
    for ax in axes:
        ax.set_xlabel("Số thuộc tính r"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(reports / "kb_r.png", dpi=130); plt.close(fig)
    lines = ["KB-E6: chi phí suy diễn theo số thuộc tính r (R4) và chất lượng (R1–R3).",
             "Tỉ số < 1 nghĩa là FKG-E đầy đủ rẻ hơn. Thời gian chỉ so trong cùng lượt chạy, cùng máy (CPU 1 luồng).", ""]
    for r_nom, x in frame.groupby("r_nominal"):
        flags = " ".join(f"ρ={rho}: tuần tự {'đạt' if x.ops_ratio_sequential.max() <= rho else 'không'},"
                         f" bảng tra(phép tính) {'đạt' if x.ops_ratio_table.max() <= rho else 'không'},"
                         f" bảng tra(thời gian lô) {'đạt' if x.time_ratio_table_batch.max() <= rho else 'không'};"
                         for rho in cfg["rho"])
        lines.append(f"r={r_nom} (r thực {sorted(int(v) for v in x.r.unique())}, {x.fold.nunique()} fold, |R| TB {x.n_rules.mean():.0f}): "
                     f"T_E/T_seq {x.ops_ratio_sequential.mean():.4f}, T_E/T_tab {x.ops_ratio_table.mean():.2f}, "
                     f"thời gian lô {x.time_ratio_table_batch.mean():.2f}, từng truy vấn {x.time_ratio_table_query.mean():.2f} | {flags}")
    (reports / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/kb_r")
    parser.add_argument("--config", type=Path, default=ROOT / "config_kb_r.json")
    parser.add_argument("--r", type=int, nargs="+", help="Subset of the nominal r values of the config")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=["all", "report"], default="all")
    parser.add_argument("--smoke", action="store_true", help="Synthetic end-to-end check; never a BRSET result")
    parser.add_argument("--cost-only", action="store_true", help="R4 only: no teacher, no training (use a new --output)")
    parser.add_argument("--reuse-prepared", type=Path, help="Full kb_r output whose prepared folds a cost-only run copies")
    args = parser.parse_args()
    output = args.output.resolve()
    cfg = read_json(args.config)
    if args.smoke:
        cfg.update(r_grid=[{"r": 4, "k_img": 2, "k_tab": 2, "n_tensor": 0}, {"r": 6, "k_img": 2, "k_tab": 2, "n_tensor": 2}],
                   fkge={**cfg["fkge"], "d_e": 4}, max_epochs=3, patience=2,
                   timing={"repeats": 1, "single_queries": 5, "sequential_queries": 1})
    if args.stage == "report":
        render(output, cfg)
        return
    specs = [s for s in cfg["r_grid"] if not args.r or s["r"] in args.r]
    if args.r and len(specs) != len(set(args.r)):
        parser.error("--r must list nominal r values of the config")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.smoke:
        data, assignments = synthetic_data(args.seed)
        scope = "SYNTHETIC_SMOKE_ONLY"
    else:
        verify_release(ROOT / "data")
        data = pd.read_csv(ROOT / "data/fusion_features.csv", dtype={"image_id": str, "patient_id": str},
                           float_precision="round_trip")
        assignments = pd.read_csv(ROOT / "data/fold_assignments.csv", dtype={"image_id": str, "patient_id": str})
        scope = "BRSET_FULL_16266_IMAGES_8524_PATIENTS"
    code = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob("*.py"))
            if not any(part in {"outputs", "private", "__pycache__"} for part in p.relative_to(ROOT).parts)}
    identity = {"scope": scope, "config": cfg, "code_sha256": code, "seed": args.seed, "device": device,
                "block_size": args.block_size, "cost_only": args.cost_only,
                "data_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()}
    manifest_path = output / "kb_r_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise SystemExit(f"{output} was made with other code/config/data/device: use a new --output")
        manifest.setdefault("resume_events", []).append({"argv": [sys.executable, *sys.argv], "hardware": hardware(device)})
    else:
        manifest = {"identity": identity, "hardware": hardware(device), "argv": [sys.executable, *sys.argv],
                    "selection": "fixed FKG-E parameters; epochs chosen on validation only (rules a and b)"}
    write_json(manifest_path, manifest)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache_root = Path(tempfile.gettempdir()) / "fkge_kb_r" / hashlib.sha256(
        (str(output) + str(sorted(code.items())) + str(args.seed)).encode()).hexdigest()[:20]
    for spec in specs:
        for fold in args.folds:
            started = time.time()
            if args.cost_only:
                res = cost_case(data, assignments, spec, fold, output, cfg, args.seed,
                                args.reuse_prepared.resolve() if args.reuse_prepared else None)
                print(f"r={spec['r']} fold {fold} (chi phí): |R|={res['n_rules']} r={res['r']} | ops E/seq "
                      f"{res['ops_fkge'] / res['ops_fisa_sequential']:.4f} E/tab {res['ops_fkge'] / res['ops_fisa_table']:.2f} | "
                      f"ms lô FISA {res['ms_fisa_table_batch']:.4f} FKG-E {res['ms_fkge_batch']:.4f} | từng truy vấn FISA "
                      f"{res['ms_fisa_table_query']:.4f} FKG-E {res['ms_fkge_query']:.4f} | {time.time() - started:.0f}s", flush=True)
                render(output, cfg)
                continue
            res = run_case(data, assignments, spec, fold, output, cfg, device, args.block_size, args.seed, cache_root)
            b = res["FKG-E"]["b_val_auc"]
            print(f"r={spec['r']} fold {fold}: |R|={res['n_rules']} r={res['r']} | AUC FISA {res['FISA']['roc_auc']:.4f} "
                  f"FKG-E(a) {res['FKG-E']['a_val_logloss']['roc_auc']:.4f} (b) {b['roc_auc']:.4f} | R2 {b['R2_agreement']:.3f} "
                  f"Dev {b['R3_dev']:.3f} | ops E/tab {res['ops_fkge'] / res['ops_fisa_table']:.2f} | ms lô FISA "
                  f"{res['ms_fisa_table_batch']:.4f} FKG-E {res['ms_fkge_batch']:.4f} | {time.time() - started:.0f}s", flush=True)
            render(output, cfg)
    if all((output / f"r{s['r']:02}" / f"fold_{f:02}" / "result.json").exists() for s in specs for f in args.folds):
        shutil.rmtree(cache_root, ignore_errors=True)
    print(f"DONE: {output}", flush=True)


if __name__ == "__main__":
    main()
