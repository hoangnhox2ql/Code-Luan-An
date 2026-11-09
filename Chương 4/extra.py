"""Chapter 4 supplementary experiments on a finished run.py output: KB2-KB6, ablation and baselines.

Every experiment reuses the main run's outer folds, inner train, validation split, fold-local
preprocessing and SMOTE, and starts from the FKG-E configuration that run.py locked on validation
(fold_XX/selected.json). Only the factor under study changes; the outer test set is never used to choose
anything. KB3 (d_e) and KB4 (lambda_I, lambda_P) are read from the 108 grid trials already trained:
for each value, the trial with the lowest validation loss over the remaining parameters is evaluated.

    python extra.py --output outputs/brset_full --device auto --resume
    python extra.py --output outputs/brset_full --only kb3,kb4,baseline     # no new FKG-E training
    python extra.py --output outputs/brset_full --stage report

Results: <output>/extra/jobs/fold_XX/*.json|csv (one file per job; finished jobs are skipped on rerun)
and <output>/extra/reports/*.csv|png.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import hashlib
import json
import math
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baselines import ML_METHODS, run_kge, run_ml
from build import IDS, TARGET
from compact_fkge import CompactFISA, select_rules, threshold_on_validation
from report import binary_metrics
from run import ROOT, hardware, read_json, sha, synthetic_data, write_json
from sfkgs import sfkgs_compact
from train_fkge import RULE_BASE, CompactFKGE, build_teacher, training_data

SCENARIOS = ("kb2", "kb3", "kb4", "kb5", "kb6", "ablation", "baseline")
ABLATION = {   # one component of objective (3.28) removed from the validation-selected configuration
    "Full FKG-E": {},
    "Bỏ L_pred (λ_P = 0)": {"lam_P": 0.0},
    "Bỏ L_inf (λ_I = 0)": {"lam_I": 0.0},
    "Bỏ L_rule (λ_R = 0)": {"lam_R": 0.0},
    "Bỏ bảo toàn cạnh (λ_E = 0)": {"lam_E": 0.0},
    "Bỏ bảo toàn nút (λ_N = 0)": {"lam_N": 0.0},
    "Bỏ cấu trúc (λ_E = λ_N = 0)": {"lam_E": 0.0, "lam_N": 0.0},
    "Gộp trung bình đều (α = r/(r+1))": {"alpha": "mean"},
}
KGE_METHODS = ("Node2Vec", "TransE", "DistMult")


def metrics(y, prediction, score):
    m = binary_metrics(y, prediction, score)
    m["sensitivity"] = m["recall"]
    m["specificity"] = float(np.mean(np.asarray(prediction)[np.asarray(y) == 0] == 0))
    return m


def ov_key(overrides):
    return ",".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "default"


def timed(function, repeats=3):
    """Warm-up call, then the fastest of ``repeats`` timed calls (seconds) and the last result."""
    function()
    best = np.inf
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - started)
    return result, best


def ratio_key(r):
    return "full" if r >= 1.0 else f"ratio_{r:g}"


# ============================================================================ one outer fold
class Fold:
    def __init__(self, output, fold, ecfg, mcfg, seed, device, block_size, cache_root, main_cache, data):
        self.dir = output / f"fold_{fold:02}"
        self.fold, self.ecfg, self.mcfg, self.seed = fold, ecfg, mcfg, seed
        self.device, self.block_size, self.data = device, block_size, data
        self.jobs = output / "extra" / "jobs" / self.dir.name
        self.cache = cache_root / self.dir.name
        self.main_cache = main_cache / self.dir.name if main_cache else None
        if read_json(self.dir / "preparation.json").get("rule_base") != RULE_BASE:
            raise SystemExit(f"{self.dir}: the main run used another rule base; run run.py of this package first")
        self.selected = read_json(self.dir / "selected.json")
        self.params = self.selected["parameters"]
        with np.load(self.dir / "prepared.npz") as archive:
            self.p = {name: archive[name] for name in archive.files}
        # Cleaned rule base; L_pred/L_inf/L_rule queries = every kept training row, whatever the rule base (4.4.1).
        self.full, self.queries, _ = training_data(self.p)
        self.teacher_input = {**self.queries, "Rval": self.p["Rval"], "yval": self.p["yval"]}
        self.test_ids = pd.read_csv(self.dir / "test_ids.csv", dtype={"image_id": str, "patient_id": str})
        self._rb = {}

    # ------------------------------------------------------------------ rule bases
    def rulebase(self, key):
        """Subsets of the cleaned rule base (unique, non-conflicting rules, weight 1): full; S-FKGS with theta;
        ratio r = ceil(r |R|) rules drawn without replacement. Each kept rule keeps its rows for the leakage mask."""
        if key in self._rb:
            return self._rb[key]
        full = self.full
        meta = {"key": key, "rules_after_cleaning": len(full["R"])}
        if key == "full":
            rb = full
        elif key == "fkgs":
            started = time.time()
            keep, rep, m = sfkgs_compact(full["R"], full["y"], full["weights"], self.ecfg["sfkgs_theta"])
            rb = select_rules(full, keep)
            meta.update(method="S-FKGS", theta=self.ecfg["sfkgs_theta"], budget_m=m, rep=rep,
                        select_seconds=time.time() - started)
        elif key.startswith("ratio_"):
            r = float(key.split("_", 1)[1])
            n = len(full["R"])
            keep = np.sort(np.random.default_rng(self.ecfg["kb6_seed"]).choice(n, int(math.ceil(r * n)), replace=False))
            rb = select_rules(full, keep)
            meta.update(method="random rules without replacement", ratio=r, seed=self.ecfg["kb6_seed"])
        else:
            raise ValueError(key)
        if len(np.unique(rb["y"])) != 2:
            raise ValueError(f"Rule base {key} lost a class")
        meta.update(n_rules=len(rb["R"]))
        self._rb[key] = (rb, meta)
        return rb, meta

    def teacher(self, key):
        rb, _ = self.rulebase(key)
        for directory in ([self.main_cache] if key == "full" and self.main_cache else []) + [self.cache / key]:
            if (directory / "teacher_complete.json").exists() and str(
                    np.load(directory / "teacher_small.npz").get("rule_base")) == RULE_BASE:
                break
        else:
            directory = self.cache / key
            directory.mkdir(parents=True, exist_ok=True)
            _, seconds = build_teacher(self.teacher_input, directory, rb)
            write_json(directory / "teacher_complete.json", {"seconds": seconds})
        small = dict(np.load(directory / "teacher_small.npz"))
        return small, np.load(directory / "teacher_attribution.npy", mmap_mode="r"), read_json(
            directory / "teacher_complete.json")["seconds"]

    # ------------------------------------------------------------------ jobs
    def _save(self, key, record, score=None, prediction=None):
        self.jobs.mkdir(parents=True, exist_ok=True)
        if score is not None:
            frame = self.test_ids.copy()
            frame["score"], frame["prediction"] = score, np.asarray(prediction, int)
            frame.to_csv(self.jobs / f"{key}.csv", index=False)
        record.update(job=key, fold=self.fold)
        write_json(self.jobs / f"{key}.json", record)

    def done(self, key):
        return (self.jobs / f"{key}.json").exists()

    def _evaluate(self, model):
        probability, seconds = timed(lambda: model.probabilities(self.p["Rtest"]))
        score = np.log(probability[:, 1]) - np.log(probability[:, 0])
        prediction = (score > model.threshold).astype(int)
        return score, prediction, seconds

    def trial(self, index):
        key = f"trial_{index:03}"
        record = read_json(self.dir / "trials" / f"{key}.json")
        p = self.p
        rb, _ = self.rulebase("full")
        model = CompactFKGE(rb, self.queries, np.full((len(self.queries["R"]), 2), 0.5), None, self.queries["y"],
                            seed=self.seed, device=self.device, block_size=self.block_size, **record["parameters"])
        checkpoint = np.load(self.dir / "trials" / f"{key}.npz")
        model.params = [model.tensor(checkpoint[name]) for name in ("Z", "C", "U")]
        model.threshold = float(checkpoint["threshold"])
        score, prediction, seconds = self._evaluate(model)
        self._save(key, {"kind": "fkge_grid_trial", "rulebase": "full", "parameters": record["parameters"],
                         "validation_log_loss": record["validation_log_loss"], "epochs": record["epochs"],
                         "train_seconds": record["seconds"], "n_rules": len(rb["R"]),
                         "ms_per_query": seconds / len(score) * 1000,
                         "test": metrics(p["ytest"], prediction, score)}, score, prediction)

    def fkge(self, rb_key, overrides):
        key = f"fkge__{rb_key}__{ov_key(overrides)}"
        rb, meta = self.rulebase(rb_key)
        small, attribution, teacher_seconds = self.teacher(rb_key)
        p = self.p
        params = {**self.params, **overrides}
        print(f"START {self.dir.name} {key}: {params}", flush=True)
        model = CompactFKGE(rb, self.queries, small["pF"], attribution, small["yF"], seed=self.seed,
                            device=self.device, block_size=self.block_size,
                            **params).fit(p["Rval"], p["yval"], max_epochs=self.mcfg["max_epochs"],
                                          patience=self.mcfg["patience"], verbose=True)
        score, prediction, seconds = self._evaluate(model)
        self._save(key, {"kind": "fkge", "rulebase": meta, "parameters": params, "overrides": overrides,
                         "validation_log_loss": model.val_loss, "epochs": model.epochs,
                         "stop_reason": model.stop_reason, "train_seconds": model.seconds,
                         "teacher_seconds": teacher_seconds, "n_rules": len(rb["R"]),
                         "ms_per_query": seconds / len(score) * 1000, "history": model.history,
                         "test": metrics(p["ytest"], prediction, score)}, score, prediction)
        del model, attribution
        gc.collect()

    def fisa(self, rb_key):
        key = f"fisa__{rb_key}"
        rb, meta = self.rulebase(rb_key)
        p = self.p
        model = CompactFISA(rb["R"], rb["y"], rb["weights"])
        Dv, mv = model.log_values(p["Rval"])
        threshold = threshold_on_validation(p["yval"], np.where(mv, Dv[:, 1] - Dv[:, 0], 0.0))
        (D, matched), seconds = timed(lambda: model.log_values(p["Rtest"]))
        score = np.where(matched, D[:, 1] - D[:, 0], 0.0)
        prediction = np.where(matched, (score > threshold).astype(int), model.fallback)
        self._save(key, {"kind": "fisa", "rulebase": meta, "threshold": threshold, "fit_seconds": model.fit_seconds,
                         "n_rules": len(rb["R"]), "ms_per_query": seconds / len(score) * 1000,
                         "matched_rate": float(matched.mean()),
                         "test": metrics(p["ytest"], prediction, score)}, score, prediction)

    def kge(self, method):
        key = f"kge__{method}"
        best, threshold, trials = run_kge(method, {**self.full, "Rval": self.p["Rval"], "yval": self.p["yval"]},
                                          self.ecfg["kge"], self.seed, self.device)
        score, seconds = timed(lambda: best["model"].scores(self.p["Rtest"]))
        prediction = (score > threshold).astype(int)
        self._save(key, {"kind": "kge", "method": method, "selected": {"d": best["d"], "k": best["k"]},
                         "validation_log_loss": best["validation_log_loss"], "threshold": threshold,
                         "train_seconds": best["train_seconds"], "ms_per_query": seconds / len(score) * 1000,
                         "trials": trials, "test": metrics(self.p["ytest"], prediction, score)}, score, prediction)

    def ml(self, method):
        key = f"ml__{method.replace(' ', '_')}"
        with (self.dir / "preprocessing.pkl").open("rb") as stream:
            processing, _ = pickle.load(stream)
        frames = {}
        for name in ("train_original", "validation", "test"):
            ids = pd.read_csv(self.dir / f"{name}_ids.csv", dtype={"image_id": str, "patient_id": str})
            frame = ids[["image_id"]].merge(self.data, on="image_id", how="left", validate="one_to_one")
            if frame[TARGET].isna().any():
                raise ValueError(f"{name} IDs missing from the data")
            frames[name] = frame
        X, y = processing.transform(frames["train_original"]), frames["train_original"][TARGET].to_numpy(int)
        Xv, yv = processing.transform(frames["validation"]), frames["validation"][TARGET].to_numpy(int)
        Xt, yt = processing.transform(frames["test"]), frames["test"][TARGET].to_numpy(int)
        if not np.array_equal(yt, self.p["ytest"]) or not np.array_equal(yv, self.p["yval"]):
            raise AssertionError("Baseline rows are not aligned with the prepared fold")
        model, params, loss, threshold, train_seconds, trials = run_ml(method, X, y, Xv, yv, self.ecfg["ml"], self.seed)
        score, seconds = timed(lambda: model.predict_proba(Xt)[:, 1])
        prediction = (score > threshold).astype(int)
        self._save(key, {"kind": "ml", "method": method, "selected": params, "validation_log_loss": loss,
                         "threshold": threshold, "train_seconds": train_seconds, "train_rows": len(y),
                         "ms_per_query": seconds / len(score) * 1000, "trials": trials,
                         "test": metrics(yt, prediction, score)}, score, prediction)


# ============================================================================ which jobs
def trial_records(directory):
    return [read_json(path) for path in sorted((directory / "trials").glob("trial_[0-9][0-9][0-9].json"))]


def profile(records, names):
    """For each value of the parameters ``names``: the trial with the lowest validation loss."""
    best = {}
    for record in records:
        value = tuple(record["parameters"][n] for n in names)
        if value not in best or record["validation_log_loss"] < best[value]["validation_log_loss"]:
            best[value] = record
    return dict(sorted(best.items()))


def plan(fold_dir, scenarios, ecfg):
    jobs = []
    records = trial_records(fold_dir)
    chosen = read_json(fold_dir / "selected.json")["trial"]
    jobs.append(("trial", chosen))   # re-evaluated main model: consistency check and same-session timing
    if "kb3" in scenarios:
        jobs += [("trial", r["trial"]) for r in profile(records, ["d_e"]).values()]
    if "kb4" in scenarios:
        jobs += [("trial", r["trial"]) for r in profile(records, ["lam_I", "lam_P"]).values()]
        jobs += [("trial", r["trial"]) for r in profile(records, ["alpha"]).values()]
    if "kb2" in scenarios:
        jobs += [("fisa", "full"), ("fisa", "fkgs"), ("fkge", "fkgs", {})]
    if "kb6" in scenarios:
        for r in ecfg["kb6_ratios"]:
            jobs.append(("fisa", ratio_key(r)))
            if r < 1.0:
                jobs.append(("fkge", ratio_key(r), {}))
    if "kb5" in scenarios:
        jobs += [("fkge", "full", {"K": k}) for k in ecfg["kb5_K"] if k != 5]
    if "ablation" in scenarios:
        jobs += [("fkge", "full", ov) for ov in ABLATION.values() if ov]
    if "baseline" in scenarios:
        jobs += [("fisa", "full")] + [("ml", m) for m in ML_METHODS] + [("kge", m) for m in KGE_METHODS]
    unique = {}
    for job in jobs:
        unique.setdefault(job_key(job), job)
    # cheap jobs first; FKG-E on the full graph last
    return sorted(unique.values(), key=lambda j: ["trial", "fisa", "ml", "kge", "fkge"].index(j[0]))


def job_key(job):
    kind = job[0]
    if kind == "trial":
        return f"trial_{job[1]:03}"
    if kind == "fkge":
        return f"fkge__{job[1]}__{ov_key(job[2])}"
    if kind == "ml":
        return f"ml__{job[1].replace(' ', '_')}"
    return f"{kind}__{job[1]}"


# ============================================================================ reports
def load_job(jobs, fold, key):
    path = jobs / f"fold_{fold:02}" / f"{key}.json"
    return read_json(path) if path.exists() else None


METRIC_COLUMNS = ["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity", "f1", "f1_macro",
                  "accuracy"]


def _row(job, **extra):
    return {**extra, **{m: job["test"][m] for m in METRIC_COLUMNS}, "ms_per_query": job.get("ms_per_query"),
            "validation_log_loss": job.get("validation_log_loss")}


def summarize(frame, by):
    numeric = [c for c in frame.columns if c not in {"fold", "trial", "is_selected", *by}
               and pd.api.types.is_numeric_dtype(frame[c])]
    stats = frame.groupby(by, sort=False)[numeric].agg(["mean", "std"])
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    counts = frame.groupby(by, sort=False).fold.nunique().rename("n_folds")
    return stats.join(counts).reset_index()


LABELS = {"roc_auc": "ROC-AUC", "average_precision": "AP", "balanced_accuracy": "BalAcc", "sensitivity": "Sens",
          "specificity": "Spec", "validation_log_loss": "Val loss", "n_rules": "|R|", "fisa_auc": "FISA AUC",
          "fisa_bacc": "FISA BalAcc", "fisa_ms": "FISA ms", "fkge_auc": "FKG-E AUC", "fkge_bacc": "FKG-E BalAcc",
          "fkge_ms": "FKG-E ms", "speedup": "Tăng tốc", "rep": "Rep", "n_folds": "Số fold"}


def table_image(frame, path, title):
    """Table whose column widths follow their longest cell, so long Vietnamese labels are not clipped."""
    import matplotlib.pyplot as plt
    shown = frame.map(lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v))
    header = [str(c) for c in shown.columns]
    chars = np.array([max([len(h)] + [len(x) for x in shown[c]]) for h, c in zip(header, shown.columns)], float) + 2
    fig, ax = plt.subplots(figsize=(max(6.0, chars.sum() * 0.095), 1.4 + 0.42 * len(shown)))
    ax.axis("off")
    ax.set_title(title, pad=12)
    table = ax.table(cellText=shown.values, colLabels=header, loc="center", cellLoc="center",
                     colWidths=list(chars / chars.sum()))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render(output, scope, ecfg, folds):
    import matplotlib.pyplot as plt
    jobs, report = output / "extra" / "jobs", output / "extra" / "reports"
    report.mkdir(parents=True, exist_ok=True)
    notes = [f"Scope: {scope}", "Mean and sample SD (ddof=1) across outer folds; n_folds is shown per row.",
             "Every configuration was locked on validation; test metrics never chose anything.",
             "KB3/KB4: sensitivity of the main grid; each value uses its best-validation trial."]
    tables = {}

    def emit(name, frame, by, title, columns):
        if frame.empty:
            return
        frame.to_csv(report / f"{name}_by_fold.csv", index=False)
        stats = summarize(frame, by)
        stats.to_csv(report / f"{name}_mean_std.csv", index=False)
        shown = stats[by].copy()
        for c in columns:
            shown[LABELS.get(c, c)] = [f"{m:.4g} ± {d:.2g}" if np.isfinite(d) else f"{m:.4g}"
                                       for m, d in zip(stats[f"{c}_mean"], stats[f"{c}_std"])]
        shown[LABELS["n_folds"]] = stats.n_folds
        table_image(shown, report / f"{name}_mean_std.png", f"{scope}: {title}")
        tables[name] = stats
        n = int(stats.n_folds.min())
        notes.append(f"{name}: {n} outer fold(s) complete" + ("" if n == 5 else " — INCOMPLETE, do not report"))

    main_cols = ["roc_auc", "average_precision", "balanced_accuracy", "sensitivity", "specificity"]
    # consistency: re-evaluated selected trial must reproduce run.py's held-out metrics
    for fold in folds:
        d = output / f"fold_{fold:02}"
        chosen = read_json(d / "selected.json")["trial"]
        job = load_job(jobs, fold, f"trial_{chosen:03}")
        if job:
            main = read_json(d / "results.json")["FKG-E"]
            if any(abs(job["test"][m] - main[m]) > 1e-9 for m in ("roc_auc", "balanced_accuracy")):
                raise AssertionError(f"fold {fold}: re-evaluated selected model differs from results.json")
    # KB2
    rows = []
    for fold in folds:
        need = [load_job(jobs, fold, k) for k in ("fisa__full", "fisa__fkgs", "trial_%03d" % read_json(
            output / f"fold_{fold:02}/selected.json")["trial"], "fkge__fkgs__default")]
        if all(need):
            ff, fs, ef, es = need
            for tag, F, E in (("FKG-MM đầy đủ", ff, ef), ("FKGS θ=%g" % ecfg["sfkgs_theta"], fs, es)):
                rows.append({"fold": fold, "rule_base": tag, "n_rules": F["n_rules"],
                             "rep": F["rulebase"].get("rep", 1.0), "fisa_auc": F["test"]["roc_auc"],
                             "fisa_bacc": F["test"]["balanced_accuracy"], "fisa_ms": F["ms_per_query"],
                             "fkge_auc": E["test"]["roc_auc"], "fkge_ap": E["test"]["average_precision"],
                             "fkge_bacc": E["test"]["balanced_accuracy"], "fkge_ms": E["ms_per_query"],
                             "speedup": F["ms_per_query"] / E["ms_per_query"]})
    emit("kb2_fkgs", pd.DataFrame(rows), ["rule_base"], "KB2 FKG-MM vs FKGS",
         ["n_rules", "fisa_auc", "fkge_auc", "fkge_bacc", "speedup"])
    # KB3 / KB4 / alpha profile
    for name, names, title in (("kb3_d_e", ["d_e"], "KB3 chiều nhúng d_e"),
                               ("kb4_lambda", ["lam_I", "lam_P"], "KB4 (λ_I, λ_P)"),
                               ("profile_alpha", ["alpha"], "Hồ sơ α")):
        rows = []
        for fold in folds:
            d = output / f"fold_{fold:02}"
            prof = profile(trial_records(d), names)
            got = {v: load_job(jobs, fold, f"trial_{r['trial']:03}") for v, r in prof.items()}
            if all(got.values()):
                for v, job in got.items():
                    extra = dict(zip(names, v))
                    if name == "kb4_lambda":
                        extra["nu"] = v[0] / v[1]
                    rows.append({"fold": fold, **extra, "is_selected": job["parameters"] == read_json(
                        d / "selected.json")["parameters"], "trial": int(job["job"][6:]), **_row(job),
                        "epochs": job["epochs"]})
        frame = pd.DataFrame(rows)
        emit(name, frame, names + (["nu"] if name == "kb4_lambda" else []), title,
             ["validation_log_loss", "roc_auc", "balanced_accuracy"])
        if name == "kb4_lambda" and not frame.empty:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            stats = tables[name]
            for ax, column in zip(axes, ("validation_log_loss_mean", "roc_auc_mean", "balanced_accuracy_mean")):
                grid = stats.pivot(index="lam_I", columns="lam_P", values=column)
                image = ax.imshow(grid.values, cmap="viridis", origin="lower")
                ax.set(xticks=range(grid.shape[1]), xticklabels=grid.columns, yticks=range(grid.shape[0]),
                       yticklabels=grid.index, xlabel="λ_P", ylabel="λ_I", title=column.replace("_mean", ""))
                for (i, j), value in np.ndenumerate(grid.values):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="w", fontsize=8)
                fig.colorbar(image, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(report / "kb4_heatmap.png", dpi=160)
            plt.close(fig)
    # KB5
    rows = []
    for fold in folds:
        chosen = read_json(output / f"fold_{fold:02}/selected.json")["trial"]
        got = {k: load_job(jobs, fold, f"trial_{chosen:03}" if k == 5 else f"fkge__full__K={k}") for k in ecfg["kb5_K"]}
        if all(got.values()):
            rows += [{"fold": fold, "K": k, **_row(job), "epochs": job["epochs"]} for k, job in got.items()]
    emit("kb5_K", pd.DataFrame(rows), ["K"], "KB5 số mẫu âm K", ["validation_log_loss", "roc_auc", "balanced_accuracy"])
    # KB6
    rows = []
    for fold in folds:
        chosen = read_json(output / f"fold_{fold:02}/selected.json")["trial"]
        for r in ecfg["kb6_ratios"]:
            F = load_job(jobs, fold, f"fisa__{ratio_key(r)}")
            E = load_job(jobs, fold, f"trial_{chosen:03}" if r >= 1.0 else f"fkge__{ratio_key(r)}__default")
            if F and E:
                rows.append({"fold": fold, "ratio": r, "n_rules": F["n_rules"], "fisa_ms": F["ms_per_query"],
                             "fkge_ms": E["ms_per_query"], "fisa_auc": F["test"]["roc_auc"],
                             "fkge_auc": E["test"]["roc_auc"], "fisa_bacc": F["test"]["balanced_accuracy"],
                             "fkge_bacc": E["test"]["balanced_accuracy"]})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame[frame.groupby("fold").ratio.transform("count") == len(ecfg["kb6_ratios"])]
    emit("kb6_rules", frame, ["ratio"], "KB6 thời gian suy diễn theo |R|", ["n_rules", "fisa_ms", "fkge_ms", "fkge_auc"])
    if "kb6_rules" in tables:
        stats = tables["kb6_rules"]
        fig, ax = plt.subplots(figsize=(6, 4))
        for column, label in (("fisa_ms", "FISA"), ("fkge_ms", "FKG-E")):
            ax.errorbar(stats.n_rules_mean, stats[f"{column}_mean"], yerr=stats[f"{column}_std"], marker="o",
                        capsize=3, label=label)
        ax.set(xlabel="|R| (luật duy nhất)", ylabel="ms / truy vấn", title="KB6")
        ax.legend()
        fig.tight_layout()
        fig.savefig(report / "kb6_time.png", dpi=160)
        plt.close(fig)
    # Ablation
    rows = []
    for fold in folds:
        chosen = read_json(output / f"fold_{fold:02}/selected.json")["trial"]
        got = {name: load_job(jobs, fold, f"trial_{chosen:03}" if not ov else f"fkge__full__{ov_key(ov)}")
               for name, ov in ABLATION.items()}
        if all(got.values()):
            rows += [{"fold": fold, "variant": name, **_row(job), "epochs": job["epochs"]} for name, job in got.items()]
    emit("ablation", pd.DataFrame(rows), ["variant"], "Ablation FKG-E", main_cols)
    # Baselines
    rows, predictions = [], []
    names = [("FISA (Chương 2)", "fisa__full")] + [(f"{m} + kNN", f"kge__{m}") for m in KGE_METHODS] + \
            [(m, f"ml__{m.replace(' ', '_')}") for m in ML_METHODS]
    for fold in folds:
        chosen = read_json(output / f"fold_{fold:02}/selected.json")["trial"]
        got = {label: load_job(jobs, fold, key) for label, key in names + [("FKG-E (đề xuất)", f"trial_{chosen:03}")]}
        if all(got.values()):
            for label, job in got.items():
                rows.append({"fold": fold, "method": label, **_row(job), "train_seconds": job.get("train_seconds"),
                             "selected": json.dumps(job.get("selected", job.get("parameters")), ensure_ascii=False)})
                frame = pd.read_csv(jobs / f"fold_{fold:02}" / f"{job['job']}.csv", dtype={"image_id": str})
                predictions.append(frame.assign(fold=fold, method=label))
    emit("baseline", pd.DataFrame(rows), ["method"], "So sánh baseline", main_cols)
    if predictions:
        oof = pd.concat(predictions, ignore_index=True)
        pooled = [{"method": label, "n_folds": g.fold.nunique(), **metrics(g[TARGET], g.prediction, g.score)}
                  for label, g in oof.groupby("method", sort=False)]
        pd.DataFrame(pooled).to_csv(report / "baseline_pooled_oof.csv", index=False)
        oof.to_csv(report / "baseline_predictions_oof.csv", index=False)
    (report / "README.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes), flush=True)


# ============================================================================ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Finished (or partially finished) run.py output")
    parser.add_argument("--config", type=Path, default=ROOT / "config_extra.json")
    parser.add_argument("--only", default=",".join(SCENARIOS), help="Comma list of " + ",".join(SCENARIOS))
    parser.add_argument("--stage", choices=["all", "report"], default="all")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    scenarios = [s.strip() for s in args.only.split(",") if s.strip()]
    if not set(scenarios) <= set(SCENARIOS):
        parser.error(f"--only must be a subset of {SCENARIOS}")
    main_manifest = read_json(output / "run_manifest.json")
    mid = main_manifest["identity"]
    ecfg = read_json(args.config)
    folds = [f for f in args.folds if (output / f"fold_{f:02}" / "results.json").exists()]
    skipped = sorted(set(args.folds) - set(folds))
    if skipped:
        print(f"Skipping folds without a finished main run: {skipped}", flush=True)
    if args.stage == "report":
        render(output, mid["scope"], ecfg, folds)
        return
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    seed = mid["seed"]
    code = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob("*.py"))
            if not any(part in {"outputs", "private", "__pycache__"} for part in p.relative_to(ROOT).parts)}
    identity = {"config_extra": ecfg, "main_identity_sha256": hashlib.sha256(
        json.dumps(mid, sort_keys=True).encode()).hexdigest(), "code_sha256": code, "device": device,
        "block_size": args.block_size, "seed": seed}
    manifest_path = output / "extra" / "extra_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if not args.resume or manifest["identity"] != identity:
            raise SystemExit("extra/ already exists: rerun with --resume and the same code/config/device, "
                             "or move extra/ away")
        manifest.setdefault("resume_events", []).append({"argv": [sys.executable, *sys.argv], "hardware": hardware(device)})
    else:
        manifest = {"identity": identity, "hardware": hardware(device), "argv": [sys.executable, *sys.argv],
                    "selection": "every extra configuration starts from fold_XX/selected.json (validation only)"}
    write_json(manifest_path, manifest)
    if mid["scope"] == "SYNTHETIC_SMOKE_ONLY":
        data, _ = synthetic_data(seed)
    else:
        data = pd.read_csv(ROOT / "data/fusion_features.csv", dtype={"image_id": str, "patient_id": str},
                           float_precision="round_trip")
        digest = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
        if digest != mid["data_sha256"]:
            raise SystemExit("data/fusion_features.csv differs from the data of the main run")
    torch.manual_seed(seed)
    np.random.seed(seed)
    main_cache = Path(main_manifest["temporary_teacher_cache"]) if main_manifest.get("temporary_teacher_cache") else None
    cache_root = Path(tempfile.gettempdir()) / "fkge_brset_extra" / hashlib.sha256(
        (str(output) + json.dumps(identity, sort_keys=True)).encode()).hexdigest()[:20]
    manifest["temporary_teacher_cache"] = str(cache_root)
    write_json(manifest_path, manifest)
    for fold in folds:
        F = Fold(output, fold, ecfg, mid["config"], seed, device, args.block_size, cache_root, main_cache, data)
        todo = plan(F.dir, scenarios, ecfg)
        for n, job in enumerate(todo, 1):
            key = job_key(job)
            if F.done(key):
                continue
            started = time.time()
            if job[0] == "trial":
                F.trial(job[1])
            elif job[0] == "fkge":
                F.fkge(job[1], job[2])
            else:
                getattr(F, job[0])(job[1])
            print(f"fold {fold}: [{n}/{len(todo)}] {key} done in {time.time() - started:.0f}s", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
        render(output, mid["scope"], ecfg, folds)
    manifest["completed"] = {f"fold_{f:02}": all((output / "extra/jobs" / f"fold_{f:02}" / f"{job_key(j)}.json").exists()
                                                 for j in plan(output / f"fold_{f:02}", scenarios, ecfg)) for f in folds}
    write_json(manifest_path, manifest)
    print(f"DONE: {output / 'extra'}", flush=True)


if __name__ == "__main__":
    main()
