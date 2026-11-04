"""Độ đo, chọn ngưỡng trên tập xác thực và kiểm định thống kê."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)


def choose_threshold(score_val: np.ndarray, y_val: np.ndarray, criterion: str = "balacc") -> float:
    """Ngưỡng τ* trên điểm (ví dụ log D1 - log D0) chọn trên tập xác thực.
    criterion: "balacc" (= Youden J) hoặc "f1"."""
    s = np.asarray(score_val, float)
    cand = np.unique(s)
    if len(cand) > 2000:
        cand = np.unique(np.quantile(s, np.linspace(0, 1, 2001)))
    cand = np.concatenate([[-np.inf], (cand[:-1] + cand[1:]) / 2, [np.inf]]) if len(cand) > 1 else np.array([cand[0]])
    best, best_t = -1.0, 0.0
    P = y_val == 1; N = ~P
    for t in cand:
        pred = s > t
        if criterion == "f1":
            v = f1_score(y_val, pred, zero_division=0)
        else:
            sens = pred[P].mean() if P.any() else 0
            spec = (~pred[N]).mean() if N.any() else 0
            v = (sens + spec) / 2
        if v > best:
            best, best_t = v, t
    return float(best_t)


def binary_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    finite = np.where(np.isfinite(score), score, np.sign(score) * 1e12)
    try:
        auc = roc_auc_score(y, finite); ap = average_precision_score(y, finite)
    except ValueError:
        auc = ap = np.nan
    return {"acc": (tp + tn) / len(y), "sens": sens, "spec": spec, "prec": prec,
            "f1": f1_score(y, pred, zero_division=0), "balacc": np.nanmean([sens, spec]),
            "mcc": matthews_corrcoef(y, pred), "auc_roc": auc, "auc_pr": ap,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def summarize(df: pd.DataFrame, by: List[str], metrics: List[str]) -> pd.DataFrame:
    """Trung bình ± SD và khoảng tin cậy 95% theo t (df = số phép đo - 1)."""
    rows = []
    for key, g in df.groupby(by, sort=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["n"] = len(g)
        for m in metrics:
            v = g[m].astype(float).dropna().to_numpy()
            if len(v) == 0:
                continue
            mu, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
            h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else np.nan
            row[m] = mu; row[m + "_sd"] = sd; row[m + "_ci_lo"] = mu - h; row[m + "_ci_hi"] = mu + h
        rows.append(row)
    return pd.DataFrame(rows)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def nadeau_bengio(diff: np.ndarray, n_train: float, n_test: float) -> float:
    """Kiểm định t hiệu chỉnh cho k-fold lặp (Nadeau & Bengio 2003): p hai phía."""
    d = np.asarray(diff, float); J = len(d)
    if J < 2 or d.std(ddof=1) == 0:
        return np.nan
    var = d.var(ddof=1) * (1 / J + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return float(2 * stats.t.sf(abs(t), J - 1))


def paired_tests(df: pd.DataFrame, ref: str, metric: str, method_col: str = "method",
                 unit_cols=("repeat", "fold"), n_train=None, n_test=None) -> pd.DataFrame:
    piv = df.pivot_table(index=list(unit_cols), columns=method_col, values=metric)
    rows = []
    for m in piv.columns:
        if m == ref:
            continue
        d = (piv[ref] - piv[m]).dropna().to_numpy()
        try:
            pw = stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
        except ValueError:
            pw = np.nan
        pnb = nadeau_bengio(d, n_train, n_test) if n_train and n_test else np.nan
        rows.append({"ref": ref, "vs": m, "metric": metric, "n_pairs": len(d),
                     "mean_diff": d.mean() if len(d) else np.nan, "p_wilcoxon": pw, "p_nb": pnb})
    out = pd.DataFrame(rows)
    if len(out):
        for c in ("p_wilcoxon", "p_nb"):
            ok = out[c].notna()
            out.loc[ok, c + "_holm"] = holm(out.loc[ok, c].to_numpy())
    return out


def friedman_ranks(df: pd.DataFrame, metric: str, method_col="method", unit_cols=("repeat", "fold")):
    piv = df.pivot_table(index=list(unit_cols), columns=method_col, values=metric).dropna()
    ranks = piv.rank(axis=1, ascending=False)
    p = stats.friedmanchisquare(*[piv[c] for c in piv.columns]).pvalue if piv.shape[1] >= 3 and len(piv) >= 2 else np.nan
    return ranks.mean().sort_values(), p


def kendall_w(df: pd.DataFrame, metric: str, method_col="method", unit_cols=("repeat", "fold")) -> float:
    """Hệ số đồng thuận Kendall's W của thứ hạng các phương pháp giữa các fold (0: không, 1: hoàn toàn)."""
    piv = df.pivot_table(index=list(unit_cols), columns=method_col, values=metric).dropna()
    m, n = piv.shape
    if m < 2 or n < 2:
        return np.nan
    R = piv.rank(axis=1, ascending=False).sum(axis=0).to_numpy()
    S = ((R - R.mean()) ** 2).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n)))
