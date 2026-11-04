"""ROC / Precision-Recall figures for Chapter 3 from <out>/predictions/fold_XX.csv.gz (one column per model).

Curves pool the out-of-fold scores of the 5 outer folds (every image exactly once); the legend gives the pooled AP.
Scores of different folds come from different models, so pooled curves and fold-mean AP (Bảng 7) are reported
separately and need not coincide. Random-classifier reference: precision = positive rate.
Colors are fixed per model (validated categorical slots of the dataviz reference palette, light mode).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FKG_UM = [("M1|FKG-MM (đề xuất)|FKG", "FKG-MM (đề xuất)", SLOT[0]),
          ("M1|FKG-UM (Bảng)|FKG", "FKG-UM (Bảng)", SLOT[1]),
          ("M1|FKG-UM (Ảnh)|FKG", "FKG-UM (Ảnh)", SLOT[2])]
NON_GRAPH = [("M1|FKG-MM (đề xuất)|FKG", "FKG-MM (đề xuất)", SLOT[0]),
             ("M1|Logistic Regression|Tabular", "Logistic Regression (bảng)", SLOT[3]),
             ("M1|XGBoost|Tabular", "XGBoost (bảng)", SLOT[4]),
             ("M2|Early Fusion (MLP)|Multimodal", "Early Fusion (MLP)", SLOT[5]),
             ("M2|Late Fusion (Ensemble)|Multimodal", "Late Fusion (Ensemble)", SLOT[6])]
OPERATORS = [("M5|FKG-MM (đề xuất)|FKG", "Φsel (đề xuất)", SLOT[0]),
             ("M5|FKG-MM (Filter)|FKG", "Filter", SLOT[1]),
             ("M5|FKG-MM (Hadamard)|FKG", "Hadamard", SLOT[2]),
             ("M5|FKG-MM (Tensor)|FKG", "Tensor (TT 3.5)", SLOT[3]),
             ("M5|FKG-MM (Wrapper)|FKG", "Wrapper", SLOT[4])]
AP_ROWS = [k for k, _, _ in FKG_UM] + [k for k, _, _ in NON_GRAPH[1:]] + [k for k, _, _ in OPERATORS[1:]]


def vn(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def load_predictions(results):
    files = sorted(glob.glob(os.path.join(results, "predictions", "fold_*.csv.gz")))
    if not files:
        return None
    frames = [pd.read_csv(f, dtype={"image_id": str, "patient_id": str}).assign(fold=os.path.basename(f)[:7]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if df.image_id.duplicated().any() or (df.groupby("patient_id").fold.nunique() > 1).any():
        raise AssertionError("Out-of-fold predictions repeat an image or split a patient across folds")
    return df


def _style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set(xlim=(0, 1), ylim=(0, 1.01))
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    from matplotlib.ticker import FuncFormatter
    comma = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax.xaxis.set_major_formatter(comma)
    if ylabel:
        ax.yaxis.set_major_formatter(comma)


def _curves(ax_roc, ax_pr, df, models, title_roc, title_pr):
    y, prev = df["y"].to_numpy(), df["y"].mean()
    for key, label, color in models:
        if key not in df:
            continue
        s = df[key].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, s)
        p, r, _ = precision_recall_curve(y, s)
        ax_roc.plot(fpr, tpr, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round",
                    label=f"{label} · AUC {vn(roc_auc_score(y, s))}")
        ax_pr.plot(r, p, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round",
                   label=f"{label} · AP {vn(average_precision_score(y, s))}")
    ax_roc.plot([0, 1], [0, 1], color=INK2, lw=1, alpha=0.6)
    ax_pr.axhline(prev, color=INK2, lw=1, alpha=0.6)
    # below the reference line, at the left: no curve runs there (precision < positive rate)
    ax_pr.text(0.015, prev - 0.012, f"ngẫu nhiên = tỉ lệ dương {vn(100 * prev, 1)}%", ha="left", va="top",
               fontsize=7.5, color=INK2)
    _style(ax_roc, "Tỉ lệ dương tính giả (1 − Spec.)", "Độ nhạy (Sens.)")
    _style(ax_pr, "Recall (độ nhạy)", "Precision")
    ax_roc.set_title(title_roc, fontsize=10, color=INK, loc="left")
    ax_pr.set_title(title_pr, fontsize=10, color=INK, loc="left")
    for ax in (ax_roc, ax_pr):
        ax.legend(fontsize=7.5, frameon=False, loc="lower right" if ax is ax_roc else "upper right", labelcolor=INK)


def pooled_table(df, scenario_rows):
    rows = []
    y = df["y"].to_numpy()
    for key in [c for c in df.columns if "|" in c]:
        s = df[key].to_numpy(float)
        p, r, _ = precision_recall_curve(y, s)
        per_fold = df.groupby("fold").apply(lambda g: average_precision_score(g["y"], g[key]), include_groups=False)
        sc, model, data = key.split("|")
        rows.append({"scenario": sc, "model": model, "data": data, "n_images": len(y), "n_folds": df.fold.nunique(),
                     "pooled_auc_roc": roc_auc_score(y, s), "pooled_ap": average_precision_score(y, s),
                     "pooled_pr_auc_trap": auc(r, p), "fold_mean_ap": per_fold.mean(), "fold_sd_ap": per_fold.std(ddof=1),
                     "pos_rate": y.mean()})
    return pd.DataFrame(rows)


def render_plots(results):
    """Writes PNG (200 dpi) + PDF figures and pooled_oof.csv into <results>/hinh/. Returns written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load_predictions(results)
    if df is None:
        return []
    out = os.path.join(results, "hinh")
    os.makedirs(out, exist_ok=True)
    written = []
    pooled_table(df, None).to_csv(os.path.join(out, "pooled_oof.csv"), index=False)
    written.append(os.path.join(out, "pooled_oof.csv"))
    nf = df.fold.nunique()

    def save(fig, name):
        for ext in ("png", "pdf"):
            path = os.path.join(out, f"{name}.{ext}")
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            written.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9.4))
    _curves(axes[0, 0], axes[0, 1], df, FKG_UM, "(a) ROC – FKG-MM và FKG đơn mô thức", "(b) PR – FKG-MM và FKG đơn mô thức")
    _curves(axes[1, 0], axes[1, 1], df, NON_GRAPH, "(c) ROC – FKG-MM và mô hình không dùng đồ thị",
            "(d) PR – FKG-MM và mô hình không dùng đồ thị")
    fig.suptitle(f"KB-M1, KB-M2: đường ROC và Precision–Recall gộp ngoài fold ({nf} fold, {len(df)} ảnh)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "roc_pr_kb_m1_m2")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    _curves(axes[0], axes[1], df, OPERATORS, "(a) ROC – năm toán tử dung hợp", "(b) PR – năm toán tử dung hợp")
    fig.suptitle(f"KB-M5: đường ROC và Precision–Recall gộp ngoài fold ({nf} fold)", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "roc_pr_kb_m5")

    names = {k: l for k, l, _ in FKG_UM + NON_GRAPH + OPERATORS}
    names["M5|FKG-MM (đề xuất)|FKG"] = "Φsel (đề xuất)"
    keys = [k for k in AP_ROWS if k in df]
    per = {k: df.groupby("fold").apply(lambda g, k=k: average_precision_score(g["y"], g[k]), include_groups=False)
           for k in keys}
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(keys) + 1.4))
    for i, k in enumerate(keys[::-1]):
        v = per[k].to_numpy()
        ax.scatter(v, np.full(len(v), i), s=22, color=SLOT[0], alpha=0.45, linewidths=0, zorder=3)
        m, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
        ax.plot([m - sd, m + sd], [i, i], color=SLOT[0], lw=2, solid_capstyle="round", zorder=4)
        ax.scatter([m], [i], s=46, color=SLOT[0], edgecolors=SURFACE, linewidths=1.5, zorder=5)
        # label right of every mark of the row (fold dots can lie outside mean ± SD)
        ax.text(max(v.max(), m + sd) + 0.02, i, f"{vn(m)} ± {vn(sd)}", va="center", fontsize=7.5, color=INK2)
    prev = df["y"].mean()
    ax.axvline(prev, color=INK2, lw=1, alpha=0.6)
    ax.text(prev + 0.008, len(keys) - 0.4, f"ngẫu nhiên {vn(100 * prev, 1)}%", fontsize=7.5, color=INK2, va="bottom")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([names.get(k, k.split("|")[1]) for k in keys[::-1]], fontsize=8, color=INK)
    _style(ax, "AP (average precision) trên fold kiểm tra", "")
    ax.set(xlim=(0, 1), ylim=(-0.7, len(keys) - 0.1))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"AP theo fold: điểm nhạt = từng fold, điểm đậm ± đoạn = trung bình ± SD ({nf} fold)",
                 fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "ap_theo_fold")
    return written
