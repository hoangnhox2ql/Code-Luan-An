"""Báo cáo kết quả FISA trung bình nhân: Bảng 7, Bảng 8 và phân tích theo câu hỏi KB-M1–KB-M6.
Mỗi ô: trung bình ± độ lệch chuẩn qua các fold (%). Kiểm định cặp: Δ (điểm %), số fold thắng,
p của t hiệu chỉnh Nadeau–Bengio (kiểm định chính), Wilcoxon (với n = 5, p nhỏ nhất là 0,0625)."""
import os

import numpy as np
import pandas as pd
from scipy import stats

COLS = [("acc", "Acc"), ("f1", "F1"), ("auc_roc", "AUC"), ("auc_pr", "AP"), ("pr_auc_trap", "PR-AUC"), ("sens", "Sens."),
        ("spec", "Spec."), ("balacc", "BalAcc")]


def _ms(g, m):
    v = g[m].astype(float).dropna() if g is not None and m in g else pd.Series(dtype=float)
    return "–" if v.empty else f"{100 * v.mean():.1f} ± {100 * v.std(ddof=1):.1f}"


# Nadeau-Bengio corrected t: n_te/n_tr of the folds actually used (set in build() from audit_data.csv).
# The source hard-coded n_tr=1778, n_te=242 (the former ~1,500-image data); with k = 5 folds n_te/n_tr ~ 1/4.
_NB = {"n_tr": 1778, "n_te": 242}


def _pair(a, b, m="auc_roc", n_tr=None, n_te=None):
    if a is None or b is None:
        return None
    n_tr, n_te = n_tr or _NB["n_tr"], n_te or _NB["n_te"]
    c = a.index.intersection(b.index); d = (a.loc[c, m] - b.loc[c, m]).to_numpy(float)
    if len(d) < 2:
        return None
    sd = d.std(ddof=1)
    p_nb = 2 * stats.t.sf(abs(d.mean() / np.sqrt(sd ** 2 * (1 / len(d) + n_te / n_tr))), len(d) - 1) if sd > 0 else np.nan
    try:
        p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
    except ValueError:
        p_w = np.nan
    return {"d": d.mean(), "w": int((d > 0).sum()), "n": len(d), "p": p_nb, "pw": p_w}


def _txt(r):
    if r is None:
        return "thiếu dữ liệu"
    f = lambda x: "–" if np.isnan(x) else f"{x:.3f}"
    return f"Δ = {100 * r['d']:+.1f}; thắng {r['w']}/{r['n']}; p = {f(r['p'])}; p_W = {f(r['pw'])}"


def build(out):
    df = pd.read_csv(os.path.join(out, "per_fold.csv"))
    au = pd.read_csv(os.path.join(out, "audit_data.csv")) if os.path.exists(os.path.join(out, "audit_data.csv")) else None
    if au is not None:
        _NB.update(n_tr=float(au.n_train_orig.mean()), n_te=float(au.n_test.mean()))
    g = lambda sc, model, data=None: (lambda q: q.set_index("fold") if len(q) else None)(
        df[(df.scenario == sc) & (df.model == model) & ((df.data == data) if data is not None else True)])
    nf = df.fold.nunique(); maj = 100 * df.acc_majority.mean()
    import json
    man = os.path.join(out, "run_manifest.json")
    fisa_name = json.load(open(man, encoding="utf-8")).get("fisa", "") if os.path.exists(man) else ""
    title = "FISA-GM-w (trung bình nhân có trọng số bộ ba)" if fisa_name.startswith("FISA-GM-w") else "FISA trung bình nhân"
    rb = json.load(open(man, encoding="utf-8")).get("rule_base", "") if os.path.exists(man) else ""
    ff = json.load(open(man, encoding="utf-8")).get("fcm_fit", "") if os.path.exists(man) else ""
    rb = f"{rb}; hàm thuộc FCM ước lượng trên {ff}" if ff else rb
    L = [f"# FKG-MM với {title} – 5 fold theo bệnh nhân", "", *([f"Cơ sở luật FKG: {rb}.", ""] if rb else []),
         f"{nf} fold; bệnh nhân trùng train/test = {int(au.patient_overlap.sum()) if au is not None else '–'}; "
         f"Acc của bộ luôn đoán âm = {maj:.1f}% (tỉ lệ dương ≈ {100 - maj:.1f}%) – Acc chỉ là độ đo phụ.",
         f"AP = average precision (sklearn); PR-AUC = diện tích dưới đường Precision–Recall theo quy tắc hình thang; "
         f"mốc của bộ đoán ngẫu nhiên cho cả hai là tỉ lệ dương ≈ {100 - maj:.1f}%.", ""]
    # ---------------- Bảng 7
    rows7 = [("Logistic Regression", "Tabular", g("M1", "Logistic Regression", "Tabular")),
             ("Random Forest", "Tabular", g("M1", "Random Forest", "Tabular")),
             ("XGBoost", "Tabular", g("M1", "XGBoost", "Tabular")), ("MLP", "Tabular", g("M1", "MLP", "Tabular")),
             ("MLP (đặc trưng ảnh)", "Image", g("M1", "MLP", "Image")),
             ("Early Fusion (MLP)", "Multimodal", g("M2", "Early Fusion (MLP)")),
             ("Late Fusion (Ensemble)", "Multimodal", g("M2", "Late Fusion (Ensemble)")),
             ("Crisp KG (không mờ)", "Multimodal (KG)", g("M4", "Multimodal + Graph, không mờ (Crisp KG)")),
             ("FKG-UM (Ảnh)", "Unimodal FKG", g("M1", "FKG-UM (Ảnh)")), ("FKG-UM (Bảng)", "Unimodal FKG", g("M1", "FKG-UM (Bảng)")),
             ("**FKG-MM (đề xuất)**", "**Multimodal FKG**", g("M1", "FKG-MM (đề xuất)"))]
    L += ["## Bảng 7: So sánh tổng hợp với các phương pháp cơ sở", "",
          "| Method | Modality | " + " | ".join(n for _, n in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for name, mod, q in rows7:
        L.append(f"| {name} | {mod} | " + (" | ".join(_ms(q, m) for m, _ in COLS) if q is not None else "chưa chạy" + " |" * (len(COLS) - 1)) + " |")
    L += ["", "_ResNet-50, EfficientNet-B0, Joint Fusion, Cross-modal Attention cần ảnh gốc – không có trong bộ dữ liệu này._"]
    # ---------------- Bảng 8
    abl = [("Image-only", "✓", "–", "✓", "✓", "Image-only (FKG-UM Ảnh)"), ("Tabular-only", "–", "✓", "✓", "✓", "Tabular-only (FKG-UM Bảng)"),
           ("Multimodal w/o FKG (Early Fusion MLP)", "✓", "✓", "–", "–", "Multimodal w/o FKG (Early Fusion MLP)"),
           ("Multimodal + Graph, không mờ (Crisp KG)", "✓", "✓", "–", "✓", "Multimodal + Graph, không mờ (Crisp KG)"),
           ("Multimodal + Mờ, không đồ thị", "✓", "✓", "✓", "–", "Multimodal + Mờ, không đồ thị"),
           ("**Full FKG-MM**", "✓", "✓", "✓", "✓", "Full FKG-MM")]
    L += ["", "## Bảng 8: Ablation study mở rộng của FKG-MM", "",
          "| Variant | Image | Tabular | Fuzzy | Graph | Accuracy | AUC | AP | PR-AUC | BalAcc | F1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, i, t, fz, gp, key in abl:
        q = g("M4", key)
        L.append(f"| {name} | {i} | {t} | {fz} | {gp} | " + " | ".join(_ms(q, m) for m in ("acc", "auc_roc", "auc_pr", "pr_auc_trap", "balacc", "f1")) + " |")
    # ---------------- câu hỏi
    mm = g("M1", "FKG-MM (đề xuất)")
    L += ["", "## Trả lời câu hỏi thực nghiệm (AUC là độ đo chính; BalAcc bổ sung)", ""]
    L += ["### KB-M1 – FKG-MM có vượt đơn mô thức?", ""]
    for nm, data in (("FKG-UM (Bảng)", None), ("FKG-UM (Ảnh)", None), ("Logistic Regression", "Tabular"),
                     ("MLP", "Tabular"), ("MLP", "Image")):
        q = g("M1", nm, data)
        L.append(f"- FKG-MM − {nm}{' (' + data + ')' if data else ''}: AUC {_txt(_pair(mm, q))}; "
                 f"AP {_txt(_pair(mm, q, 'auc_pr'))}; BalAcc {_txt(_pair(mm, q, 'balacc'))}")
    L += ["", "### KB-M2 – cấu trúc FKG so với hợp nhất không dùng FKG", ""]
    for nm in ("Early Fusion (MLP)", "Late Fusion (Ensemble)"):
        q = g("M2", nm)
        L.append(f"- FKG-MM − {nm}: AUC {_txt(_pair(mm, q))}; AP {_txt(_pair(mm, q, 'auc_pr'))}; "
                 f"BalAcc {_txt(_pair(mm, q, 'balacc'))}")
    m3 = df[df.scenario == "M3"]
    if len(m3):
        L += ["", "### KB-M3 – số mức ngôn ngữ", "", "| Số mức | Acc | AUC | AP | PR-AUC | BalAcc | Sens. |",
              "|---|---|---|---|---|---|---|"]
        for nm, q in m3.groupby("model", sort=False):
            L.append(f"| {nm} | " + " | ".join(_ms(q, m) for m in ("acc", "auc_roc", "auc_pr", "pr_auc_trap", "balacc", "sens")) + " |")
        ref = g("M3", "FKG-MM, 5 mức")
        for nm in m3.model.unique():
            if nm != "FKG-MM, 5 mức":
                L.append(f"- {nm} − 5 mức: AUC {_txt(_pair(g('M3', nm), ref))}")
    L += ["", "### KB-M4 – vai trò của mờ và đồ thị", ""]
    full = g("M4", "Full FKG-MM")
    for nm in ("Multimodal + Graph, không mờ (Crisp KG)", "Multimodal + Mờ, không đồ thị", "Multimodal w/o FKG (Early Fusion MLP)"):
        L.append(f"- Full FKG-MM − {nm}: AUC {_txt(_pair(full, g('M4', nm)))}; AP {_txt(_pair(full, g('M4', nm), 'auc_pr'))}")
    m5 = df[df.scenario == "M5"]
    if len(m5):
        piv = m5.pivot_table(index="fold", columns="model", values="auc_roc").dropna(axis=1)
        L += ["", "### KB-M5 – toán tử hợp nhất", "", "| Toán tử | Acc | AUC | AP | PR-AUC | BalAcc | Hạng AUC TB |",
              "|---|---|---|---|---|---|---|"]
        rk = piv.rank(axis=1, ascending=False).mean()
        for nm, q in m5.groupby("model", sort=False):
            L.append(f"| {nm} | " + " | ".join(_ms(q, m) for m in ("acc", "auc_roc", "auc_pr", "pr_auc_trap", "balacc"))
                     + f" | {rk.get(nm, np.nan):.1f} |")
        if piv.shape[1] >= 3 and piv.shape[0] >= 2:
            k, n = piv.shape[1], piv.shape[0]; R = piv.rank(axis=1, ascending=False).sum()
            W = 12 * ((R - R.mean()) ** 2).sum() / (n ** 2 * (k ** 3 - k))
            L.append(f"\nFriedman p = {stats.friedmanchisquare(*[piv[c] for c in piv.columns]).pvalue:.4f}; Kendall's W = {W:.2f}.")
    m6 = df[df.scenario == "M6"]
    if len(m6):
        full6 = g("M6", "FKG-MM đầy đủ")
        L += ["", "### KB-M6 – rút gọn bằng FKGS", "",
              "| Phương pháp | Tỉ lệ luật | AUC | AP | PR-AUC | BalAcc | Rep | Thời gian chọn + xây (s) | So với đầy đủ (AUC) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for (nm, data), q in m6.groupby(["model", "data"], sort=False):
            q = q.set_index("fold"); t = (q.t_select + q.t_fit).mean()
            L.append(f"| {nm} | {data} | {_ms(q, 'auc_roc')} | {_ms(q, 'auc_pr')} | {_ms(q, 'pr_auc_trap')} | {_ms(q, 'balacc')} | "
                     f"{'–' if q.rep.isna().all() else f'{q.rep.mean():.3f}'} | {t:.3f} | "
                     f"{'—' if nm == 'FKG-MM đầy đủ' else _txt(_pair(q, full6))} |")
    ep = df[df.scenario == "EPS"]
    if len(ep):
        L += ["", f"### Độ nhạy theo ε của {title}", "", "| ε | Acc | AUC | AP | PR-AUC | BalAcc |", "|---|---|---|---|---|---|"]
        for data, q in ep.groupby("data", sort=False):
            L.append(f"| {data} | " + " | ".join(_ms(q, m) for m in ("acc", "auc_roc", "auc_pr", "pr_auc_trap", "balacc")) + " |")
    text = "\n".join(L) + "\n"
    open(os.path.join(out, "ket_qua.md"), "w", encoding="utf-8").write(text)
    return text
