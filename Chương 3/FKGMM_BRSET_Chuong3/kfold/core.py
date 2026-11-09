"""FKG-UM / FKG-MM / 5 chiến lược hợp nhất / FKGS với FISA TRUNG BÌNH NHÂN trên bộ dữ liệu
KFold_feature_selection_rerun_20260921 (5 fold theo patient_id; đặc trưng chọn trên fold huấn luyện;
BorderlineSMOTE chỉ trên tập huấn luyện).

FISA trung bình nhân (FISA-GM):
  với truy vấn x, mỗi bộ ba thuộc tính t cho C_l(t) = tổng trọng số B của các luật nhãn l khớp x trên t
  (C thô, không chuẩn hoá); D_l(x) = exp( (1/T) Σ_t log(C_l(t) + ε) ); nhãn = argmax_l D_l(x);
  điểm xếp hạng s(x) = log D_1 − log D_0 = (1/T) Σ_t log[(C_1(t)+ε)/(C_0(t)+ε)].
Mọi bộ ba đều đóng góp (khác FISA gốc chỉ dùng max và min). Tập huấn luyện đã cân bằng bằng SMOTE nên argmax
không thiên lệch về lớp đa số.

FISA-GM-w (bộ suy diễn chính từ 02/11/2026, sau kiểm toán; kfold/weighted.py):
  s_w(x) = Σ_t w_t s_t(x) / Σ_t w_t,  s_t(x) = log(C_1(t)+ε) − log(C_0(t)+ε),  w_t = max(AUC_t − 1/2, 0),
  AUC_t ước lượng chéo 5 phần theo bệnh nhân CHỈ trên ảnh gốc của tập huấn luyện (cùng phép mờ hoá, cùng ε);
  nhãn = 1 nếu s_w(x) > 0. w_t ≡ hằng số cho lại đúng FISA-GM. cfg["aggregator"] = "geomean" giữ bản cũ.

Kịch bản (Chương 3, FKG-E tách riêng):
  M1  FKG-MM vs FKG-UM (Bảng, Ảnh) và các đối chứng đơn mô thức (LR, RF, XGBoost, MLP)
  M2  FKG-MM vs Early Fusion (MLP), Late Fusion (Ensemble)
  M3  số mức ngôn ngữ (số cụm FCM) 3 / 5 / 7 cho FKG-MM
  M4  cắt bỏ: Crisp KG (không mờ), Mờ không đồ thị (LR trên nhãn ngôn ngữ), chỉ ảnh, chỉ bảng
  M5  năm toán tử hợp nhất: Feature selection (đề xuất), Filter, Hadamard, Tensor, Wrapper
  M6  FKGS: ngẫu nhiên phân tầng, S-FKGS, C-FKGS với 15% và 20% luật
  EPS độ nhạy với ε (1e-12, 1e-9, 1e-6) cho FKG-MM
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)

from fkgmm.fisa import FKG
from fkgmm.sfkgs import RuleSampler

MODALITIES = ["table", "image", "fusion", "fusion_filter", "fusion_hadamard", "fusion_tensor", "fusion_wrapper"]
MODEL_NAME = {"table": "FKG-UM (Bảng)", "image": "FKG-UM (Ảnh)", "fusion": "FKG-MM (đề xuất)",
              "fusion_filter": "FKG-MM (Filter)", "fusion_hadamard": "FKG-MM (Hadamard)",
              "fusion_tensor": "FKG-MM (Tensor)", "fusion_wrapper": "FKG-MM (Wrapper)"}
SCENARIOS = ["M1", "M2", "M3", "M4", "M5", "M6", "EPS"]
DEFAULTS = {"aggregator": "geomean_w", "rule_base": "all", "tie": "drop", "fcm_fit": "all", "n_clusters": 5, "eps": 1e-9, "levels": [3, 5, 7], "ran": [15, 20], "eps_grid": [1e-12, 1e-9, 1e-6],
            "seed": 0, "fkg_um_table": "table"}


# ------------------------------------------------------------------ dữ liệu
def list_folds(root, modality="fusion"):
    return sorted(d for d in os.listdir(os.path.join(root, modality)) if d.startswith("fold_"))


def load(root, modality, fold):
    d = os.path.join(root, modality, fold)
    tr = pd.read_csv(os.path.join(d, "train_data.csv")); te = pd.read_csv(os.path.join(d, "test_data.csv"))
    ids_tr = pd.read_csv(os.path.join(d, "train_ids.csv")); ids_te = pd.read_csv(os.path.join(d, "test_ids.csv"))
    return {"Xtr": tr.iloc[:, :-1].to_numpy(float), "ytr": tr.iloc[:, -1].to_numpy(int),
            "Xte": te.iloc[:, :-1].to_numpy(float), "yte": te.iloc[:, -1].to_numpy(int),
            "ids_tr": ids_tr, "ids_te": ids_te, "n_orig_train": len(ids_tr), "features": list(tr.columns[:-1])}


def _selected_names(root, modality, fold, fallback):
    """Tên cột nguồn của các thuộc tính đã chọn (theo selected_features.csv của fold), nối bằng '|'."""
    path = os.path.join(root, modality, fold, "selected_features.csv")
    if not os.path.exists(path):
        return "|".join(map(str, fallback))
    sel = pd.read_csv(path)
    sel = sel[sel["selected"].astype(str).str.lower() == "true"]
    order = pd.to_numeric(sel["selected_output_column"], errors="coerce")
    return "|".join(sel.assign(_o=order).sort_values("_o")["source_column"].astype(str))


def audit(root) -> pd.DataFrame:
    rows = []
    for m in MODALITIES:
        for f in list_folds(root, m):
            D = load(root, m, f)
            rows.append({"modality": m, "fold": f, "n_train": len(D["ytr"]), "n_train_orig": D["n_orig_train"],
                         "n_smote": len(D["ytr"]) - D["n_orig_train"], "n_test": len(D["yte"]),
                         "features": D["Xtr"].shape[1], "test_pos": int(D["yte"].sum()),
                         "patient_overlap": len(set(D["ids_tr"].patient_id) & set(D["ids_te"].patient_id)),
                         "smote_rows_all_positive": bool((D["ytr"][D["n_orig_train"]:] == 1).all())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ mờ hoá FCM (ước lượng trên tập huấn luyện)
class FCMFuzzifier:
    def __init__(self, n_clusters=5, m=2.0, iters=150, tol=1e-6, mode="fcm"):
        self.c, self.m, self.iters, self.tol, self.mode = n_clusters, m, iters, tol, mode

    def _fcm(self, x):
        c = min(self.c, len(np.unique(x)))
        v = np.quantile(x, (np.arange(c) + 0.5) / c)
        for _ in range(self.iters):
            d = np.abs(x[:, None] - v[None, :]) + 1e-12
            U = 1.0 / ((d[:, :, None] / d[:, None, :]) ** (2 / (self.m - 1))).sum(2)
            Um = U ** self.m
            v_new = (Um * x[:, None]).sum(0) / Um.sum(0)
            done = np.max(np.abs(v_new - v)) < self.tol
            v = v_new
            if done:
                break
        return np.sort(v)

    def fit(self, X):
        self.centers = [np.linspace(X[:, j].min(), X[:, j].max(), self.c + 1) if self.mode == "crisp"
                        else self._fcm(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        R = np.zeros(X.shape, int)
        for j, v in enumerate(self.centers):
            if self.mode == "crisp":
                R[:, j] = np.clip(np.searchsorted(v[1:-1], X[:, j], side="right"), 0, self.c - 1)
            else:
                R[:, j] = np.argmin(np.abs(X[:, j][:, None] - v[None, :]), axis=1)   # = argmax độ thuộc FCM
        return R


# ------------------------------------------------------------------ chỉ số và FISA-GM
def metrics(y, pred, score):
    """auc_pr = AP (average_precision_score, bậc thang); pr_auc_trap = diện tích dưới đường PR theo quy tắc hình thang.
    Hai cách ước lượng PR-AUC khác nhau nên báo riêng; mốc của bộ đoán ngẫu nhiên là tỉ lệ dương (pos_rate)."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, score)
    return {"acc": (tp + tn) / len(y), "sens": tp / (tp + fn) if tp + fn else np.nan,
            "spec": tn / (tn + fp) if tn + fp else np.nan, "prec": tp / (tp + fp) if tp + fp else 0.0,
            "f1": f1_score(y, pred, zero_division=0), "balacc": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
            "mcc": matthews_corrcoef(y, pred), "auc_roc": roc_auc_score(y, score),
            "auc_pr": average_precision_score(y, score), "pr_auc_trap": auc(recall, precision),
            "pos_rate": float(np.mean(y)), "acc_majority": 1 - y.mean()}


def fisa_gm(Rtr, ytr, Rte, eps=1e-9, sampler=None, seed=0, weights=None):
    """Xây FKG trên luật huấn luyện và suy diễn bằng FISA trung bình nhân. Trả về (nhãn, điểm, thời gian)."""
    t0 = time.perf_counter(); rep = np.nan
    if sampler:
        sel = RuleSampler(seed=seed, **sampler).select(Rtr, ytr)
        Rtr, ytr, rep = Rtr[sel["index"]], ytr[sel["index"]], sel["rep"]
    t_sel = time.perf_counter() - t0
    m = FKG(c_norm="none", aggregator="geomean", eps=eps).fit(Rtr, ytr)
    t_fit = time.perf_counter() - t0 - t_sel
    t1 = time.perf_counter()
    if weights is None:
        D = m.decision_values(Rte)
        t_inf = time.perf_counter() - t1
        score = np.log(D[:, 1]) - np.log(D[:, 0])
        pred = (D[:, 1] > D[:, 0]).astype(int)
    else:                                   # FISA-GM-w: trung bình nhân có trọng số theo bộ ba
        L = np.log(m._lookup(Rte) + eps)
        score = (L[:, 1, :] - L[:, 0, :]) @ weights / weights.sum()
        t_inf = time.perf_counter() - t1
        pred = (score > 0).astype(int)
    return pred, score, {"t_select": t_sel, "t_fit": t_fit, "t_infer_ms": 1000 * t_inf / len(Rte),
                         "n_rules": len(ytr), "rep": rep}


# ------------------------------------------------------------------ thực nghiệm
def run(root, out, folds=None, scenarios=None, cfg=None) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    from fkgmm import baselines as BL
    c = {**DEFAULTS, **(cfg or {})}
    os.makedirs(out, exist_ok=True)
    folds = folds or list_folds(root)
    scenarios = scenarios or SCENARIOS
    rows: List[Dict] = []
    T0 = time.time()

    scores = {}                      # điểm của từng ảnh kiểm tra, mỗi mô hình một cột (để vẽ ROC/PR gộp ngoài fold)

    def add(sc, model, data, f, res, **kw):
        scores[f"{sc}|{model}|{data}"] = res["_score"]
        rows.append({"scenario": sc, "model": model, "data": data, "fold": f,
                     **{k: v for k, v in res.items() if k != "_score"}, **kw})

    for f in folds:
        cache = {m: load(root, m, f) for m in MODALITIES}
        # FKG-UM (Bảng): "table" = 13 cột lâm sàng (bản cũ); "table_sel" = các thuộc tính bảng FKG-MM chọn.
        # Các đối chứng LR/RF/XGBoost/MLP và Late Fusion luôn dùng nhóm "table".
        um_table = c["fkg_um_table"]
        if um_table != "table":
            if not os.path.isdir(os.path.join(root, um_table, f)):
                raise SystemExit(f"fkg_um_table = {um_table!r} nhưng chưa có {os.path.join(root, um_table, f)}: "
                                 f"chạy lại prepare cho nhóm {um_table}")
            cache[um_table] = load(root, um_table, f)
        scores.clear()

        wcache = {}

        def fit_rows(D):
            return D["Xtr"][: D["n_orig_train"]] if c["fcm_fit"] == "original" else D["Xtr"]

        def fkg(mod, n_clusters=c["n_clusters"], eps=c["eps"], mode="fcm", sampler=None):
            D = cache[mod]
            # fcm_fit = "original": hàm thuộc ước lượng trên ảnh gốc của tập huấn luyện (trước SMOTE), như Mục 3.6.3;
            # khái niệm nhị phân khi đó có đúng 2 tâm. "all": trên mọi dòng huấn luyện gồm SMOTE (bản 02/11).
            fz = FCMFuzzifier(n_clusters, mode=mode).fit(fit_rows(D))
            Rtr = fz.transform(D["Xtr"])
            w, t_w = None, 0.0
            if c["aggregator"] == "geomean_w":
                # trọng số bộ ba: ước lượng một lần cho mỗi (nhóm, phép mờ hoá, ε) trên ảnh gốc của tập huấn luyện;
                # mô hình rút gọn M6 dùng chung trọng số của tập huấn luyện đầy đủ
                from kfold.weighted import triple_weights
                key = (mod, n_clusters, mode, eps)
                if key not in wcache:
                    t0 = time.perf_counter(); n0 = D["n_orig_train"]
                    wcache[key] = (triple_weights(Rtr[:n0], D["ytr"][:n0], D["ids_tr"].patient_id.astype(str).to_numpy(),
                                                  eps, seed=c["seed"])[0], time.perf_counter() - t0)
                w, t_w = wcache[key]
            ytr, clean = D["ytr"], {}
            if c["rule_base"] == "dedup_conf":
                # luật trùng giữ 1, luật mâu thuẫn giữ nhãn có độ tin cậy cao hơn (kfold/rulebase.py); trọng số w
                # vẫn ước lượng trên ảnh gốc như bản chính để chỉ cơ sở luật thay đổi
                from kfold.rulebase import clean_rules
                Rtr, ytr, info = clean_rules(Rtr, ytr, c["tie"])
                clean = {"rules_in": info["rows_in"], "rules_kept": info["rules_out"],
                         "conflicting_antecedents": info["conflicting_antecedents"], "ties_dropped": info["ties"],
                         "rules_kept_pos": info["rules_out_per_label"].get(1, 0)}
            pred, score, tm = fisa_gm(Rtr, ytr, fz.transform(D["Xte"]), eps, sampler, c["seed"], w)
            return {**metrics(D["yte"], pred, score), **tm, **clean, "t_weights": t_w, "_score": score,
                    "w_nonzero": np.nan if w is None else int((w > 0).sum()), "n_triples_w": np.nan if w is None else len(w)}

        def base(model, mod):
            D = cache[mod]
            m = BL.make_model(model, c["seed"]); t0 = time.perf_counter(); m.fit(D["Xtr"], D["ytr"])
            p = m.predict_proba(D["Xte"])[:, 1]
            return {**metrics(D["yte"], (p >= 0.5).astype(int), p), "t_fit": time.perf_counter() - t0, "_score": p}

        need_main = bool({"M1", "M2", "M4", "M5"} & set(scenarios))
        main = {}
        if need_main:
            main = {mod: fkg(mod) for mod in MODALITIES if mod != "table"}
            main["table"] = {**fkg(um_table), "fkg_um_table": um_table,
                             "fkg_um_features": _selected_names(root, um_table, f, cache[um_table]["features"]),
                             "n_features": cache[um_table]["Xtr"].shape[1]}
        if "M1" in scenarios:
            for mod in ("fusion", "table", "image"):
                add("M1", MODEL_NAME[mod], "FKG", f, main[mod])
            for mdl, nm in (("lr", "Logistic Regression"), ("rf", "Random Forest"), ("xgb", "XGBoost"), ("mlp", "MLP")):
                add("M1", nm, "Tabular", f, base(mdl, "table"))
                add("M1", nm, "Image", f, base(mdl, "image"))
        if "M2" in scenarios:
            add("M2", "FKG-MM (đề xuất)", "FKG", f, main["fusion"])
            add("M2", "Early Fusion (MLP)", "Multimodal", f, base("mlp", "fusion"))
            mt = BL.make_model("mlp", c["seed"]).fit(cache["table"]["Xtr"], cache["table"]["ytr"])
            mi = BL.make_model("mlp", c["seed"]).fit(cache["image"]["Xtr"], cache["image"]["ytr"])
            p = (mt.predict_proba(cache["table"]["Xte"])[:, 1] + mi.predict_proba(cache["image"]["Xte"])[:, 1]) / 2
            add("M2", "Late Fusion (Ensemble)", "Multimodal", f,
                {**metrics(cache["table"]["yte"], (p >= 0.5).astype(int), p), "_score": p})
        if "M3" in scenarios:
            for L in c["levels"]:
                add("M3", f"FKG-MM, {L} mức", "FKG", f, fkg("fusion", n_clusters=L))
        if "M4" in scenarios:
            add("M4", "Full FKG-MM", "FKG", f, main["fusion"])
            add("M4", "Image-only (FKG-UM Ảnh)", "FKG", f, main["image"])
            add("M4", "Tabular-only (FKG-UM Bảng)", "FKG", f, main["table"])
            add("M4", "Multimodal + Graph, không mờ (Crisp KG)", "FKG", f, fkg("fusion", mode="crisp"))
            D = cache["fusion"]; fz = FCMFuzzifier(c["n_clusters"]).fit(fit_rows(D))
            lr = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                               LogisticRegression(max_iter=3000)).fit(fz.transform(D["Xtr"]), D["ytr"])
            p = lr.predict_proba(fz.transform(D["Xte"]))[:, 1]
            add("M4", "Multimodal + Mờ, không đồ thị", "LR trên nhãn ngôn ngữ", f,
                {**metrics(D["yte"], (p >= 0.5).astype(int), p), "_score": p})
            add("M4", "Multimodal w/o FKG (Early Fusion MLP)", "Multimodal", f, base("mlp", "fusion"))
        if "M5" in scenarios:
            for mod in MODALITIES[2:]:
                add("M5", MODEL_NAME[mod], "FKG", f, main[mod])
        if "M6" in scenarios:
            add("M6", "FKG-MM đầy đủ", "100% luật", f, fkg("fusion"))
            for ran in c["ran"]:
                for meth, lab in (("random", "Ngẫu nhiên phân tầng"), ("s", "S-FKGS"), ("c", "C-FKGS")):
                    add("M6", lab, f"{ran}% luật", f, fkg("fusion", sampler={"method": meth, "theta": ran / 100}))
        if "EPS" in scenarios:
            for e in c["eps_grid"]:
                add("EPS", "FKG-MM (đề xuất)", f"ε = {e:g}", f, fkg("fusion", eps=e))
        ids = cache["fusion"]["ids_te"]
        pred_dir = os.path.join(out, "predictions"); os.makedirs(pred_dir, exist_ok=True)
        pd.DataFrame({"image_id": ids.image_id.astype(str), "patient_id": ids.patient_id.astype(str),
                      "y": cache["fusion"]["yte"], **scores}).to_csv(os.path.join(pred_dir, f"{f}.csv.gz"), index=False)
        print(f"[{f}] xong ({(time.time() - T0) / 60:.1f} phút)", flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(out, "per_fold.csv")
    if os.path.exists(path):
        prev = pd.read_csv(path)
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(["scenario", "model", "data", "fold"], keep="last")
    df.to_csv(path, index=False)
    json.dump({"data": os.path.abspath(root), "folds": sorted(df.fold.unique().tolist()), "config": c,
               "fcm_fit": ("ảnh gốc của tập huấn luyện (trước SMOTE)" if c["fcm_fit"] == "original"
                           else "mọi dòng huấn luyện (gồm SMOTE)"),
               "rule_base": ("luật trùng giữ 1; mâu thuẫn giữ nhãn có độ tin cậy cao hơn (hoà: " + c["tie"] + ")"
                             if c["rule_base"] == "dedup_conf" else "mọi dòng huấn luyện (gồm SMOTE)"),
               "fisa": ("FISA-GM-w: trung bình nhân có trọng số bộ ba (kfold/weighted.py), C thô, s_w > 0"
                        if c["aggregator"] == "geomean_w" else "trung bình nhân (geomean), C thô, argmax"), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(out, "run_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return df
