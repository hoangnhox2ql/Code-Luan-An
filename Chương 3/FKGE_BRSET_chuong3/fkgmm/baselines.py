"""Đối chứng cho KB-M1, KB-M2. Mọi đối chứng dùng cùng fold, cùng tập xác thực để chọn ngưỡng.

- Bảng: LR, RF, XGBoost (nếu đã cài, ngược lại HistGradientBoosting), MLP.
- Ảnh: LR trên đặc trưng ảnh (thủ công hoặc học sâu đã PCA) = "linear probe".
  Lưu ý: ResNet-50/EfficientNet-B0 ở đây là trích đặc trưng đông cứng + LR, KHÔNG phải
  tinh chỉnh đầu-cuối như các bài học sâu; cần ghi rõ khi báo cáo.
- Early Fusion: MLP trên [bảng || ảnh]. Late Fusion: trung bình xác suất của MLP-bảng và LR-ảnh.
Joint Fusion, Cross-modal Attention và RETFound chưa được cài đặt (cần PyTorch + GPU).
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import random_oversample
from .metrics import binary_metrics, choose_threshold


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log(1 - p)


def make_model(name: str, seed: int):
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                      class_weight="balanced", random_state=seed)
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=-1)
        except ImportError:
            return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed,
                                                  class_weight="balanced")
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                             early_stopping=True, max_iter=500,
                                                             random_state=seed))
    raise ValueError(name)


def run_baseline(name: str, model: str, X_tr, y_tr, X_va, y_va, X_te, y_te, seed: int,
                 balance: str = "none", criterion: str = "balacc") -> Dict:
    # sklearn MLP không có class_weight: luôn lấy mẫu lặp lớp thiểu số trên tập huấn luyện
    # (các mô hình khác dùng class_weight="balanced"); tập val/test giữ nguyên phân phối.
    if model == "mlp":
        (X_tr,), y_tr, _ = random_oversample([X_tr], y_tr, seed)
    t0 = time.perf_counter()
    m = make_model(model, seed).fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0
    s_va = _logit(m.predict_proba(X_va)[:, 1])
    thr = choose_threshold(s_va, y_va, criterion)
    t1 = time.perf_counter()
    s_te = _logit(m.predict_proba(X_te)[:, 1])
    t_inf = time.perf_counter() - t1
    return {"method": name, **binary_metrics(y_te, s_te, thr), "threshold": thr,
            "n_features": X_tr.shape[1], "t_fit": t_fit, "t_infer": t_inf,
            "t_infer_per_query_ms": 1000 * t_inf / len(y_te), "_score": s_te, "_model": m}


def late_fusion(name, m_tab, m_img, Xt_va, Xi_va, y_va, Xt_te, Xi_te, y_te, criterion="balacc"):
    p_va = (m_tab.predict_proba(Xt_va)[:, 1] + m_img.predict_proba(Xi_va)[:, 1]) / 2
    p_te = (m_tab.predict_proba(Xt_te)[:, 1] + m_img.predict_proba(Xi_te)[:, 1]) / 2
    thr = choose_threshold(_logit(p_va), y_va, criterion)
    return {"method": name, **binary_metrics(y_te, _logit(p_te), thr), "threshold": thr,
            "n_features": Xt_te.shape[1] + Xi_te.shape[1], "_score": _logit(p_te)}
