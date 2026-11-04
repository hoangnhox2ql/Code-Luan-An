"""Dữ liệu BRSET: đọc nhãn, ghép đặc trưng ảnh, tiền xử lý bảng (Thuật toán 3.3), chia fold.

BRSET (PhysioNet, Nakayama và cs. 2023/2024): thư mục gồm `labels_brset.csv` (hoặc labels.csv)
và thư mục ảnh `fundus_photos/` (tên tệp = image_id + .jpg). Tên cột được so khớp không phân
biệt hoa/thường; cột thiếu được bỏ qua kèm cảnh báo.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger(__name__)

# Cột chẩn đoán/tổn thương: TUYỆT ĐỐI không dùng làm đặc trưng (rò rỉ nhãn).
LEAKAGE_COLUMNS = {
    "diabetic_retinopathy", "dr_sdrg", "dr_icdr", "macular_edema", "scar", "nevus", "amd",
    "vascular_occlusion", "hypertensive_retinopathy", "drusens", "hemorrhage",
    "retinal_detachment", "myopic_fundus", "increased_cup_disc", "other",
}


def _find(df: pd.DataFrame, name: str) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    return low.get(name.lower())


def load_labels(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    path = os.path.join(d["root"], d["labels_file"])
    df = pd.read_csv(path)
    ren = {}
    for key in ("image_id", "patient_id", "label"):
        col = _find(df, d[f"{key}_col"])
        if col is None:
            raise KeyError(f"Không thấy cột '{d[f'{key}_col']}' trong {path}")
        ren[col] = key
    df = df.rename(columns=ren)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = (df["label"] > 0).astype(int)
    df["image_id"] = df["image_id"].astype(str)
    return df


def encode_table(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, List[str]]:
    """Giai đoạn 1 của Thuật toán 3.3: chọn cột, mã hoá, đánh dấu giá trị phi sinh lý là thiếu.
    Không dùng nhãn, không ước lượng tham số."""
    tcfg = cfg["table"]
    out = pd.DataFrame(index=df.index)
    names = []
    for spec in tcfg["columns"]:
        name = spec["name"]
        if name.lower() in LEAKAGE_COLUMNS:
            raise ValueError(f"Cột {name} là cột chẩn đoán – rò rỉ nhãn.")
        col = _find(df, name)
        for alias in spec.get("aliases", []):
            col = col or _find(df, alias)
        if col is None:
            log.warning("Bỏ qua cột bảng không có trong dữ liệu: %s", name)
            continue
        s = df[col]
        if spec.get("map"):
            m = {str(k).strip().lower(): v for k, v in spec["map"].items()}
            s = s.map(lambda v: m.get(str(v).strip().lower(), np.nan) if pd.notna(v) else np.nan)
        s = pd.to_numeric(s, errors="coerce")
        lo, hi = spec.get("valid_min"), spec.get("valid_max")
        if lo is not None:
            s = s.where(s >= lo)
        if hi is not None:
            s = s.where(s <= hi)
        out[name] = s.astype(float)
        names.append(name)
    return out, names


@dataclass
class TablePrep:
    """Giai đoạn 2 của Thuật toán 3.3 – mọi tham số ước lượng trên tập huấn luyện."""
    q: int = 12
    seed: int = 0

    def fit(self, T: np.ndarray, y: np.ndarray) -> "TablePrep":
        self.med_ = np.nanmedian(T, axis=0)
        self.med_ = np.where(np.isfinite(self.med_), self.med_, 0.0)
        Tf = self._fill(T)
        disc = np.column_stack([_discretize_for_mi(Tf[:, j]) for j in range(Tf.shape[1])])
        mi = mutual_info_classif(disc, y, discrete_features=True, random_state=self.seed)
        self.mi_ = mi
        self.J_ = list(np.argsort(-mi, kind="stable")[: min(self.q, T.shape[1])])
        return self

    def _fill(self, T):
        T = np.array(T, dtype=float, copy=True)
        bad = ~np.isfinite(T)
        T[bad] = np.take(self.med_, np.nonzero(bad)[1])
        return T

    def transform(self, T):
        return self._fill(T)[:, self.J_]


def _discretize_for_mi(x, bins=10):
    u = np.unique(x)
    if len(u) <= bins:
        return np.searchsorted(u, x)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, x, side="right")


def random_oversample(X_list: List[np.ndarray], y: np.ndarray, seed: int):
    """Lấy mẫu lặp lớp thiểu số đến cân bằng – CHỈ áp dụng cho tập huấn luyện."""
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    idx = [np.arange(len(y))]
    for lab, c in zip(labels, counts):
        if c < counts.max():
            idx.append(rng.choice(np.flatnonzero(y == lab), counts.max() - c, replace=True))
    idx = np.concatenate(idx)
    return [X[idx] for X in X_list], y[idx], idx


def outer_folds(y, groups, n_splits: int, seed: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Chia theo bệnh nhân, phân tầng theo nhãn (yêu cầu Y4)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from sgkf.split(np.zeros(len(y)), y, groups)


def inner_split(y, groups, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Tách tập xác thực từ tập huấn luyện, vẫn theo bệnh nhân."""
    k = max(2, int(round(1 / val_fraction)))
    tr, va = next(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
                  .split(np.zeros(len(y)), y, groups))
    return tr, va


def assert_no_patient_overlap(groups, *index_sets):
    sets = [set(np.asarray(groups)[i]) for i in index_sets]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            inter = sets[a] & sets[b]
            if inter:
                raise AssertionError(f"Rò rỉ bệnh nhân giữa các phần: {len(inter)} bệnh nhân chung")


def image_path(cfg: dict, image_id: str) -> str:
    d = cfg["data"]
    return os.path.join(d["root"], d["images_dir"], str(image_id) + d["image_ext"])


def load_dataset(cfg: dict) -> Dict:
    """Ghép bảng nhãn với đặc trưng ảnh đã trích.

    - Đặc trưng thủ công (scripts/01) là BẮT BUỘC: dùng cho FKG-MM như luận án.
    - Đặc trưng ResNet-50 đông cứng (scripts/02) là TUỲ CHỌN: dùng cho đối chứng "linear probe"
      và cho biến thể FKG-MM dùng đặc trưng sâu nếu image.fkgmm_source = deep/both.
    """
    df = load_labels(cfg)
    T, tab_names = encode_table(df, cfg)
    hand = pd.read_csv(cfg["image"]["features_file"], dtype={"image_id": str}).drop_duplicates("image_id")
    merged = df[["image_id", "patient_id", "label"]].join(T).merge(hand, on="image_id", how="inner")
    hand_names = [c for c in hand.columns if c != "image_id"]
    deep_names, I_deep = [], None
    dpath = cfg["image"].get("deep_features_file")
    if dpath and os.path.exists(dpath):
        deep = pd.read_csv(dpath, dtype={"image_id": str}).drop_duplicates("image_id")
        merged = merged.merge(deep, on="image_id", how="inner")
        deep_names = [c for c in deep.columns if c != "image_id"]
        I_deep = merged[deep_names].to_numpy(dtype=np.float32)
    dropped = len(df) - len(merged)
    if dropped:
        log.warning("%d dòng không ghép được với đặc trưng ảnh và bị loại (bước 2, Thuật toán 3.3)", dropped)
    return {
        "image_id": merged["image_id"].to_numpy(),
        "path": np.array([image_path(cfg, i) for i in merged["image_id"]]),
        "groups": merged["patient_id"].astype(str).to_numpy(),
        "y": merged["label"].to_numpy().astype(int),
        "T": merged[tab_names].to_numpy(dtype=float),
        "tab_names": tab_names,
        "I": merged[hand_names].to_numpy(dtype=float),
        "img_names": hand_names,
        "I_deep": I_deep,
        "deep_names": deep_names,
    }
