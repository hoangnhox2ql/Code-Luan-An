"""Đồ thị tri thức mờ (FKG) và thuật toán suy diễn FISA.

Cài đặt này tái hiện CHÍNH XÁC phép tính của `fisa_module.cpp` trong kho
https://github.com/thanhst/Fuzzy-Knowledge-Graph (các hàm calculateA, calculateM,
calculateB, calculateC, FISA) và quy trình của bài APIN (CT2, FKG_S.py):

    A = fs.calculateA(base); M = fs.calculateM(base)
    B = fs.calculateB(base, A, M); C = fs.calculateC(base, B, n_classes)
    C = min_max_normalize(C)          # chuẩn hoá theo cột trên các dòng của base
    nhãn = fs.FISA(base, C, x, n_classes)

Khác biệt duy nhất là cách tính: bản C++ duyệt O(n^2) theo cặp luật; ở đây
các tổng theo nhóm được tính bằng bảng băm (np.unique), cho cùng kết quả số học
(kiểm chứng bằng tests/test_fisa_equivalence.py so với bản dịch từng dòng của
mã C++ trong fisa_reference.py).

Ký hiệu (khớp mã C++):
    base   : ma trận số nguyên n x (r+1); r cột tiền đề (nhãn ngôn ngữ), cột cuối là nhãn lớp.
    A[r1, q]  = #{r2 : base[r2] khớp base[r1] trên bộ 4 thuộc tính q} / n
    M[r1, i]  = #{r2 : base[r2, i] == base[r1, i] và cùng nhãn} / n
    B[r, t]   = (sum_q A[r, q]) * min(M[r, a], M[r, b], M[r, c]),  t = (a, b, c)
    C[r1, l, t] = sum_{r2 khớp r1 trên t, nhãn r2 == l} B[r2, t]
    FISA(x): với mỗi t, C_l(t) = C[r, l, t] của dòng r khớp x trên t và có nhãn l
             (0 nếu không có); D_l = max_t C_l(t) + min_t C_l(t).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np


def _encode(cols: np.ndarray, radix: int) -> np.ndarray:
    """Mã hoá mỗi dòng của ma trận số nguyên không âm (n x k) thành một int64."""
    key = np.zeros(cols.shape[0], dtype=np.int64)
    for j in range(cols.shape[1]):
        key = key * radix + cols[:, j].astype(np.int64)
    return key


@dataclass
class TripleTable:
    """Bảng tra cho một bộ ba thuộc tính t: khoá -> giá trị C (đã chuẩn hoá) theo lớp."""
    keys: np.ndarray          # khoá đã sắp xếp (u,)
    values: np.ndarray        # (u, K) giá trị C_l sau chuẩn hoá
    present: np.ndarray       # (u, K) bool: có luật nhãn l khớp khoá này hay không


@dataclass
class FKG:
    """Đồ thị tri thức mờ + FISA theo đúng pipeline APIN.

    Tham số
    -------
    c_norm : "minmax" (mặc định, như FKG_S.py/FKG_general.py của bài APIN) hoặc "none"
             (như FKG.py gốc của Lan và cs.).
    """

    c_norm: str = "minmax"
    aggregator: str = "maxmin"   # "maxmin": FISA gốc D = max_t + min_t; "mean": trung bình; "geomean": trung bình nhân
    eps: float = 1e-9            # hằng số tránh log 0 trong trung bình nhân
    skip_last_rule: bool = False # True: như notebook FKG.ipynb (vòng range(row-1) bỏ qua luật cuối khi so khớp)
    triples_: List[Tuple[int, int, int]] = field(default_factory=list, repr=False)
    tables_: List[TripleTable] = field(default_factory=list, repr=False)
    classes_: Optional[np.ndarray] = None
    radix_: int = 0
    n_rules_: int = 0
    n_attr_: int = 0
    B_: Optional[np.ndarray] = field(default=None, repr=False)
    base_: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, keep_base: bool = True) -> "FKG":
        X = np.asarray(X, dtype=np.int64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] < 4:
            raise ValueError("FISA cần ít nhất 4 thuộc tính tiền đề (do ma trận A dùng bộ 4).")
        if X.min() < 0:
            raise ValueError("Nhãn ngôn ngữ phải là số nguyên không âm.")
        n, r = X.shape
        self.classes_, yi = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        # radix đủ lớn cho mọi giá trị, cộng 1 cho mã "chưa thấy" ở truy vấn
        self.radix_ = int(X.max()) + 2
        self.n_rules_, self.n_attr_ = n, r

        # --- A: chỉ cần tổng theo dòng SA[r] = sum_q A[r, q]
        SA = np.zeros(n, dtype=np.float64)
        for q in combinations(range(r), 4):
            key = _encode(X[:, q], self.radix_)
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            SA += cnt[inv] / n

        # --- M: tần suất (giá trị thuộc tính i, nhãn) / n
        M = np.zeros((n, r), dtype=np.float64)
        for i in range(r):
            key = X[:, i].astype(np.int64) * K + yi
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            M[:, i] = cnt[inv] / n

        # --- B và C theo từng bộ ba
        self.triples_ = list(combinations(range(r), 3))
        T = len(self.triples_)
        B = np.empty((n, T), dtype=np.float64)
        self.tables_ = []
        raw_tables = []
        for t, (a, b, c) in enumerate(self.triples_):
            B[:, t] = SA * np.minimum(np.minimum(M[:, a], M[:, b]), M[:, c])
            key = _encode(X[:, [a, b, c]], self.radix_)
            ukeys, inv = np.unique(key, return_inverse=True)
            vals = np.zeros((len(ukeys), K), dtype=np.float64)
            cnts = np.zeros((len(ukeys), K), dtype=np.int64)
            np.add.at(vals, (inv, yi), B[:, t])
            np.add.at(cnts, (inv, yi), 1)
            if self.skip_last_rule:
                cnts[inv[-1], yi[-1]] -= 1          # luật cuối không tham gia so khớp (C vẫn tính đủ)
            raw_tables.append((ukeys, inv, vals, cnts))

        # --- chuẩn hoá min-max theo cột của ma trận C (n x K*T), như min_max_normalize
        for (ukeys, inv, vals, cnts) in raw_tables:
            if self.c_norm == "minmax":
                col = vals[inv]                     # giá trị C của mọi dòng base (n x K)
                cmin, cmax = col.min(axis=0), col.max(axis=0)
                rng = cmax - cmin
                with np.errstate(invalid="ignore", divide="ignore"):
                    norm = (vals - cmin) / rng
                # Cột hằng số: numpy cho NaN (0/0). Bản gốc truyền NaN vào FISA và
                # std::max_element/min_element bỏ qua so sánh NaN không xác định;
                # ở đây quy NaN về 0 để kết quả xác định (ghi rõ trong README).
                norm = np.where(np.isfinite(norm), norm, 0.0)
            elif self.c_norm == "none":
                norm = vals
            else:
                raise ValueError(f"c_norm không hợp lệ: {self.c_norm}")
            self.tables_.append(TripleTable(keys=ukeys, values=norm, present=cnts > 0))
        self.n_lookup_rows_ = n - int(self.skip_last_rule)

        if keep_base:
            self.B_ = B
            self.base_ = np.column_stack([X, yi])
        return self

    # ------------------------------------------------------------ inference
    def _lookup(self, Xq: np.ndarray) -> np.ndarray:
        """Trả về mảng (m, K, T) giá trị C_l(t) cho từng truy vấn."""
        Xq = np.asarray(Xq, dtype=np.int64)
        Xq = np.where((Xq < 0) | (Xq >= self.radix_ - 1), self.radix_ - 1, Xq)  # mã chưa thấy
        m = Xq.shape[0]
        K = len(self.classes_)
        out = np.zeros((m, K, len(self.triples_)), dtype=np.float64)
        for t, ((a, b, c), tab) in enumerate(zip(self.triples_, self.tables_)):
            key = _encode(Xq[:, [a, b, c]], self.radix_)
            pos = np.searchsorted(tab.keys, key)
            pos_c = np.clip(pos, 0, len(tab.keys) - 1)
            hit = tab.keys[pos_c] == key
            vals = np.where(tab.present[pos_c], tab.values[pos_c], 0.0)
            out[:, :, t] = np.where(hit[:, None], vals, 0.0)
        return out

    def decision_values(self, Xq: np.ndarray) -> np.ndarray:
        """D_l(x) = max_t C_l(t) + min_t C_l(t); trả về (m, K)."""
        Cv = self._lookup(Xq)
        if self.aggregator == "maxmin":
            return Cv.max(axis=2) + Cv.min(axis=2)
        if self.aggregator == "mean":
            return Cv.mean(axis=2)
        if self.aggregator == "geomean":
            return np.exp(np.log(Cv + self.eps).mean(axis=2))
        raise ValueError(self.aggregator)

    def predict_factor(self, Xq: np.ndarray, factor: float = 9.0) -> np.ndarray:
        """Quy tắc của notebook FKG.ipynb: dự đoán lớp 0 nếu D0 > factor·D1, ngược lại lớp 1 (hai lớp)."""
        D = self.decision_values(Xq)
        return np.where(D[:, 0] > factor * D[:, 1], self.classes_[0], self.classes_[1])

    def predict_argmax(self, Xq: np.ndarray) -> np.ndarray:
        """Quy tắc chọn nhãn của bài APIN: argmax D (hoà -> chỉ số nhỏ nhất, như std::max_element)."""
        return self.classes_[np.argmax(self.decision_values(Xq), axis=1)]

    def score(self, Xq: np.ndarray, pos_label=1, eps: float = 1e-12) -> np.ndarray:
        """Điểm liên tục cho bài toán hai lớp: log(D_pos + eps) - log(D_neg + eps).

        Dùng cho AUC và cho ngưỡng hiệu chỉnh tau* chọn trên tập xác thực
        (Định nghĩa 3.21). Với nhiều lớp, dùng decision_values.
        """
        D = self.decision_values(Xq)
        if len(self.classes_) != 2:
            raise ValueError("score() chỉ dành cho bài toán hai lớp.")
        p = int(np.where(self.classes_ == pos_label)[0][0])
        return np.log(D[:, p] + eps) - np.log(D[:, 1 - p] + eps)

    def confidence(self, Xq: np.ndarray) -> np.ndarray:
        """Độ tin cậy như FISA_with_confidence: D_best / sum(D)."""
        D = self.decision_values(Xq)
        s = D.sum(axis=1)
        return np.where(s > 0, D.max(axis=1) / np.where(s > 0, s, 1), 0.0)

    # --------------------------------------------------------- truy vết luật
    def explain(self, x: np.ndarray, label_index: int, top: int = 10) -> Dict:
        """Căn cứ luật của FISA tại x cho nhãn label_index (Định nghĩa 3.21(i)).

        Lấy hai nút ba thuộc tính c-, c+ đạt min và max của C_l(t); các luật nhãn l
        khớp x trên c-/c+ được gán trọng số B[t, c]. Trả về top luật và trọng số.
        """
        if self.base_ is None:
            raise RuntimeError("Cần fit(keep_base=True) để truy vết luật.")
        x = np.asarray(x, dtype=np.int64)
        Cv = self._lookup(x[None, :])[0, label_index]
        t_min, t_max = int(np.argmin(Cv)), int(np.argmax(Cv))
        w = np.zeros(self.base_.shape[0])
        for t in {t_min, t_max}:
            a, b, c = self.triples_[t]
            m = ((self.base_[:, a] == x[a]) & (self.base_[:, b] == x[b]) &
                 (self.base_[:, c] == x[c]) & (self.base_[:, -1] == label_index))
            w[m] += self.B_[m, t]
        order = np.argsort(-w)[:top]
        order = order[w[order] > 0]
        tot = w.sum()
        return {
            "triple_min": self.triples_[t_min], "triple_max": self.triples_[t_max],
            "D": float(Cv.max() + Cv.min()),
            "rules": [(int(i), float(w[i] / tot) if tot > 0 else 0.0) for i in order],
        }

    def coverage(self, Xq: np.ndarray) -> np.ndarray:
        """Tỉ lệ bộ ba của truy vấn khớp ít nhất một luật (mọi nhãn)."""
        Xq = np.asarray(Xq, dtype=np.int64)
        Xq = np.where((Xq < 0) | (Xq >= self.radix_ - 1), self.radix_ - 1, Xq)
        hits = np.zeros(Xq.shape[0])
        for (a, b, c), tab in zip(self.triples_, self.tables_):
            key = _encode(Xq[:, [a, b, c]], self.radix_)
            pos = np.clip(np.searchsorted(tab.keys, key), 0, len(tab.keys) - 1)
            hits += tab.keys[pos] == key
        return hits / len(self.triples_)


# ---------------------------------------------------------------------------
# Bổ sung cho FKG-E: căn cứ luật theo lô và FISA "so khớp tuần tự"
def attribution_batch(fkg: FKG, Xq: np.ndarray, label_idx: np.ndarray, top: Optional[int] = None,
                      normalized: bool = True):
    """Căn cứ luật a_F (Định nghĩa 3.21(i)) cho nhiều truy vấn.

    Trả về danh sách (rows, weights) với rows là chỉ số dòng trong base_ của fkg, weights chuẩn hoá
    tổng 1 (rỗng nếu không luật nào khớp). Nếu top: giữ top luật có trọng số lớn nhất rồi chuẩn hoá lại.
    normalized=True: trọng số của các luật tại nút c được nhân với C̃_c / C_c (C̃ là giá trị đã chuẩn hoá
    min–max mà FISA dùng), để tổng trọng số trước chuẩn hoá bằng đúng D_ŷ = C̃(c-) + C̃(c+) – tức là
    phân rã chính xác giá trị quyết định như Định nghĩa 3.21(i) khẳng định (với c_norm="none" hệ số = 1).
    """
    if fkg.base_ is None:
        raise RuntimeError("Cần fit(keep_base=True)")
    base = fkg.base_
    if not hasattr(fkg, "_groups"):
        fkg._groups = []
        for (a, b, c) in fkg.triples_:
            key = _encode(base[:, [a, b, c, -1]], max(fkg.radix_, len(fkg.classes_) + 1))
            order = np.argsort(key, kind="stable")
            ks = key[order]
            fkg._groups.append((ks, order))
    Cv = fkg._lookup(Xq)                                      # (m, K, T)
    Xq = np.asarray(Xq, dtype=np.int64)
    rad = max(fkg.radix_, len(fkg.classes_) + 1)
    out = []
    for i in range(Xq.shape[0]):
        l = int(label_idx[i])
        cv = Cv[i, l]
        rows_all, w_all = [], []
        for t in {int(np.argmin(cv)), int(np.argmax(cv))}:
            a, b, c = fkg.triples_[t]
            key = _encode(np.array([[Xq[i, a], Xq[i, b], Xq[i, c], l]]), rad)[0]
            ks, order = fkg._groups[t]
            lo, hi = np.searchsorted(ks, key, "left"), np.searchsorted(ks, key, "right")
            if hi > lo:
                rows = order[lo:hi]
                w = fkg.B_[rows, t]
                if normalized and w.sum() > 0:
                    w = w * (cv[t] / w.sum())
                rows_all.append(rows); w_all.append(w)
        if not rows_all:
            out.append((np.array([], int), np.array([])))
            continue
        rows = np.concatenate(rows_all); w = np.concatenate(w_all)
        if top and len(rows) > top:
            k = np.argsort(-w)[:top]; rows, w = rows[k], w[k]
        if w.sum() <= 0:
            out.append((np.array([], int), np.array([])))
            continue
        out.append((rows, w / w.sum()))
    return out


def fisa_sequential(fkg: FKG, Xq: np.ndarray) -> np.ndarray:
    """FISA so khớp tuần tự: với mỗi truy vấn, duyệt mọi luật trên mọi bộ ba (O(|R| C(r,3))),
    giống vòng lặp của fisa_module.cpp. Dùng để đo chi phí (KB-E2); kết quả trùng decision_values."""
    base = fkg.base_
    X, yb = base[:, :-1], base[:, -1]
    K, T = len(fkg.classes_), len(fkg.triples_)
    # giá trị C chuẩn hoá của từng dòng base cho từng bộ ba (tra một lần, như ma trận C của bản C++)
    if not hasattr(fkg, "_Crow"):
        Crow = np.zeros((len(yb), K, T))
        for t, ((a, b, c), tab) in enumerate(zip(fkg.triples_, fkg.tables_)):
            key = _encode(X[:, [a, b, c]], fkg.radix_)
            pos = np.searchsorted(tab.keys, key)
            Crow[:, :, t] = tab.values[pos]
        fkg._Crow = Crow
    Crow = fkg._Crow
    D = np.zeros((len(Xq), K))
    for i, x in enumerate(np.asarray(Xq, dtype=np.int64)):
        Cv = np.zeros((K, T))
        for t, (a, b, c) in enumerate(fkg.triples_):
            m = (X[:, a] == x[a]) & (X[:, b] == x[b]) & (X[:, c] == x[c])
            if m.any():
                for l in np.unique(yb[m]):
                    r = np.flatnonzero(m & (yb == l))[0]
                    Cv[l, t] = Crow[r, l, t]
        D[i] = Cv.max(axis=1) + Cv.min(axis=1)
    return D
