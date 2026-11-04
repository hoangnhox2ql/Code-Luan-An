"""S-FKGS (Chương 2): rút gọn tập luật bằng tham lam phân tầng theo nhãn.

- d_H(R_i, R_j) = tỉ lệ tiền đề khác nhãn ngôn ngữ (công thức 2.3).
- Sim_L(R_i, R_j) = 1 - d_H nếu cùng nhãn hệ quả, 0 nếu khác (φ(x) = 1 - x, Mệnh đề 2.28).
- Rep(S) = (1/n) Σ_r max_{s∈S} Sim_L(r, s)  (hàm facility location, đơn điệu, dưới mô-đun).
- Ngân sách m = round(θ n) được phân bổ cho các tầng (nhãn) bằng phương pháp Hamilton;
  `min_per_class` bảo đảm hạn ngạch tối thiểu cho lớp thiểu số (đề xuất ở tr. 163 của luận án).
- Trong mỗi tầng chạy tham lam (lazy greedy) trên các luật DUY NHẤT có trọng số bội,
  cho cùng giá trị Rep như chạy trên toàn bộ luật nhưng nhanh hơn nhiều.
- Khi số luật duy nhất của một tầng vượt `exact_limit`, dùng stochastic greedy
  (Mirzasoleiman và cs., 2015) với tham số ε_sg; cận kỳ vọng (1 - 1/e - ε_sg).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def hamilton(sizes: np.ndarray, m: int, min_per: Optional[np.ndarray] = None) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=float)
    base = np.zeros(len(sizes), dtype=int) if min_per is None else np.minimum(min_per, sizes).astype(int)
    rest = m - base.sum()
    if rest <= 0:
        return base
    cap = sizes - base
    quota = cap / max(cap.sum(), 1) * rest
    alloc = np.floor(quota).astype(int)
    rem = int(rest - alloc.sum())
    for k in np.argsort(-(quota - alloc))[:rem]:
        alloc[k] += 1
    return np.minimum(base + alloc, sizes.astype(int))


def _sim_block(U: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Sim = 1 - d_H giữa các dòng của Q (k x r) và U (u x r) -> (k x u)."""
    return (Q[:, None, :] == U[None, :, :]).mean(axis=2)


def _greedy_stratum(U: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator,
                    exact_limit: int, eps_sg: float, chunk: int = 512) -> List[int]:
    u = U.shape[0]
    k = min(k, u)
    if k <= 0:
        return []
    best = np.zeros(u)
    chosen: List[int] = []
    if u <= exact_limit:
        # lazy greedy với hàng đợi ưu tiên (Minoux)
        gains0 = np.empty(u)
        for s in range(0, u, chunk):
            S = _sim_block(U, U[s:s + chunk])          # ứng viên s.. so với mọi điểm
            gains0[s:s + chunk] = S @ w
        heap = [(-g, i, 0) for i, g in enumerate(gains0)]
        heapq.heapify(heap)
        it = 0
        while len(chosen) < k and heap:
            g, i, stamp = heapq.heappop(heap)
            if stamp == it:
                chosen.append(i)
                best = np.maximum(best, _sim_block(U, U[i:i + 1])[0])
                it += 1
            else:
                gain = (np.maximum(_sim_block(U, U[i:i + 1])[0] - best, 0) * w).sum()
                heapq.heappush(heap, (-gain, i, it))
        return chosen
    # stochastic greedy
    s_size = int(np.ceil(u / k * np.log(1 / eps_sg)))
    remaining = np.ones(u, dtype=bool)
    for _ in range(k):
        pool = np.flatnonzero(remaining)
        cand = rng.choice(pool, size=min(s_size, len(pool)), replace=False)
        gains = np.empty(len(cand))
        for s in range(0, len(cand), chunk):
            S = _sim_block(U, U[cand[s:s + chunk]])
            gains[s:s + chunk] = (np.maximum(S - best[None, :], 0) * w[None, :]).sum(axis=1)
        i = int(cand[int(np.argmax(gains))])
        chosen.append(i); remaining[i] = False
        best = np.maximum(best, _sim_block(U, U[i:i + 1])[0])
    return chosen


def kmodes(U: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator, iters: int = 20) -> np.ndarray:
    """k-modes (Huang 1998) có trọng số trên các luật duy nhất; trả về nhãn cụm."""
    u = U.shape[0]
    k = min(k, u)
    if k <= 1:
        return np.zeros(u, dtype=int)
    modes = U[rng.choice(u, k, replace=False, p=w / w.sum())].copy()
    lab = np.zeros(u, dtype=int)
    for _ in range(iters):
        d = (U[:, None, :] != modes[None, :, :]).sum(axis=2)
        new = d.argmin(axis=1)
        if _ > 0 and np.array_equal(new, lab):
            break
        lab = new
        for c in range(k):
            m = lab == c
            if not m.any():
                modes[c] = U[rng.integers(u)]; continue
            for j in range(U.shape[1]):
                vals = U[m, j]
                cnt = np.bincount(vals, weights=w[m], minlength=vals.max() + 1)
                modes[c, j] = cnt.argmax()
    return lab


@dataclass
class RuleSampler:
    """Ba thuật toán rút gọn có bảo đảm của Chương 2 trên cùng hàm Rep (Sim_L = 1 - d_H, cùng nhãn).

    method = "greedy": GreedyFKGS – tham lam toàn cục (cận 1 - 1/e so với OPT).
    method = "s":      S-FKGS – phân hoạch theo nhãn, ngân sách Hamilton, tham lam trong khối.
    method = "c":      C-FKGS – phân hoạch theo (nhãn, cụm k-modes), tham lam trong khối
                       (cận 1 - 1/e so với OPT_pb của phân hoạch).
    min_per_class: hạn ngạch tối thiểu cho mỗi nhãn (áp dụng cho "s" và "c").
    """
    method: str = "s"
    theta: float = 0.15
    min_per_class: int = 0
    clusters_per_label: int = 5
    exact_limit: int = 3000
    eps_sg: float = 0.05
    seed: int = 0

    def select(self, X: np.ndarray, y: np.ndarray) -> Dict:
        X = np.asarray(X); y = np.asarray(y)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        m = max(1, int(round(self.theta * n)))
        # luật duy nhất theo (tiền đề, nhãn)
        XY = np.column_stack([X, y])
        U, first, inv, mult = np.unique(XY, axis=0, return_index=True, return_inverse=True, return_counts=True)
        inv = np.ravel(inv)
        Ux, Uy, w = U[:, :-1], U[:, -1], mult.astype(float)
        if self.method == "random":
            # Đối chứng: lấy ngẫu nhiên phân tầng theo nhãn (ngân sách Hamilton), không tối ưu Rep
            labels, counts = np.unique(y, return_counts=True)
            min_per = np.full(len(labels), self.min_per_class) if self.min_per_class else None
            idx = []
            for lab, k in zip(labels, hamilton(counts, m, min_per)):
                pool = np.flatnonzero(y == lab)
                idx += rng.choice(pool, size=min(int(k), len(pool)), replace=False).tolist()
            idx = np.array(sorted(idx))
            best = np.zeros(len(U))
            for c in np.unique(inv[idx]):
                same = Uy == Uy[c]
                best[same] = np.maximum(best[same], (Ux[same] == Ux[c]).mean(axis=1))
            return {"index": idx, "rep": float((best * w).sum() / n), "m": m, "n_blocks": len(labels)}
        # Sim_L giữa luật khác nhãn = 0  -> mã hoá nhãn như một "thuộc tính" bắt buộc khớp
        if self.method == "greedy":
            blocks = [np.arange(len(U))]
            alloc = [m]
        else:
            labels, counts = np.unique(y, return_counts=True)
            min_per = np.full(len(labels), self.min_per_class) if self.min_per_class else None
            a_lab = hamilton(counts, m, min_per)
            blocks, alloc = [], []
            for lab, k_lab in zip(labels, a_lab):
                idx = np.flatnonzero(Uy == lab)
                if self.method == "s":
                    blocks.append(idx); alloc.append(int(k_lab))
                elif self.method == "c":
                    cl = kmodes(Ux[idx], w[idx], self.clusters_per_label, rng)
                    sizes = np.array([w[idx][cl == c].sum() for c in range(cl.max() + 1)])
                    a_c = hamilton(sizes, int(k_lab))
                    for c, kc in enumerate(a_c):
                        blocks.append(idx[cl == c]); alloc.append(int(kc))
                else:
                    raise ValueError(self.method)
        chosen = []
        for idx, k in zip(blocks, alloc):
            if k <= 0 or len(idx) == 0:
                continue
            Ub = np.column_stack([Ux[idx], Uy[idx]]) if self.method == "greedy" else Ux[idx]
            ch = _greedy_stratum_masked(Ub, w[idx], k, rng, self.exact_limit, self.eps_sg,
                                        label_last=self.method == "greedy")
            chosen.extend(idx[ch].tolist())
        chosen = np.array(sorted(set(chosen)), dtype=int)
        # Rep thật trên toàn bộ luật
        best = np.zeros(len(U))
        for c in chosen:
            same = Uy == Uy[c]
            best[same] = np.maximum(best[same], (Ux[same] == Ux[c]).mean(axis=1))
        rep = float((best * w).sum() / n)
        return {"index": np.sort(first[chosen]), "rep": rep, "m": m, "n_blocks": len(blocks)}


def _greedy_stratum_masked(U, w, k, rng, exact_limit, eps_sg, label_last=False):
    """Như _greedy_stratum, nhưng nếu label_last: cột cuối là nhãn, Sim = 0 khi khác nhãn."""
    if not label_last:
        return _greedy_stratum(U, w, k, rng, exact_limit, eps_sg)
    Xp, lab = U[:, :-1], U[:, -1]
    k = min(k, len(U))
    best = np.zeros(len(U)); chosen = []
    def sim(i):
        s = (Xp == Xp[i]).mean(axis=1)
        return np.where(lab == lab[i], s, 0.0)
    if len(U) <= exact_limit:
        g0 = np.array([(sim(i) * w).sum() for i in range(len(U))])
        heap = [(-g, i, 0) for i, g in enumerate(g0)]; heapq.heapify(heap); it = 0
        while len(chosen) < k and heap:
            g, i, st = heapq.heappop(heap)
            if st == it:
                chosen.append(i); best = np.maximum(best, sim(i)); it += 1
            else:
                heapq.heappush(heap, (-(np.maximum(sim(i) - best, 0) * w).sum(), i, it))
        return chosen
    s_size = int(np.ceil(len(U) / k * np.log(1 / eps_sg)))
    remaining = np.ones(len(U), dtype=bool)
    for _ in range(k):
        cand = rng.choice(np.flatnonzero(remaining), size=min(s_size, remaining.sum()), replace=False)
        gains = [(np.maximum(sim(i) - best, 0) * w).sum() for i in cand]
        i = int(cand[int(np.argmax(gains))]); chosen.append(i); remaining[i] = False
        best = np.maximum(best, sim(i))
    return chosen


@dataclass
class SFKGS:
    theta: float = 0.15
    min_per_class: int = 0
    exact_limit: int = 3000
    eps_sg: float = 0.05
    seed: int = 0

    def select(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Trả về chỉ số (trong X) của tập luật mẫu và các thống kê Rep."""
        X = np.asarray(X); y = np.asarray(y)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        labels, counts = np.unique(y, return_counts=True)
        m = max(1, int(round(self.theta * n)))
        min_per = np.full(len(labels), self.min_per_class) if self.min_per_class else None
        alloc = hamilton(counts, m, min_per)
        selected: List[int] = []
        rep_num = 0.0
        for lab, k in zip(labels, alloc):
            idx = np.flatnonzero(y == lab)
            U, inv, mult = np.unique(X[idx], axis=0, return_inverse=True, return_counts=True)
            inv = np.ravel(inv)
            ch = _greedy_stratum(U, mult.astype(float), int(k), rng, self.exact_limit, self.eps_sg)
            # đại diện của mỗi luật duy nhất được chọn: lấy một luật gốc tương ứng
            for c in ch:
                selected.append(int(idx[np.flatnonzero(inv == c)[0]]))
            if ch:
                cov = np.max(np.stack([_sim_block(U, U[c:c + 1])[0] for c in ch]), axis=0)
                rep_num += float((cov * mult).sum())
        return {"index": np.array(sorted(selected)), "rep": rep_num / n,
                "alloc": dict(zip(labels.tolist(), alloc.tolist())), "m": m}
