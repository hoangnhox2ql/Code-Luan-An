"""S-FKGS (Chương 2): rút gọn cơ sở luật bằng tham lam phân tầng theo nhãn.

Sim_L(R_i, R_j) = 1 - d_H(R_i, R_j) nếu cùng nhãn, 0 nếu khác nhãn (d_H: tỉ lệ tiền đề khác nhau).
Rep(S) = (1/n) Σ_r max_{s∈S} Sim_L(r, s)  (facility location, đơn điệu, dưới mô-đun).
Ngân sách m = round(θ n) chia cho các nhãn theo phương pháp Hamilton (có hạn ngạch tối thiểu tuỳ chọn);
trong mỗi nhãn chạy tham lam lười trên các luật DUY NHẤT với trọng số bội số (cho cùng Rep).
"""
import heapq
import numpy as np


def hamilton(sizes, m, min_per=None):
    sizes = np.asarray(sizes, float)
    base = np.zeros(len(sizes), int) if min_per is None else np.minimum(min_per, sizes).astype(int)
    rest = m - base.sum()
    if rest <= 0:
        return base
    cap = sizes - base
    q = cap / max(cap.sum(), 1) * rest
    a = np.floor(q).astype(int)
    for k in np.argsort(-(q - a))[: int(rest - a.sum())]:
        a[k] += 1
    return np.minimum(base + a, sizes.astype(int))


def _greedy(U, w, k):
    """Tham lam lười cho facility location trên các luật duy nhất U (u x r), trọng số w."""
    u = len(U); k = min(k, u)
    S = (U[:, None, :] == U[None, :, :]).mean(2)          # Sim = 1 - d_H trong cùng nhãn
    best = np.zeros(u); chosen = []
    heap = [(-(S[i] @ w), i, 0) for i in range(u)]; heapq.heapify(heap); it = 0
    while len(chosen) < k and heap:
        g, i, st = heapq.heappop(heap)
        if st == it:
            chosen.append(i); best = np.maximum(best, S[i]); it += 1
        else:
            heapq.heappush(heap, (-(np.maximum(S[i] - best, 0) @ w), i, it))
    return chosen, best


def sfkgs_select(rules, theta=0.3, min_per_class=0):
    """rules: danh sách luật (antecedent_tokens, consequent_token, tuỳ chọn weight = số dòng gốc của luật gộp).
    Trả về (chỉ số luật được chọn, Rep, m); n, ngân sách và bội số tính theo số dòng gốc."""
    ante = [tuple(r["antecedent_tokens"]) for r in rules]; lab = [r["consequent_token"] for r in rules]
    wr = np.array([float(r.get("weight", 1)) for r in rules])
    vocab = {t: i for i, t in enumerate(sorted({t for a in ante for t in a}))}
    X = np.array([[vocab[t] for t in a] for a in ante]); y = np.array(lab)
    n = int(wr.sum()); m = max(1, int(round(theta * n)))
    labels, inv_l = np.unique(y, return_inverse=True)
    counts = np.bincount(inv_l, weights=wr).astype(int)
    alloc = hamilton(counts, m, np.full(len(labels), min_per_class) if min_per_class else None)
    keep, rep = [], 0.0
    for L, k in zip(labels, alloc):
        idx = np.flatnonzero(y == L)
        U, first, inv, mult = np.unique(X[idx], axis=0, return_index=True, return_inverse=True, return_counts=True)
        mult = np.bincount(inv.ravel(), weights=wr[idx], minlength=len(U))
        ch, best = _greedy(U, mult, int(k))
        keep += [int(idx[first[c]]) for c in ch]
        rep += float(best @ mult)
    return sorted(keep), rep / n, m
