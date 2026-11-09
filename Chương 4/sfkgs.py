"""S-FKGS (Chapter 2) on the compact weighted rule base: label-stratified greedy facility location.

Sim(R_i, R_j) = 1 - d_H(R_i, R_j) within a label (d_H: fraction of differing antecedents), Rep(S) =
(1/n) sum_r max_{s in S} Sim(r, s) over all n training rule rows (weights = multiplicities). Budget
m = round(theta n) is split across labels by Hamilton's method; within a label, lazy greedy runs on the
unique antecedents with multiplicity weights. Identical selection to reference/sfkgs.py, but similarity
rows are computed on demand so memory is O(u) instead of O(u^2).
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
    u = len(U)
    k = min(k, u)
    row = lambda i: (U == U[i]).mean(1)
    best = np.zeros(u)
    chosen = []
    heap = [(-(row(i) @ w), i, 0) for i in range(u)]
    heapq.heapify(heap)
    it = 0
    while len(chosen) < k and heap:
        _, i, stamp = heapq.heappop(heap)
        S = row(i)
        if stamp == it:
            chosen.append(i)
            best = np.maximum(best, S)
            it += 1
        else:
            heapq.heappush(heap, (-(np.maximum(S - best, 0) @ w), i, it))
    return chosen, best


def sfkgs_compact(R, y, weights, theta=0.3):
    """Returns (indices of kept rows of R, Rep, m). On the cleaned rule base (unique, non-conflicting rules, weight 1)
    each chosen antecedent is one rule, kept once, as the row-level reference keeps one rule row."""
    R, y, weights = np.asarray(R), np.asarray(y), np.asarray(weights)
    if not np.allclose(weights, np.round(weights)) or weights.min() < 1:
        raise ValueError("S-FKGS needs integer rule multiplicities")
    n = int(round(weights.sum()))
    m = max(1, int(round(theta * n)))
    labels = np.unique(y)
    counts = np.array([int(round(weights[y == label].sum())) for label in labels])
    keep, rep = [], 0.0
    for label, k in zip(labels, hamilton(counts, m)):
        idx = np.flatnonzero(y == label)
        U, first, inverse = np.unique(R[idx], axis=0, return_index=True, return_inverse=True)
        mult = np.bincount(inverse.ravel(), weights=weights[idx], minlength=len(U))
        chosen, best = _greedy(U, mult, int(k))
        keep += [int(idx[first[c]]) for c in chosen]
        rep += float(best @ mult)
    return np.array(sorted(keep)), rep / n, m
