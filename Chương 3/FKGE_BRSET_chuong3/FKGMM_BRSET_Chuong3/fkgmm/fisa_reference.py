"""Bản dịch từng dòng của fisa_module.cpp (kho thanhst/Fuzzy-Knowledge-Graph) và hàm
min_max_normalize của FKG_S.py. Chỉ dùng làm "oracle" để kiểm thử fisa.py trên dữ liệu nhỏ
(độ phức tạp O(n^2 * C(r,4))). Nhãn lớp trong `base` là 1..n_classes như bản C++.
"""
from math import comb

import numpy as np


def calculateA(base):
    row, col = len(base), len(base[0])
    A = np.zeros((row, comb(col - 1, 4)))
    for r1 in range(row):
        temp = 0
        for a in range(0, col - 4):
            for b in range(a + 1, col - 3):
                for c in range(b + 1, col - 2):
                    for d in range(c + 1, col - 1):
                        count = 0
                        for r2 in range(row):
                            if (base[r1][a] == base[r2][a] and base[r1][b] == base[r2][b]
                                    and base[r1][c] == base[r2][c] and base[r1][d] == base[r2][d]):
                                count += 1
                        A[r1][temp] = count / row
                        temp += 1
    return A


def calculateM(base):
    row, col = len(base), len(base[0])
    M = np.zeros((row, col - 1))
    for t1 in range(row):
        for i in range(col - 1):
            k = 0
            for t2 in range(row):
                if base[t1][i] == base[t2][i] and base[t1][col - 1] == base[t2][col - 1]:
                    k += 1
            M[t1][i] = k / row
    return M


def calculateB(base, A, M):
    row, col = len(base), len(base[0])
    B = np.zeros((row, comb(col - 1, 3)))
    for r in range(row):
        temp = 0
        for a in range(0, col - 3):
            for b in range(a + 1, col - 2):
                for c in range(b + 1, col - 1):
                    B[r][temp] = A[r].sum() * min(M[r][a], M[r][b], M[r][c])
                    temp += 1
    return B


def calculateC(base, B, n_class):
    row, col = len(base), len(base[0])
    cs = comb(col - 1, 3)
    C = np.zeros((row, n_class * cs))
    for r1 in range(row):
        temp = 0
        for i in range(1, n_class + 1):
            for a in range(0, col - 3):
                for b in range(a + 1, col - 2):
                    for c in range(b + 1, col - 1):
                        for r2 in range(row):
                            if (base[r1][a] == base[r2][a] and base[r1][b] == base[r2][b]
                                    and base[r1][c] == base[r2][c] and base[r2][col - 1] == i):
                                C[r1][temp] += B[r2][temp % cs]
                        temp += 1
    return C


def min_max_normalize(C):
    C_min, C_max = C.min(axis=0), C.max(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Cn = (C - C_min) / (C_max - C_min)
    return np.where(np.isfinite(Cn), Cn, 0.0)   # cùng quy ước NaN -> 0 như fisa.py


def FISA(base, C, lst, n_classes):
    """Trả về (nhãn 1..K, D) giống FISA_with_confidence."""
    row, col = len(base), len(base[0])
    cols = comb(col - 1, 3)
    Cv = np.zeros((n_classes, cols))
    t = 0
    for a in range(0, col - 3):
        for b in range(a + 1, col - 2):
            for c in range(b + 1, col - 1):
                for r in range(row):
                    if base[r][a] == lst[a] and base[r][b] == lst[b] and base[r][c] == lst[c]:
                        label = base[r][col - 1] - 1
                        Cv[label][t] = C[r][t + label * cols]
                t += 1
    D = Cv.max(axis=1) + Cv.min(axis=1)
    return int(np.argmax(D)) + 1, D
