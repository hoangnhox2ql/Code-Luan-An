"""Training rule-base cleaning for Chapter 3 (rule_base = "dedup_conf", 02/11/2026).

A rule is (antecedent = vector of linguistic levels, label). On the training rule base of a fold (original
images + BorderlineSMOTE rows, after fuzzification):
  1. duplicate rules (same antecedent, same label) are kept once (weight 1);
  2. for a contradictory antecedent (rules with both labels) the label with the higher confidence
     conf(A -> l) = #{rows with antecedent A and label l} / #{rows with antecedent A} is kept and the other is
     removed; when both labels have the same confidence the antecedent carries no label information and both
     rules are removed (tie="drop"; "keep" keeps both).
Only training rows are used; validation and test rows never enter.
"""
from __future__ import annotations

import numpy as np


def clean_rules(R, y, tie="drop"):
    """Returns (R_clean, y_clean, info). Output rows are sorted by antecedent (np.unique order)."""
    R, y = np.asarray(R), np.asarray(y)
    if tie not in ("drop", "keep"):
        raise ValueError(tie)
    labels = np.unique(y)
    ante, inv = np.unique(R, axis=0, return_inverse=True)
    inv = inv.ravel()
    counts = np.zeros((len(ante), len(labels)), dtype=np.int64)
    np.add.at(counts, (inv, np.searchsorted(labels, y)), 1)
    present = counts > 0
    conflicting = present.sum(1) > 1
    best = counts.max(1, keepdims=True)
    winners = counts == best
    tied = conflicting & (winners.sum(1) > 1)
    keep = np.where(conflicting[:, None], winners, present)
    if tie == "drop":
        keep[tied] = False
    rows, cols = np.nonzero(keep)
    info = {"rows_in": int(len(y)), "antecedents": int(len(ante)), "rules_unique": int(present.sum()),
            "duplicate_rows_removed": int(len(y) - present.sum()), "conflicting_antecedents": int(conflicting.sum()),
            "ties": int(tied.sum()), "rules_out": int(len(rows)),
            "rules_out_per_label": {int(l): int((cols == i).sum()) for i, l in enumerate(labels)}}
    return ante[rows], labels[cols], info
