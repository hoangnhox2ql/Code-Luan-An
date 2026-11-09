"""Supplementary Chapter 4 code: options of the objective, subset rule bases, S-FKGS and kNN baselines."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from autograd import grad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from baselines import WeightedKNN
from compact_fkge import compress_rules, present_outside_part, select_rules, training_rules
from sfkgs import sfkgs_compact
from test_pipeline import duplicate_case, reference_module, reference_rulebase, teacher_targets
from train_fkge import CompactFKGE, build_teacher


def ratio_rulebase(rules, ratio, seed=7):
    """KB6 rule base as extra.Fold.rulebase builds it: ceil(r |R|) cleaned rules drawn without replacement."""
    n = len(rules["R"])
    return select_rules(rules, np.sort(np.random.default_rng(seed).choice(n, int(math.ceil(ratio * n)), replace=False)))


class ObjectiveOptionTests(unittest.TestCase):
    def setUp(self):
        self.rng, self.R, self.y, self.c, self.queries, self.rules, _ = duplicate_case()
        self.pF = self.rng.uniform(0.1, 1, (len(self.queries["R"]), 2))
        self.pF /= self.pF.sum(1, keepdims=True)
        self.yF = self.queries["y"].copy()

    def _compare(self, rules, reference_kwargs, compact_kwargs):
        q = self.queries
        e = teacher_targets(np.random.RandomState(4), rules, q, self.yF)
        merged, samples = reference_rulebase(rules, q)
        source = reference_module("fkge").FKGE(merged, d_e=4, lam_C=1e-3, seed=42, **reference_kwargs)
        Q, G, sw = source._query_matrix(samples), source._mask(samples), source._sample_w(samples)
        objective = lambda params: source.loss(params, Q, q["y"], self.pF, e, self.yF, G=G, sw=sw)
        expected_value, expected_gradient = float(objective(source.params)), grad(objective)(source.params)
        model = CompactFKGE(rules, q, self.pF, e, self.yF, d_e=4, seed=42, device="cpu", block_size=7,
                            **compact_kwargs)
        for actual, expected in zip(model.params, source.params):
            np.testing.assert_array_equal(actual.detach().numpy(), expected)
        value, gradient = model.value_grad()
        self.assertAlmostEqual(value, expected_value, places=10)
        for actual, expected in zip(gradient, expected_gradient):
            np.testing.assert_allclose(actual.numpy(), expected, atol=1e-10, rtol=1e-9)

    def test_ablation_and_negative_sample_options_match_reference(self):
        base = dict(lam_I=0.3, lam_P=0.5)
        for extra in ({"lam_R": 0.0}, {"lam_E": 0.0}, {"lam_N": 0.0}, {"lam_E": 0.0, "lam_N": 0.0},
                      {"lam_P": 0.0}, {"lam_I": 0.0}, {"K": 2}, {"K": 10}):
            with self.subTest(extra=extra):
                options = {**base, **extra}
                reference = {"K": 5, "lam_E": 1, "lam_N": 1, "lam_R": 1, "alpha": 0.3, **options}
                self._compare(self.rules, reference, {"alpha": 0.3, **options})

    def test_uniform_pooling_matches_reference(self):
        self._compare(self.rules, {"pooling": "mean", "lam_I": 0.3, "lam_P": 0.5},
                      {"alpha": "mean", "lam_I": 0.3, "lam_P": 0.5})

    def test_sfkgs_rule_base_with_all_training_queries_matches_reference(self):
        keep, _, _ = sfkgs_compact(self.rules["R"], self.rules["y"], self.rules["weights"], 0.4)
        self._compare(select_rules(self.rules, keep), {"alpha": 0.7, "lam_I": 0.5, "lam_P": 0.7},
                      {"alpha": 0.7, "lam_I": 0.5, "lam_P": 0.7})

    def test_sampled_rule_base_with_all_training_queries_matches_reference(self):
        for ratio in (0.4, 0.8):
            with self.subTest(ratio=ratio):
                rules = ratio_rulebase(self.rules, ratio)
                self.assertEqual(len(rules["R"]), math.ceil(ratio * len(self.rules["R"])))
                self._compare(rules, {"alpha": 0.7, "lam_I": 0.5, "lam_P": 0.7},
                              {"alpha": 0.7, "lam_I": 0.5, "lam_P": 0.7})

    def test_teacher_attributes_only_rules_present_outside_the_query_part(self):
        q = self.queries
        with tempfile.TemporaryDirectory() as directory:
            build_teacher({**q, "Rval": self.R[:10], "yval": self.y[:10]}, Path(directory), self.rules)
            targets = np.load(Path(directory) / "teacher_attribution.npy")
        for i, part in enumerate(q["parts"]):
            self.assertEqual(targets[i, ~present_outside_part(self.rules, part)].sum(), 0.0)
            self.assertAlmostEqual(targets[i].sum(), 1.0, places=12)


class SelectionTests(unittest.TestCase):
    def test_sfkgs_on_cleaned_rules_selects_as_reference(self):
        sfkgs = reference_module("sfkgs")
        rng = np.random.RandomState(3)
        for trial in range(4):
            base = rng.randint(0, 4, (60, 5))
            R = np.r_[base, base[rng.randint(0, 60, 40)]]
            y = (R[:, 0] + rng.randint(0, 2, len(R)) > 2).astype(int)
            c = compress_rules(R, y, [f"p{i % 37}" for i in range(len(R))], merge_signatures=False)
            rules, _, _ = training_rules(c["R"], c["y"], c["weights"], c["units"], c["units"] % 5)
            listed = [{"antecedent_tokens": [f"a{j:02}-L{v}" for j, v in enumerate(row)], "consequent_token": f"c{t}"}
                      for row, t in zip(rules["R"], rules["y"])]
            for theta in (0.1, 0.3, 0.6):
                ckeep, crep, cm = sfkgs_compact(rules["R"], rules["y"], rules["weights"], theta)
                keep, rep, m = sfkgs.sfkgs_select(listed, theta)
                self.assertEqual(sorted(ckeep), sorted(keep))
                self.assertEqual(cm, m)
                self.assertAlmostEqual(crep, rep, places=12)

    def test_weighted_knn_equals_knn_on_expanded_rules(self):
        rng = np.random.RandomState(5)
        E = rng.normal(size=(4 * 5 + 2, 6))
        R = rng.randint(0, 5, (30, 4))
        y = rng.randint(0, 2, 30)
        w = rng.randint(1, 4, 30).astype(float)
        Rq = rng.randint(0, 5, (12, 4))
        expanded = WeightedKNN(E, np.repeat(R, w.astype(int), 0), np.repeat(y, w.astype(int)),
                               np.ones(int(w.sum())), 7)
        np.testing.assert_allclose(WeightedKNN(E, R, y, w, 7).scores(Rq), expanded.scores(Rq))


if __name__ == "__main__":
    unittest.main()
