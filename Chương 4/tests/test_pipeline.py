"""Numerical equivalence to the Chapter 4 branch (reference/, 02/11/2026) and safeguards against leakage."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from autograd import grad
from imblearn.over_sampling import BorderlineSMOTE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compact_fkge import CompactFISA, compress_rules, connected_units, present_outside_part, training_rules
from preprocess import FusionPreprocessor, balance_training, balance_teacher_parts
from train_fkge import CompactFKGE, build_teacher, mask_entries
from run import synthetic_data


def reference_module(name):
    spec = importlib.util.spec_from_file_location(f"_source_{name}", ROOT / "reference" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokens(row):
    return [f"a{j:02}-L{value}" for j, value in enumerate(row)]


def reference_rulebase(rules, queries):
    """The cleaned rule base in the structures of the 02/11 branch (models/fkge.py): every rule weight 1; a rule
    whose rows all come from one unit u gets unit_counts {u: 1}, so the reference mask log(w - c) hides it from
    the queries of u only; samples carry group (unit) and weight (rows)."""
    sources = [set() for _ in range(len(rules["R"]))]
    for k, unit in zip(rules["copy_rule"], rules["copy_unit"]):
        sources[k].add(int(unit))
    rule_list = [{"id": k, "antecedent_tokens": tokens(row), "consequent_token": f"class-{label}", "weight": 1.0,
                  "unit_counts": {next(iter(sources[k])): 1.0} if len(sources[k]) == 1 else {}}
                 for k, (row, label) in enumerate(zip(rules["R"], rules["y"]))]
    samples = [{"membership": dict.fromkeys(tokens(row), 1.0), "label": f"class-{label}", "group": int(unit),
                "weight": float(w)}
               for row, label, unit, w in zip(queries["R"], queries["y"], queries["units"], queries["weights"])]
    vocab = sorted({t for r in rule_list for t in r["antecedent_tokens"]}) + ["class-0", "class-1"]
    return SimpleNamespace(rules=rule_list, vocab=vocab, class_tokens=["class-0", "class-1"]), samples


def duplicate_case(seed=8):
    """Duplicate rules inside one patient and across patients, conflicting rules; teacher part = unit % 5."""
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 3, (24, 4))
    R = np.r_[base, base[:6], base[3:9], base[:2], base[10:13]]
    y = np.r_[np.arange(24) % 2, np.arange(6) % 2, np.arange(3, 9) % 2, np.arange(2) % 2, 1 - np.arange(10, 13) % 2]
    patients = ([f"p{i}" for i in range(24)] + [f"p{i}" for i in range(6)] + [f"q{i}" for i in range(6)] + ["r", "r"]
                + [f"s{i}" for i in range(3)])
    compact = compress_rules(R, y, patients, merge_signatures=False)     # as run.prepare_fold
    parts = compact["units"] % 5
    rules, keep, info = training_rules(compact["R"], compact["y"], compact["weights"], compact["units"], parts)
    queries = {"R": compact["R"][keep], "y": compact["y"][keep], "weights": compact["weights"][keep],
               "units": compact["units"][keep], "parts": parts[keep]}
    return rng, R, y, compact, queries, rules, info


def teacher_targets(rng, rules, queries, yF):
    """Random teacher attribution: rules of label yF[q] that the query's teacher part may use."""
    e = rng.uniform(0.1, 1, (len(queries["R"]), len(rules["R"])))
    for q, part in enumerate(queries["parts"]):
        e[q] *= (rules["y"] == yF[q]) & present_outside_part(rules, part)
    return e / np.maximum(e.sum(1, keepdims=True), 1e-300)


def dense_mask(rules, units):
    query, rule, value = mask_entries(rules, units)
    log_weights = np.tile(np.log(rules["weights"]), (len(units), 1))
    log_weights[query, rule] = value
    return log_weights


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.rng, self.R, self.y, self.compact, self.queries, self.rules, self.info = duplicate_case()

    def test_conflicting_rules_removed_and_duplicates_kept_once(self):
        pairs = {(tuple(r), t) for r, t in zip(self.R, self.y)}
        conflicting = {a for a, t in pairs if (a, 1 - t) in pairs}
        self.assertGreater(len(conflicting), 0)
        expected = sorted((a, t) for a, t in pairs if a not in conflicting)
        self.assertEqual(sorted((tuple(r), t) for r, t in zip(self.rules["R"], self.rules["y"])), expected)
        np.testing.assert_array_equal(self.rules["weights"], 1.0)
        for k, (row, label) in enumerate(zip(self.rules["R"], self.rules["y"])):
            self.assertEqual(self.rules["rows"][k], ((self.R == row).all(1) & (self.y == label)).sum())
        kept_rows = np.array([tuple(r) not in conflicting for r in self.R])
        self.assertEqual(self.queries["weights"].sum(), kept_rows.sum())
        self.assertEqual(self.info["conflicting_rows_removed"], (~kept_rows).sum())
        self.assertEqual(self.info["conflicting_antecedents"], len(conflicting))

    def test_fisa_values_and_attributions_match_reference(self):
        merged, samples = reference_rulebase(self.rules, self.queries)
        source = reference_module("fisa").FISA(merged, aggregator="geomean", node_size=3, eps=1e-9).fit()
        optimized = CompactFISA(self.rules["R"], self.rules["y"], self.rules["weights"])
        query = self.R[:12]
        logs, _ = optimized.log_values(query)
        for index, row in enumerate(query):
            membership = dict.fromkeys(tokens(row), 1.0)
            D, _, _ = source.decision_values(membership)
            np.testing.assert_allclose(logs[index], np.log([D["class-0"] + 1e-12, D["class-1"] + 1e-12]), atol=1e-11)
            expected = np.zeros(len(self.rules["R"]))
            for rule, value in source.attribution(membership).items():
                expected[rule] = value
            label = int(np.argmax(logs[index]))
            np.testing.assert_allclose(optimized.attribution(query[index:index + 1], np.array([label]))[0], expected,
                                       atol=1e-11)

    def test_mask_matches_reference_log_weights(self):
        merged, samples = reference_rulebase(self.rules, self.queries)
        source = reference_module("fkge").FKGE(merged, d_e=4, seed=42)
        actual = dense_mask(self.rules, self.queries["units"])
        np.testing.assert_allclose(actual, source._mask(samples), atol=1e-14)
        _, _, value = mask_entries(self.rules, self.queries["units"])
        self.assertTrue((value == -1e6).any() and (value == 0).any())   # own-only rules hidden, shared rules kept

    def test_fkge_loss_and_gradient_match_reference_and_block_sizes(self):
        q, rules = self.queries, self.rules
        pF = self.rng.uniform(0.1, 1, (len(q["R"]), 2))
        pF /= pF.sum(1, keepdims=True)
        yF = (q["y"] + (np.arange(len(q["y"])) % 4 == 0)) % 2
        eF = teacher_targets(self.rng, rules, q, yF)
        merged, samples = reference_rulebase(rules, q)
        source = reference_module("fkge").FKGE(merged, alpha=0.3, d_e=4, K=5, lam_I=0.3, lam_P=0.5,
                                               lam_E=1, lam_N=1, lam_R=1, lam_C=1e-3, seed=42)
        Q, G, sw = source._query_matrix(samples), source._mask(samples), source._sample_w(samples)
        objective = lambda params: source.loss(params, Q, q["y"], pF, eF, yF, G=G, sw=sw)
        expected_value, expected_gradient = float(objective(source.params)), grad(objective)(source.params)
        for block_size in (2, 11, 1000):
            model = CompactFKGE(rules, q, pF, eF, yF, alpha=0.3, d_e=4, lam_I=0.3, lam_P=0.5, seed=42,
                                device="cpu", block_size=block_size)
            for actual, expected in zip(model.params, source.params):
                np.testing.assert_array_equal(actual.detach().numpy(), expected)
            value, gradient = model.value_grad()
            self.assertAlmostEqual(value, expected_value, places=10)
            for actual, expected in zip(gradient, expected_gradient):
                np.testing.assert_allclose(actual.numpy(), expected, atol=1e-10, rtol=1e-9)

    def test_teacher_matches_reference_cross_fitting(self):
        """build_teacher: the FISA of part p keeps the cleaned rules that also occur outside p (the keep-one rule base
        rebuilt without p); D, the decision and the attribution of each query match reference/fisa.py."""
        fisa = reference_module("fisa")
        q, rules = self.queries, self.rules
        with tempfile.TemporaryDirectory() as directory:
            build_teacher({**q, "Rval": self.R[:10], "yval": self.y[:10]}, Path(directory), rules)
            small = dict(np.load(Path(directory) / "teacher_small.npz"))
            targets = np.load(Path(directory) / "teacher_attribution.npy")
        merged, samples = reference_rulebase(rules, q)
        full = fisa.FISA(merged, aggregator="geomean", node_size=3, eps=1e-9).fit()
        for row, Dv in zip(self.R[:10], small["validation_log_values"]):
            D, _, _ = full.decision_values(dict.fromkeys(tokens(row), 1.0))
            np.testing.assert_allclose(Dv, np.log([D["class-0"] + 1e-12, D["class-1"] + 1e-12]), atol=1e-11)
        row_parts = (self.compact["units"] % 5)[self.compact["inverse"]]
        for part in range(5):
            present = present_outside_part(rules, part)
            outside = {(tuple(r), t) for r, t, p in zip(self.R, self.y, row_parts) if p != part}
            self.assertEqual([(tuple(r), t) in outside for r, t in zip(rules["R"], rules["y"])], present.tolist())
            kept = [r for k, r in enumerate(merged.rules) if present[k]]
            source = fisa.FISA(SimpleNamespace(rules=kept, vocab=merged.vocab, class_tokens=merged.class_tokens),
                               aggregator="geomean", node_size=3, eps=1e-9, decision="calibrated").fit()
            source.threshold = float(small["threshold"])
            for i in np.flatnonzero(q["parts"] == part):
                D, _, matched = source.decision_values(samples[i]["membership"])
                np.testing.assert_allclose(small["Dlog"][i], np.log([D["class-0"] + 1e-12, D["class-1"] + 1e-12]),
                                           atol=1e-11)
                self.assertEqual(small["yF"][i], merged.class_tokens.index(source._decide(D, matched)))
                expected = np.zeros(len(rules["R"]))
                for rule, value in source.attribution(samples[i]["membership"]).items():
                    expected[rule] = value
                np.testing.assert_allclose(targets[i], expected, atol=1e-11)


class LeakageTests(unittest.TestCase):
    def test_teacher_partitions_keep_real_patients_and_both_smote_parents_together(self):
        data, _ = synthetic_data(42)
        X = data[["Contrast Feature", "Energy Feature"]].to_numpy()
        original_y = data.diabetic_retinopathy.to_numpy()
        _, y, groups, parents, parts = balance_teacher_parts(X, original_y, data.patient_id, 42)
        self.assertGreater(len(parents), 0)
        for part in range(5):
            self.assertEqual(set(y[parts != part]), {0, 1})
        for offset, (left, right) in enumerate(parents):
            self.assertEqual(parts[len(data) + offset], parts[left])
            self.assertEqual(parts[len(data) + offset], parts[right])
        units = connected_units(groups, np.zeros(len(groups)), parents, merge_signatures=False)
        for unit in np.unique(units):
            self.assertEqual(len(np.unique(parts[units == unit])), 1)

    def test_equal_fuzzy_patterns_do_not_merge_different_real_patients(self):
        units = connected_units(["p1", "p2"], ["same_rule", "same_rule"], merge_signatures=False)
        self.assertNotEqual(units[0], units[1])

    def test_query_sees_exactly_the_rules_of_other_units(self):
        """Mask = keep-one rule base rebuilt without the query's unit (restricted to the cleaned rules)."""
        _, R, y, compact, queries, rules, _ = duplicate_case(11)
        units = compact["original_units"]
        visible = dense_mask(rules, queries["units"]) == 0
        for q, unit in enumerate(queries["units"]):
            others = {(tuple(r), t) for r, t, u in zip(R, y, units) if u != unit}
            self.assertEqual(visible[q].tolist(), [(tuple(r), t) in others for r, t in zip(rules["R"], rules["y"])])
        self.assertTrue((~visible).any())

    def test_smote_matches_original_and_every_child_is_linked_to_both_parents(self):
        rng = np.random.RandomState(19)
        X = rng.normal(size=(90, 4))
        y = np.r_[np.zeros(70, dtype=int), np.ones(20, dtype=int)]
        patients = [f"p{i}" for i in range(len(y))]
        actual, target, groups, parents = balance_training(X, y, patients, 42)
        expected, expected_y = BorderlineSMOTE(random_state=42, k_neighbors=5, m_neighbors=10).fit_resample(X, y)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(target, expected_y)
        self.assertGreater(len(parents), 0)
        units = connected_units(groups, np.arange(len(groups)), parents)
        for offset, (left, right) in enumerate(parents):
            self.assertEqual(units[len(y) + offset], units[left])
            self.assertEqual(units[len(y) + offset], units[right])

    def test_validation_does_not_change_fitted_medians_or_feature_selection(self):
        data, _ = synthetic_data(42)
        train, val = data.iloc[:150].copy(), data.iloc[150:].copy()
        model = FusionPreprocessor(2, 2).fit(train, train.diabetic_retinopathy)
        medians = [branch[1].copy() for branch in model.branches]
        selected = model.feature_names.copy()
        val.loc[:, "patient_age"] = 1e12
        model.transform(val)
        self.assertEqual(model.feature_names, selected)
        for expected, branch in zip(medians, model.branches):
            np.testing.assert_array_equal(expected, branch[1])


if __name__ == "__main__":
    unittest.main()
