"""KB-E6 (cost versus number of attributes r): tensor fusion branch, trajectory stopping and R4 implementations."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build import IMAGE_COLUMNS, TABLE_COLUMNS
from compact_fkge import CompactFISA
from kb_r import PrecomputedFKGE, TableFISA, operation_counts, select_epoch, sequential_log_values, trace_fit
from preprocess import FusionPreprocessor
from run import prepare_fold, synthetic_data
from train_fkge import CompactFKGE, build_teacher, training_data


class TensorFusionTests(unittest.TestCase):
    def setUp(self):
        data, _ = synthetic_data(42)
        self.train, self.val = data.iloc[:150].copy(), data.iloc[150:].copy()
        self.labels = self.train.diabetic_retinopathy

    def test_zero_tensor_features_leave_the_pipeline_unchanged(self):
        base = FusionPreprocessor(2, 3).fit(self.train, self.labels)
        same = FusionPreprocessor(2, 3, 0).fit(self.train, self.labels)
        self.assertIsNone(same.tensor)
        self.assertEqual(base.feature_names, same.feature_names)
        np.testing.assert_array_equal(base.transform(self.val), same.transform(self.val))
        del base.tensor                                   # a preprocessor pickled before KB-E6
        np.testing.assert_array_equal(base.transform(self.val), same.transform(self.val))

    def test_tensor_features_are_fitted_on_training_only(self):
        model = FusionPreprocessor(2, 3, 4).fit(self.train, self.labels)
        plain = FusionPreprocessor(2, 3).fit(self.train, self.labels)
        out = model.transform(self.val)
        self.assertEqual(out.shape[1], 2 + 3 + 4)
        np.testing.assert_array_equal(out[:, :5], plain.transform(self.val))
        scaled = [branch[2].transform(self.val[branch[0]].fillna(branch[1])) for branch in model.branches]
        products = (scaled[0][:, :, None] * scaled[1][:, None, :]).reshape(len(self.val), -1)
        scaler, selector = model.tensor
        expected = (products - scaler.mean_) / scaler.scale_
        np.testing.assert_allclose(out[:, 5:], expected[:, selector.get_support()], atol=1e-12)
        self.assertEqual(len(model.feature_names), 9)
        self.assertTrue(all(" x " in name for name in model.feature_names[5:]))
        self.assertEqual(scaler.n_features_in_, len(IMAGE_COLUMNS) * len(TABLE_COLUMNS))


class TrajectoryAndCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        data, assignments = synthetic_data(42)
        directory = Path(cls.tmp.name) / "fold"
        prepare_fold(data, assignments, 1, directory, 42, {"k_img": 2, "k_tab": 2, "n_tensor": 2})
        with np.load(directory / "prepared.npz") as archive:
            cls.prepared = {name: archive[name] for name in archive.files}
        cls.rules, cls.queries, _ = training_data(cls.prepared)
        cache = Path(cls.tmp.name) / "teacher"
        cache.mkdir()
        build_teacher({**cls.queries, "Rval": cls.prepared["Rval"], "yval": cls.prepared["yval"]}, cache, cls.rules)
        cls.small = dict(np.load(cache / "teacher_small.npz"))
        cls.attribution = np.load(cache / "teacher_attribution.npy", mmap_mode="r")

    @classmethod
    def tearDownClass(cls):
        del cls.attribution
        cls.tmp.cleanup()

    def model(self):
        return CompactFKGE(self.rules, self.queries, self.small["pF"], self.attribution, self.small["yF"], alpha=0.7,
                           d_e=4, lam_I=0.3, lam_P=0.5, seed=42, device="cpu", block_size=17)

    def test_rule_a_on_the_trajectory_is_compact_fkge_fit(self):
        Rval, yval = self.prepared["Rval"], self.prepared["yval"]
        fitted = self.model().fit(Rval, yval, max_epochs=8, patience=3)
        trace, snapshots = trace_fit(self.model(), Rval, yval, 8)
        index = select_epoch(trace, "val_logloss", False, 3)
        self.assertEqual(trace[index]["val_logloss"], fitted.val_loss)
        for actual, expected in zip(snapshots[index], fitted.params):
            np.testing.assert_array_equal(actual, expected.detach().numpy())

    def test_r4_implementations_give_the_same_values(self):
        model = self.model()
        params = [p.detach().numpy() for p in model.params]
        R = self.prepared["Rtest"]
        np.testing.assert_allclose(PrecomputedFKGE(model, params).probabilities(R), model.probabilities(R), atol=1e-12)
        fisa = CompactFISA(self.rules["R"], self.rules["y"], self.rules["weights"])
        logs, _ = fisa.log_values(R)
        np.testing.assert_allclose(TableFISA(fisa).log_values(R), logs, atol=1e-12)
        for row, expected in zip(R[:5], logs[:5]):
            np.testing.assert_allclose(sequential_log_values(fisa, row), expected, atol=1e-12)

    def test_operation_counts(self):
        counts = operation_counts(16, 1000, 16)
        self.assertEqual(counts["nodes"], 560)
        self.assertEqual(counts["ops_fisa_table"], 14 * 560)
        self.assertEqual(counts["ops_fisa_sequential"], 8 * 560 * 1000 + 4 * 560)
        self.assertEqual(counts["ops_fkge"], 1000 * 36 + 16 * 16 + 32)


if __name__ == "__main__":
    unittest.main()
