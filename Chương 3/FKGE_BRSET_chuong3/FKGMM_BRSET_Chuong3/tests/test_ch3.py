"""Chapter 3 code: verbatim port of the fusion strategies, equality with the reference pipeline on real
BRSET rows, train-only fitting, data layout, FISA-GM checks of FKGMM_BRSET3_nested_20261102."""
import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chuong3 as C3
import fusion_strategies as FS
from build import TARGET

SOURCE = ROOT / "reference/preprocess_source/Preprocess_kfold_feature_selection.py"
SOURCE_HEAD = ROOT / "reference/preprocess_source/Preprocess_kfold_feature_selection_HEAD_43a784c.py"
CFG = SimpleNamespace(**{k: v for k, v in json.loads((ROOT / "config_ch3.json").read_text()).items()
                         if k != "smoke_overrides"})


def reference_preprocess():
    spec = importlib.util.spec_from_file_location("_reference_preprocess", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def segments(path):
    text = path.read_text(encoding="utf-8")
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.get_source_segment(text, node)
    return out


def real_subset(fold=1, n_train_patients=700, n_test_patients=200, seed=0):
    """Patient-sampled rows of one package fold (both classes kept), small enough for the wrapper search."""
    data = pd.read_csv(ROOT / "data/fusion_features.csv", dtype={"image_id": str, "patient_id": str},
                       float_precision="round_trip")
    fold_of = pd.read_csv(ROOT / "data/fold_assignments.csv", dtype={"image_id": str}).fold.to_numpy()
    rng = np.random.RandomState(seed)
    parts = []
    for rows, n in ((data[fold_of != fold], n_train_patients), (data[fold_of == fold], n_test_patients)):
        pos = rows[rows[TARGET] == 1].patient_id.unique()
        neg = np.setdiff1d(rows.patient_id.unique(), pos)
        keep = np.r_[rng.choice(pos, n // 4, replace=False), rng.choice(neg, n - n // 4, replace=False)]
        parts.append(rows[rows.patient_id.isin(keep)].reset_index(drop=True))
    return parts


class VerbatimPortTests(unittest.TestCase):
    def test_every_ported_function_and_constant_is_identical_to_the_reference(self):
        ported, reference, head = segments(ROOT / "fusion_strategies.py"), segments(SOURCE), segments(SOURCE_HEAD)
        names = [n for n in ported if n != "prepare_fusion_tensor_cross_svd_fold"]
        self.assertGreaterEqual(len(names), 30)
        for name in names:
            self.assertEqual(ported[name], reference[name], name)
        legacy = head["prepare_fusion_tensor_fold"].replace("def prepare_fusion_tensor_fold(",
                                                            "def prepare_fusion_tensor_cross_svd_fold(", 1)
        self.assertEqual(ported["prepare_fusion_tensor_cross_svd_fold"], legacy)


class ReferencePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = reference_preprocess()
        cls.train, cls.test = real_subset()

    def reference_fold(self, group, fold_seed=43):
        """Preprocess_kfold_feature_selection.prepare_modality for one fold, step by step."""
        ref, cols = self.ref, C3.source_columns(group)
        train_x, test_x = self.train[cols].reset_index(drop=True), self.test[cols].reset_index(drop=True)
        train_y, test_y = self.train[TARGET].reset_index(drop=True), self.test[TARGET].reset_index(drop=True)
        if group == "fusion_tensor":
            # the reference read a pre-imputed CSV; this package fills with train medians first
            train_x, test_x = ref.clean_numeric_split(train_x, test_x)
        prepare = {"image": ref.prepare_image_fold, "table": ref.prepare_table_fold,
                   "fusion": ref.prepare_fusion_fold}.get(group)
        if prepare:
            tr, te, cluster, rows = prepare(train_x, test_x, train_y, CFG, fold_seed)
        else:
            tr, te, cluster, rows = ref.prepare_fusion_variant_fold(group, train_x, test_x, train_y, CFG, fold_seed)
        tr, te, rows, dropped = ref.drop_train_constant_columns(tr, te, rows)
        train_fold, test_fold = ref.build_fis_frame(tr, train_y), ref.build_fis_frame(te, test_y)
        if dropped:
            cluster = [v for i, v in enumerate(cluster[:-1]) if i not in set(dropped)] + [cluster[-1]]
        train_fold, status = ref.apply_train_smote(train_fold, seed=fold_seed, max_k_neighbors=CFG.smote_k_neighbors)
        return train_fold, test_fold, cluster, pd.DataFrame(rows), status

    def test_all_seven_groups_equal_reference_on_real_brset_rows(self):
        for group in C3.GROUPS:
            with self.subTest(group=group):
                cols = C3.source_columns(group)
                tr, te, info = C3.prepare_group_fold(group, self.train[cols], self.test[cols], self.train[TARGET],
                                                     self.test[TARGET], CFG, 43)
                rtr, rte, rcluster, rrows, rstatus = self.reference_fold(group)
                pd.testing.assert_frame_equal(tr, rtr)
                pd.testing.assert_frame_equal(te, rte)
                self.assertEqual(info["cluster"], rcluster)
                self.assertEqual(info["smote"], rstatus)
                pd.testing.assert_frame_equal(pd.DataFrame(info["feature_rows"]), rrows)
                self.assertFalse(tr.isna().any().any() or te.isna().any().any(), "missing values reached FIS input")
        self.assertTrue(self.train[C3.TABLE_COLUMNS].isna().any().any(), "subset should exercise missing values")

    def test_tensor_is_algorithm_3_5_with_rho_6(self):
        cols = C3.source_columns("fusion_tensor")
        tr, te, info = C3.prepare_group_fold("fusion_tensor", self.train[cols], self.test[cols], self.train[TARGET],
                                             self.test[TARGET], CFG, 43)
        self.assertEqual(tr.shape[1] - 1, 2 * CFG.tensor_rank)
        self.assertEqual(sorted({r["branch"] for r in info["feature_rows"]}), ["tensor_img", "tensor_tab"])

    def test_outputs_on_training_rows_do_not_depend_on_test_rows(self):
        changed = self.test.copy()
        numeric = [*C3.TABLE_COLUMNS, *C3.IMAGE_COLUMNS]
        changed[numeric] = changed[numeric] * 3.0 + 7.0
        changed[TARGET] = 1 - changed[TARGET]
        for group in C3.GROUPS:
            with self.subTest(group=group):
                cols = C3.source_columns(group)
                a = C3.prepare_group_fold(group, self.train[cols], self.test[cols], self.train[TARGET],
                                          self.test[TARGET], CFG, 43)
                b = C3.prepare_group_fold(group, self.train[cols], changed[cols], self.train[TARGET],
                                          changed[TARGET], CFG, 43)
                pd.testing.assert_frame_equal(a[0], b[0])
                self.assertEqual(a[2]["cluster"], b[2]["cluster"])
                pd.testing.assert_frame_equal(pd.DataFrame(a[2]["feature_rows"]), pd.DataFrame(b[2]["feature_rows"]))

    def test_smote_rows_are_positive_and_appended_after_real_rows(self):
        cols = C3.source_columns("fusion")
        tr, _, info = C3.prepare_group_fold("fusion", self.train[cols], self.test[cols], self.train[TARGET],
                                            self.test[TARGET], CFG, 43)
        n = len(self.train)
        self.assertTrue(info["smote"].startswith("borderline_smote"))
        self.assertTrue((tr[TARGET].iloc[n:] == 1).all())
        self.assertEqual(int(tr[TARGET].sum()), len(tr) - int((self.train[TARGET] == 0).sum()))


class LayoutAndFisaTests(unittest.TestCase):
    def test_smoke_layout_is_read_by_chapter3_core(self):
        from kfold.core import audit, list_folds, load
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {k: v for k, v in vars(CFG).items()}
            C3.prepare(Path(tmp), C3.GROUPS, [1, 2], cfg, smoke=True, resume=False)
            root = str(Path(tmp) / "data")
            au = audit(root)
            self.assertEqual(len(au), 14)
            self.assertTrue((au.patient_overlap == 0).all() and au.smote_rows_all_positive.all())
            for f in list_folds(root):
                ref = load(root, "table", f)
                for g in C3.GROUPS[1:]:
                    D = load(root, g, f)
                    self.assertEqual(D["ids_te"].image_id.tolist(), ref["ids_te"].image_id.tolist())
                    self.assertEqual(D["n_orig_train"], ref["n_orig_train"])
            with self.assertRaises(SystemExit):        # same out without --resume is refused
                C3.prepare(Path(tmp), C3.GROUPS, [1, 2], cfg, smoke=True, resume=False)
            cfg["k_img"] = 5                            # other config on the same out is refused even with --resume
            with self.assertRaises(SystemExit):
                C3.prepare(Path(tmp), C3.GROUPS, [1, 2], cfg, smoke=True, resume=True)

    def test_predictions_are_saved_out_of_fold_and_reproduce_metrics_and_figures(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        from kfold.core import run
        from kfold.plots import load_predictions, render_plots
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {k: v for k, v in vars(CFG).items()}
            C3.prepare(Path(tmp), C3.GROUPS, [1, 2], cfg, smoke=True, resume=False)
            res = Path(tmp) / "ket_qua"
            df = run(str(Path(tmp) / "data"), str(res), ["fold_01", "fold_02"], ["M1", "M2", "M5"])
            pred = load_predictions(str(res))
            self.assertEqual(len(pred), sum(len(pd.read_csv(Path(tmp) / f"data/fusion/fold_0{k}/test_ids.csv")) for k in (1, 2)))
            for _, r in df.iterrows():
                g = pred[pred.fold == r.fold]
                key = f"{r.scenario}|{r.model}|{r.data}"
                self.assertAlmostEqual(roc_auc_score(g.y, g[key]), r.auc_roc, places=12)
                self.assertAlmostEqual(average_precision_score(g.y, g[key]), r.auc_pr, places=12)
            paths = render_plots(str(res))
            for name in ("roc_pr_kb_m1_m2", "roc_pr_kb_m5", "ap_theo_fold"):
                for ext in ("png", "pdf"):
                    self.assertTrue((res / "hinh" / f"{name}.{ext}").stat().st_size > 1000, name)
            pooled = pd.read_csv(res / "hinh" / "pooled_oof.csv")
            self.assertTrue((pooled.n_images == len(pred)).all() and (pooled.n_folds == 2).all())
            self.assertEqual(len(paths), 7)

    def test_geomean_formula_and_fisa_equivalence_with_cpp_translation(self):
        from fkgmm import fisa_reference as ref
        from fkgmm.fisa import FKG
        from kfold.core import fisa_gm
        rng = np.random.default_rng(0)
        X = rng.integers(0, 3, (150, 5))
        y = (X[:, 0] + rng.integers(0, 2, 150) > 2).astype(int)
        Q = X[:30]
        Cv = FKG(c_norm="none").fit(X, y)._lookup(Q)
        gm = FKG(c_norm="none", aggregator="geomean", eps=1e-9).fit(X, y).decision_values(Q)
        np.testing.assert_allclose(np.log(gm), np.log(Cv + 1e-9).mean(2), atol=1e-10)
        pred, score, _ = fisa_gm(X, y, Q)
        np.testing.assert_allclose(score, np.log(gm[:, 1]) - np.log(gm[:, 0]), atol=1e-10)
        self.assertTrue(((score > 0).astype(int) == pred).all())
        for seed, n, r, levels, K, norm in [(0, 40, 5, 3, 2, "minmax"), (1, 60, 6, 3, 2, "none"),
                                            (2, 50, 5, 5, 3, "minmax"), (4, 45, 6, 3, 4, "none")]:
            rng = np.random.default_rng(seed)
            Xr = rng.integers(0, levels, size=(n, r))
            yr = rng.integers(1, K + 1, size=n)
            yr[:K] = np.arange(1, K + 1)
            base = np.column_stack([Xr, yr]).tolist()
            A, M = ref.calculateA(base), ref.calculateM(base)
            B = ref.calculateB(base, A, M)
            Cr = ref.calculateC(base, B, K)
            if norm == "minmax":
                Cr = ref.min_max_normalize(Cr)
            model = FKG(c_norm=norm).fit(Xr, yr)
            Xq = rng.integers(0, levels + 1, size=(25, r))
            D = model.decision_values(Xq)
            for i, x in enumerate(Xq):
                _, D_ref = ref.FISA(base, Cr, x.tolist(), K)
                np.testing.assert_allclose(D[i], D_ref, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(model.B_, B, rtol=1e-12)

    def test_pr_auc_metrics_are_ap_and_trapezoid_reported_separately(self):
        from sklearn.metrics import auc, average_precision_score, precision_recall_curve
        from kfold.core import metrics
        rng = np.random.default_rng(4)
        y = (rng.random(400) < 0.07).astype(int)
        score = y * 0.8 + rng.normal(0, 0.7, 400)
        m = metrics(y, (score > 0).astype(int), score)
        precision, recall, _ = precision_recall_curve(y, score)
        self.assertAlmostEqual(m["auc_pr"], average_precision_score(y, score), places=12)
        self.assertAlmostEqual(m["pr_auc_trap"], auc(recall, precision), places=12)
        self.assertNotAlmostEqual(m["auc_pr"], m["pr_auc_trap"], places=6)
        self.assertAlmostEqual(m["pos_rate"], y.mean(), places=12)
        null = metrics(y, np.zeros(400, int), rng.random(400))           # random scores: PR-AUC ~ prevalence
        self.assertLess(abs(null["auc_pr"] - y.mean()), 0.05)

    def test_report_shows_ap_and_pr_auc_columns(self):
        from kfold import report
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"scenario": "M1", "model": name, "data": data, "fold": f"fold_0{k}", "acc": .9, "f1": .5,
                     "auc_roc": .9, "auc_pr": .4 + k / 100, "pr_auc_trap": .39 + k / 100, "sens": .8, "spec": .9,
                     "balacc": .85, "acc_majority": .934}
                    for k in range(1, 4) for name, data in (("FKG-MM (đề xuất)", "FKG"), ("FKG-UM (Bảng)", "FKG"))]
            pd.DataFrame(rows).to_csv(Path(tmp) / "per_fold.csv", index=False)
            text = report.build(tmp)
        self.assertIn("| AUC | AP | PR-AUC |", text)
        self.assertIn("42.0 ± 1.0", text)                 # AP of FKG-MM over 3 folds, in %
        self.assertIn("41.0 ± 1.0", text)                 # PR-AUC (trapezoid)
        line = next(l for l in text.splitlines() if l.startswith("- FKG-MM − FKG-UM (Bảng):"))
        self.assertIn("; AP Δ = +0.0; thắng 0/3", line)    # paired AP difference is reported (equal AP here)

    def test_nadeau_bengio_uses_the_fold_sizes_of_the_audit(self):
        from kfold import report
        a = pd.DataFrame({"fold": [1, 2, 3], "auc_roc": [0.80, 0.82, 0.79]}).set_index("fold")
        b = pd.DataFrame({"fold": [1, 2, 3], "auc_roc": [0.78, 0.81, 0.79]}).set_index("fold")
        report._NB.update(n_tr=13000.0, n_te=3250.0)
        p_quarter = report._pair(a, b)["p"]
        self.assertAlmostEqual(p_quarter, report._pair(a, b, n_tr=4.0, n_te=1.0)["p"], places=12)
        self.assertGreater(p_quarter, report._pair(a, b, n_tr=1778, n_te=242)["p"])



class AuditFixTests(unittest.TestCase):
    def fold(self, seed=3, n=400, n_smote=150, n_test=200):
        rng = np.random.RandomState(seed)
        y = (rng.rand(n + n_test) < 0.3).astype(int)
        X = np.column_stack([y + rng.normal(0, 0.6, len(y)), y + rng.normal(0, 0.8, len(y)),
                             rng.normal(size=len(y)), rng.normal(size=len(y)), (rng.rand(len(y)) < 0.5).astype(float)])
        Xs = X[:n][y[:n] == 1][:n_smote] + rng.normal(0, 0.05, (min(n_smote, int(y[:n].sum())), 5))
        ids = pd.DataFrame({"patient_id": [f"p{i // 2}" for i in range(n)], "image_id": [f"i{i}" for i in range(n)]})
        return {"Xtr": np.r_[X[:n], Xs], "ytr": np.r_[y[:n], np.ones(len(Xs), int)], "Xte": X[n:], "yte": y[n:],
                "ids_tr": ids, "n_orig_train": n}

    def test_unweighted_variant_equals_core_fisa_gm(self):
        from kfold.core import FCMFuzzifier, fisa_gm
        from kfold.weighted import fisa_gm_variant
        D = self.fold()
        fz = FCMFuzzifier(5).fit(D["Xtr"])
        pred, score, _ = fisa_gm(fz.transform(D["Xtr"]), D["ytr"], fz.transform(D["Xte"]))
        p2, s2, _ = fisa_gm_variant(D)
        np.testing.assert_array_equal(pred, p2)
        np.testing.assert_allclose(score, s2, rtol=0, atol=1e-12)

    def test_weights_ignore_test_rows_and_smote_rows_and_drop_noise_triples(self):
        from kfold.core import FCMFuzzifier
        from kfold.weighted import fisa_gm_variant, triple_weights
        D = self.fold()
        n0 = D["n_orig_train"]
        R = FCMFuzzifier(5).fit(D["Xtr"]).transform(D["Xtr"])
        w, auc = triple_weights(R[:n0], D["ytr"][:n0], D["ids_tr"].patient_id.to_numpy())
        self.assertGreater(w[0], 0)                           # triple (0,1,2) contains both informative columns
        noise = [i for i, t in enumerate(__import__("itertools").combinations(range(5), 3)) if not ({0, 1} & set(t))]
        self.assertTrue(np.all(w[noise] <= 0.06))
        changed = dict(D, Xte=D["Xte"] * 5 + 1, yte=1 - D["yte"])
        a, b = fisa_gm_variant(D, weighted=True)[2], fisa_gm_variant(changed, weighted=True)[2]
        self.assertEqual(a, b)
        smote_changed = dict(D, ytr=np.r_[D["ytr"][:n0], 1 - D["ytr"][n0:]])
        from kfold.weighted import BinaryAwareFuzzifier
        self.assertIsNotNone(BinaryAwareFuzzifier)
        R2 = FCMFuzzifier(5).fit(smote_changed["Xtr"]).transform(smote_changed["Xtr"])
        w2, _ = triple_weights(R2[:n0], smote_changed["ytr"][:n0], D["ids_tr"].patient_id.to_numpy())
        np.testing.assert_array_equal(w, w2)

    def test_core_default_is_gm_w_and_equals_the_audit_variant(self):
        from kfold.core import DEFAULTS, FCMFuzzifier, fisa_gm
        from kfold.weighted import fisa_gm_variant, triple_weights
        self.assertEqual(DEFAULTS["aggregator"], "geomean_w")
        D = self.fold()
        n0 = D["n_orig_train"]
        fz = FCMFuzzifier(5).fit(D["Xtr"])
        R = fz.transform(D["Xtr"])
        w, _ = triple_weights(R[:n0], D["ytr"][:n0], D["ids_tr"].patient_id.to_numpy())
        pred, score, _ = fisa_gm(R, D["ytr"], fz.transform(D["Xte"]), weights=w)
        p2, s2, _ = fisa_gm_variant(D, weighted=True)
        np.testing.assert_array_equal(pred, p2)
        np.testing.assert_allclose(score, s2, rtol=0, atol=1e-12)
        # constant weights give back FISA-GM exactly
        p3, s3, _ = fisa_gm(R, D["ytr"], fz.transform(D["Xte"]), weights=np.full(len(w), 0.37))
        p4, s4, _ = fisa_gm(R, D["ytr"], fz.transform(D["Xte"]))
        np.testing.assert_array_equal(p3, p4)
        np.testing.assert_allclose(s3, s4, rtol=0, atol=1e-10)

    def test_clean_rules_dedup_conflict_and_tie(self):
        from kfold.rulebase import clean_rules
        R = np.array([[0, 1], [0, 1], [0, 1],            # A: 3 x label 0 ...
                      [0, 1],                            # ... and 1 x label 1 -> keep 0
                      [2, 2], [2, 2],                    # B: 1 x label 0, 1 x label 1 -> tie, drop both
                      [1, 0], [1, 0], [1, 0],            # C: only label 1 (duplicates) -> keep once
                      [3, 3]])                           # D: single label 0
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 0])
        Rc, yc, info = clean_rules(R, y)
        got = {(tuple(r), int(l)) for r, l in zip(Rc, yc)}
        self.assertEqual(got, {((0, 1), 0), ((1, 0), 1), ((3, 3), 0)})
        self.assertEqual((info["conflicting_antecedents"], info["ties"], info["rules_out"]), (2, 1, 3))
        Rk, yk, _ = clean_rules(R, y, tie="keep")
        self.assertEqual({(tuple(r), int(l)) for r, l in zip(Rk, yk)}, got | {((2, 2), 0), ((2, 2), 1)})
        self.assertEqual(len({tuple(r) for r in Rc}), len(Rc))   # one rule per antecedent after cleaning

    def test_core_dedup_conf_uses_the_cleaned_rule_base(self):
        import kfold.core as core
        from kfold.rulebase import clean_rules
        from kfold.weighted import triple_weights
        D = self.fold()
        n0 = D["n_orig_train"]
        fz = core.FCMFuzzifier(5).fit(D["Xtr"])
        R = fz.transform(D["Xtr"])
        w, _ = triple_weights(R[:n0], D["ytr"][:n0], D["ids_tr"].patient_id.to_numpy())
        Rc, yc, _ = clean_rules(R, D["ytr"])
        pred, score, _ = core.fisa_gm(Rc, yc, fz.transform(D["Xte"]), weights=w)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            for g in core.MODALITIES:
                d = root / g / "fold_01"
                d.mkdir(parents=True)
                pd.DataFrame(np.c_[D["Xtr"], D["ytr"]], columns=[*map(str, range(5)), TARGET]).to_csv(d / "train_data.csv", index=False)
                pd.DataFrame(np.c_[D["Xte"], D["yte"]], columns=[*map(str, range(5)), TARGET]).to_csv(d / "test_data.csv", index=False)
                D["ids_tr"].to_csv(d / "train_ids.csv", index=False)
                pd.DataFrame({"patient_id": [f"t{i}" for i in range(len(D["yte"]))],
                              "image_id": [f"j{i}" for i in range(len(D["yte"]))]}).to_csv(d / "test_ids.csv", index=False)
            for rule_base, expected in (("dedup_conf", score), ("all", None)):
                df = core.run(str(root), str(Path(tmp) / rule_base), ["fold_01"], ["EPS"],
                              {"rule_base": rule_base, "eps_grid": [1e-9]})
                if expected is not None:
                    from sklearn.metrics import roc_auc_score
                    self.assertAlmostEqual(df.auc_roc.iloc[0], roc_auc_score(D["yte"], expected), places=12)
                    self.assertEqual(df.rules_kept.iloc[0], len(yc))
                else:
                    self.assertNotIn("rules_kept", df.columns)

    def test_fcm_fit_original_gives_two_levels_for_binary_and_matches_manual_pipeline(self):
        import kfold.core as core
        from kfold.weighted import triple_weights
        D = self.fold()
        n0 = D["n_orig_train"]
        D["Xtr"][n0:, 4] = np.linspace(0.05, 0.95, len(D["Xtr"]) - n0)        # SMOTE-like interpolation of a binary column
        self.assertEqual(len(core.FCMFuzzifier(5).fit(D["Xtr"]).centers[4]), 5)       # "all": 5 levels on a binary concept
        fz = core.FCMFuzzifier(5).fit(D["Xtr"][:n0])
        self.assertEqual(len(fz.centers[4]), 2)                                          # "original": |Lambda_j| = 2
        R = fz.transform(D["Xtr"])
        w, _ = triple_weights(R[:n0], D["ytr"][:n0], D["ids_tr"].patient_id.to_numpy())
        _, score, _ = core.fisa_gm(R, D["ytr"], fz.transform(D["Xte"]), weights=w)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            for g in core.MODALITIES:
                d = root / g / "fold_01"
                d.mkdir(parents=True)
                pd.DataFrame(np.c_[D["Xtr"], D["ytr"]], columns=[*map(str, range(5)), TARGET]).to_csv(d / "train_data.csv", index=False)
                pd.DataFrame(np.c_[D["Xte"], D["yte"]], columns=[*map(str, range(5)), TARGET]).to_csv(d / "test_data.csv", index=False)
                D["ids_tr"].to_csv(d / "train_ids.csv", index=False)
                pd.DataFrame({"patient_id": [f"t{i}" for i in range(len(D["yte"]))],
                              "image_id": [f"j{i}" for i in range(len(D["yte"]))]}).to_csv(d / "test_ids.csv", index=False)
            from sklearn.metrics import roc_auc_score
            df = core.run(str(root), str(Path(tmp) / "o"), ["fold_01"], ["EPS"], {"fcm_fit": "original", "eps_grid": [1e-9]})
            self.assertAlmostEqual(df.auc_roc.iloc[0], roc_auc_score(D["yte"], score), places=12)

    def test_binary_aware_fuzzifier_keeps_two_levels(self):
        from kfold.weighted import BinaryAwareFuzzifier
        D = self.fold()
        X = D["Xtr"].copy()
        X[D["n_orig_train"]:, 4] = np.linspace(0.1, 0.9, len(X) - D["n_orig_train"])   # SMOTE-like interpolation
        R = BinaryAwareFuzzifier(5, n_orig=D["n_orig_train"]).fit(X).transform(X)
        self.assertEqual(sorted(np.unique(R[:, 4])), [0, 1])
        self.assertGreater(len(np.unique(R[:, 0])), 2)


if __name__ == "__main__":
    unittest.main()
