"""Chapter 3 (FKG-UM / FKG-MM, five fusion strategies, M1-M6, EPS) on the package's 16,266 images.

Two stages:
  prepare  build the 7 data groups (table, image, fusion, fusion_filter, fusion_hadamard, fusion_tensor,
           fusion_wrapper) for the package's 5 patient-disjoint outer folds (data/fold_assignments.csv,
           the same folds as Chapter 4). Every imputation, scaling, selection, projection and
           BorderlineSMOTE is fitted on the fold's training images only (fusion_strategies.py, verbatim
           from Preprocess_kfold_feature_selection.py). Layout = KFold_feature_selection_* folders.
  run      Chapter 3 scenarios with FISA-GM (kfold/core.py, unchanged from FKGMM_BRSET3_nested_20261102)
           and the report kfold/report.py.

    python chuong3.py all --out outputs/chuong3 --resume
    python chuong3.py prepare --out outputs/chuong3 --modalities fusion fusion_tensor --folds 1
    python chuong3.py run --out outputs/chuong3 --scenarios M1 M5 --resume
    python chuong3.py report --out outputs/chuong3
    python chuong3.py all --smoke --out outputs/chuong3_smoke     # synthetic data, never a BRSET result

Output: <out>/data/<group>/fold_XX/{train_data,test_data,train_ids,test_ids,selected_features}.csv +
metadata.json; <out>/ket_qua/{per_fold.csv, audit_data.csv, ket_qua.md, run_manifest.json}.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import fusion_strategies as FS
from build import IMAGE_COLUMNS, TABLE_COLUMNS, TARGET

ROOT = Path(__file__).resolve().parent
GROUPS = ["table", "image", "fusion", "fusion_filter", "fusion_hadamard", "fusion_tensor", "fusion_wrapper"]
# Nhóm bổ sung: FKG-UM (Bảng) dùng đúng cách chọn thuộc tính bảng của FKG-MM (Φsel): chuẩn hoá z + ANOVA trên
# fold huấn luyện, k = k_table_um. Nhóm "table" (MinMax + MI, 13 cột) giữ nguyên cho LR/RF/XGBoost/MLP và Late Fusion.
EXTRA_GROUPS = ["table_sel"]
ALL_GROUPS = GROUPS + EXTRA_GROUPS
def prepare_table_sel_fold(train_x, test_x, train_y, args, fold_seed):
    """FKG-UM (Bảng) với đúng các thuộc tính bảng mà FKG-MM (Φsel) chọn: giống hệt nhánh bảng của
    FS.prepare_fusion_fold (StandardScaler trên cột lâm sàng, điền thiếu bằng trung vị fold huấn luyện, SelectKBest
    f_classif), chỉ khác k = args.k_table_um. Với k_table_um = k_tab, tập thuộc tính trùng đúng với FKG-MM trong
    cùng fold; với k_table_um < k_tab là k thuộc tính xếp hạng ANOVA cao nhất trong tập đó."""
    from sklearn.feature_selection import f_classif
    from sklearn.preprocessing import StandardScaler
    table_columns, _ = FS.split_fusion_columns(train_x.columns)
    train_tab, test_tab = FS.scale_split(train_x[table_columns], test_x[table_columns], StandardScaler())
    k = int(getattr(args, "k_table_um", args.k_tab))
    train_sel, test_sel, columns, rows = FS.select_k_best(train_tab, test_tab, train_y, k, f_classif,
                                                          branch="table", output_start=0)
    cluster = [5] * len(columns) + [2]          # như nhánh fusion (Φsel): mọi thuộc tính dùng cùng số mức
    return train_sel, test_sel, cluster, rows


PREPARE = {"image": FS.prepare_image_fold, "table": FS.prepare_table_fold, "fusion": FS.prepare_fusion_fold,
           "fusion_filter": FS.prepare_fusion_filter_fold, "fusion_hadamard": FS.prepare_fusion_hadamard_fold,
           "fusion_tensor": FS.prepare_fusion_tensor_fold, "fusion_wrapper": FS.prepare_fusion_wrapper_fold}
PREPARE["table_sel"] = prepare_table_sel_fold
assert FS.IMAGE_FEATURE_COLUMNS == IMAGE_COLUMNS and FS.TABLE_FEATURE_COLUMNS == TABLE_COLUMNS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda v: v.tolist() if hasattr(v, "tolist")
                              else (v.item() if hasattr(v, "item") else str(v))) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_columns(group):
    if group == "image":
        return list(IMAGE_COLUMNS)
    if group == "table_sel":
        return [*TABLE_COLUMNS, *IMAGE_COLUMNS]      # split_fusion_columns lấy đúng các cột lâm sàng như Φsel
    if group == "table":
        return list(TABLE_COLUMNS)
    return [*TABLE_COLUMNS, *IMAGE_COLUMNS]


def prepare_group_fold(group, train_x, test_x, train_y, test_y, args, fold_seed):
    """One (group, fold) exactly as Preprocess_kfold_feature_selection.prepare_modality does it, with one
    package-specific step: the clinical columns of this package keep their missing values (the source
    pipeline read a globally imputed CSV), so Algorithm 3.5 receives train-median-filled inputs, the same
    clean_numeric_split every other branch applies first. Returns (train_fold, test_fold, info)."""
    if group == "fusion_tensor":
        train_x, test_x = FS.clean_numeric_split(train_x, test_x)
    train_sel, test_sel, cluster, rows = PREPARE[group](train_x, test_x, train_y, args, fold_seed)
    train_sel, test_sel, rows, dropped = FS.drop_train_constant_columns(train_sel, test_sel, rows)
    train_fold, test_fold = FS.build_fis_frame(train_sel, train_y), FS.build_fis_frame(test_sel, test_y)
    if dropped:
        cluster = [v for i, v in enumerate(cluster[:-1]) if i not in set(dropped)] + [cluster[-1]]
    smote = "disabled"
    if not args.skip_smote:
        train_fold, smote = FS.apply_train_smote(train_fold, seed=fold_seed, max_k_neighbors=args.smote_k_neighbors)
    return train_fold, test_fold, {"cluster": cluster, "feature_rows": rows, "dropped_constant_columns": dropped,
                                   "smote": smote}


def synthetic_data(seed):
    """Small fixture for --smoke only (same generator as run.py of the FKG-E package); never BRSET rows."""
    from build import IDS
    from portable_folds import create_assignments
    rng = np.random.RandomState(seed)
    patients = np.repeat(np.arange(100), 2)
    y = (patients % 4 == 0).astype(int)
    data = pd.DataFrame({"image_id": [f"smoke_{i:04}" for i in range(len(y))],
                         "patient_id": [f"smoke_p{p:03}" for p in patients], TARGET: y})
    for column in [*TABLE_COLUMNS, *IMAGE_COLUMNS]:
        data[column] = rng.normal(size=len(y)) + 0.4 * y
    data.loc[::9, "patient_age"] = np.nan
    assignments, _ = create_assignments(data[IDS], TARGET, folds=5, seed=seed)
    return data, assignments


def load_inputs(smoke, seed):
    if smoke:
        data, assignments = synthetic_data(seed)
        return data, assignments, "SYNTHETIC_SMOKE_ONLY"
    from build import verify_release
    verify_release(ROOT / "data")
    data = pd.read_csv(ROOT / "data/fusion_features.csv", dtype={"image_id": str, "patient_id": str},
                       float_precision="round_trip")
    assignments = pd.read_csv(ROOT / "data/fold_assignments.csv", dtype={"image_id": str, "patient_id": str})
    if not data[["image_id", "patient_id", TARGET]].equals(assignments[["image_id", "patient_id", TARGET]]):
        raise ValueError("fusion_features.csv and fold_assignments.csv rows differ")
    return data, assignments, "BRSET_FULL_16266_IMAGES_8524_PATIENTS"


def prepare(out, groups, folds, cfg, smoke, resume):
    data, assignments, scope = load_inputs(smoke, cfg["seed"])
    args = SimpleNamespace(**cfg)
    identity = {"config": {k: v for k, v in cfg.items() if k not in ("aggregator", "rule_base", "tie", "fcm_fit",
                                                                     "fkg_um_table")},
                "scope": scope, "fusion_strategies_sha256": sha(ROOT / "fusion_strategies.py"),
                "chuong3_sha256": sha(ROOT / "chuong3.py"),
                "data_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
                "split_sha256": hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest()}
    root = out / "data"
    manifest_path = root / "prepare_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8"))["identity"] != identity:
            raise SystemExit(f"{root} was prepared with other code/config/data: use a new --out")
        if not resume:
            raise SystemExit(f"{root} exists: add --resume to continue, or use a new --out")
    write_json(manifest_path, {"identity": identity, "python": sys.version, "platform": platform.platform(),
                               "packages": {p: importlib.metadata.version(p) for p in
                                            ("numpy", "pandas", "scikit-learn", "imbalanced-learn")}})
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    for group in groups:
        records = []
        for fold in folds:
            fold_dir = root / group / f"fold_{fold:02d}"
            if (fold_dir / "metadata.json").exists():
                records.append(json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8")))
                continue
            started = time.perf_counter()
            fold_seed = cfg["seed"] + fold
            train = data[assignments.fold.to_numpy() != fold].reset_index(drop=True)
            test = data[assignments.fold.to_numpy() == fold].reset_index(drop=True)
            overlap = set(train.patient_id) & set(test.patient_id)
            if overlap:
                raise AssertionError(f"fold {fold}: {len(overlap)} patients in both train and test")
            cols = source_columns(group)
            train_fold, test_fold, info = prepare_group_fold(group, train[cols], test[cols], train[TARGET],
                                                             test[TARGET], args, fold_seed)
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_fold.to_csv(fold_dir / "train_data.csv", index=False)
            test_fold.to_csv(fold_dir / "test_data.csv", index=False)
            train[["patient_id", "image_id"]].to_csv(fold_dir / "train_ids.csv", index=False)
            test[["patient_id", "image_id"]].to_csv(fold_dir / "test_ids.csv", index=False)
            pd.DataFrame(info.pop("feature_rows")).to_csv(fold_dir / "selected_features.csv", index=False)
            record = {"created_at": datetime.now().isoformat(timespec="seconds"), "modality": group, "fold": fold,
                      "folds": len(folds), "seed": cfg["seed"], "fold_seed": fold_seed,
                      "splitter": "package fold_assignments.csv (StratifiedGroupKFold by patient_id)",
                      "split_group_column": "patient_id", "feature_selection_fit": "train_fold_only",
                      "missing_values": "train-fold medians (clean_numeric_split)", **info,
                      "train_source_rows": len(train), "test_source_rows": len(test),
                      "train_rows": len(train_fold), "test_rows": len(test_fold),
                      "smote_rows": len(train_fold) - len(train),
                      "train_patient_count": train.patient_id.nunique(), "test_patient_count": test.patient_id.nunique(),
                      "patient_overlap_count": 0, "feature_count": train_fold.shape[1] - 1,
                      "preprocess_time_seconds": time.perf_counter() - started}
            write_json(fold_dir / "metadata.json", record)
            records.append(record)
            print(f"[prepare] {group} fold {fold}: {record['feature_count']} features, {len(train_fold)} train rows "
                  f"({record['smote_rows']} SMOTE), {record['preprocess_time_seconds']:.1f}s", flush=True)
        write_json(root / group / "manifest.json", {"modality": group, "fold_records": records, "identity": identity})
    write_json(root / "manifest_all.json", {"modalities": [g for g in GROUPS if (root / g / "manifest.json").exists()],
                                            "identity": identity})
    return scope


def completed_folds(per_fold, scenarios):
    if not per_fold.exists():
        return set()
    df = pd.read_csv(per_fold)
    return {f for f, g in df.groupby("fold") if set(scenarios) <= set(g.scenario)}


def run_scenarios(out, folds, scenarios, cfg, resume, data=None):
    from kfold.core import audit, run
    data, results = data or out / "data", out / "ket_qua"
    need = GROUPS + ([cfg["fkg_um_table"]] if cfg.get("fkg_um_table", "table") != "table" else [])
    missing = [g for g in need for f in folds if not (data / g / f"fold_{f:02d}" / "metadata.json").exists()]
    if missing:
        raise SystemExit(f"Prepare the data groups first; missing: {sorted(set(missing))}")
    results.mkdir(parents=True, exist_ok=True)
    au = audit(str(data))
    au.to_csv(results / "audit_data.csv", index=False)
    if int(au.patient_overlap.sum()):
        raise SystemExit("Patient overlap between train and test")
    names = [f"fold_{f:02d}" for f in folds]
    if resume:
        done = completed_folds(results / "per_fold.csv", scenarios)
        names = [n for n in names if n not in done]
        if done:
            print(f"[run] already complete, skipped: {sorted(done)}", flush=True)
    elif (results / "per_fold.csv").exists():
        raise SystemExit(f"{results} has results: add --resume, or use a new --out")
    if names:
        run(str(data), str(results), names, scenarios, {"n_clusters": cfg["n_clusters"], "eps": cfg["eps"],
                                                        "seed": cfg["fkg_seed"], "aggregator": cfg["aggregator"],
                                                        "rule_base": cfg["rule_base"], "tie": cfg["tie"],
                                                        "fcm_fit": cfg["fcm_fit"],
                                                        "fkg_um_table": cfg.get("fkg_um_table", "table")})
    report(out, data)


def report(out, data=None):
    from kfold.report import build
    results = out / "ket_qua"
    text = build(str(results))
    data = data or out / "data"
    manifest = {"data_dir": str(data), "data": json.loads((data / "prepare_manifest.json").read_text(encoding="utf-8")),
                "code_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob("*.py"))
                                if not any(x in {"outputs", "__pycache__"} for x in p.relative_to(ROOT).parts)},
                "folds_in_results": sorted(pd.read_csv(results / "per_fold.csv").fold.unique().tolist()),
                "argv": [sys.executable, *sys.argv], "time": datetime.now().isoformat(timespec="seconds")}
    from kfold.plots import render_plots
    manifest["figures"] = [str(Path(p).relative_to(results)) for p in render_plots(str(results))]
    write_json(results / "chuong3_manifest.json", manifest)
    n = len(manifest["folds_in_results"])
    print(text if n == 5 else text + f"\n**CHƯA ĐỦ 5 FOLD ({n}/5) — không dùng để báo cáo.**\n", flush=True)


VARIANTS = {"GM": (False, False), "GM-w": (True, False), "GM+bin": (False, True), "GM-w+bin": (True, True)}
NAMES = {"table": "FKG-UM (Bảng)", "image": "FKG-UM (Ảnh)", "fusion": "FKG-MM Φsel (đề xuất)",
         "fusion_filter": "FKG-MM Filter", "fusion_hadamard": "FKG-MM Hadamard", "fusion_tensor": "FKG-MM Tensor",
         "fusion_wrapper": "FKG-MM Wrapper"}


def audit(out, folds, cfg, data=None):
    """Audit of FISA-GM (kfold/weighted.py): every group x {GM, GM-w, GM+bin, GM-w+bin}. Variant GM must
    reproduce the main run exactly. The fixes are pre-registered; no test metric chooses anything."""
    from kfold.core import load, metrics
    from kfold.report import _NB, _ms, _pair, _txt
    from kfold.weighted import fisa_gm_variant
    data, res = data or out / "data", out / "ket_qua_kiem_toan"
    res.mkdir(parents=True, exist_ok=True)
    path = res / "per_fold.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["group"], r["variant"], r["fold"]) for r in rows}
    for f in folds:
        fold = f"fold_{f:02d}"
        for g in GROUPS:
            D = load(str(data), g, fold)
            for v, (weighted, binary) in VARIANTS.items():
                if (g, v, fold) in done:
                    continue
                t0 = time.perf_counter()
                pred, score, info = fisa_gm_variant(D, weighted, binary, cfg["n_clusters"], cfg["eps"], cfg["fkg_seed"])
                rows.append({"group": g, "model": NAMES[g], "variant": v, "fold": fold,
                             **metrics(D["yte"], pred, score), **info, "seconds": time.perf_counter() - t0})
            pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[audit] {fold} xong", flush=True)
    df = pd.DataFrame(rows)
    main = pd.read_csv(out / "ket_qua/per_fold.csv")
    ref = main[(main.scenario == "M5") | ((main.scenario == "M1") & main.model.str.startswith("FKG-UM"))]
    ref = ref.assign(group=ref.model.map({"FKG-UM (Bảng)": "table", "FKG-UM (Ảnh)": "image", "FKG-MM (đề xuất)": "fusion",
                                          "FKG-MM (Filter)": "fusion_filter", "FKG-MM (Hadamard)": "fusion_hadamard",
                                          "FKG-MM (Tensor)": "fusion_tensor", "FKG-MM (Wrapper)": "fusion_wrapper"}))
    main_fisa = json.loads((out / "ket_qua/run_manifest.json").read_text(encoding="utf-8")).get("fisa", "")
    main_variant = "GM-w" if main_fisa.startswith("FISA-GM-w") else "GM"
    chk = df[df.variant == main_variant].merge(ref[["group", "fold", "auc_roc", "balacc"]], on=["group", "fold"],
                                               suffixes=("", "_main"))
    max_diff = float((chk[["auc_roc", "balacc"]].to_numpy() - chk[["auc_roc_main", "balacc_main"]].to_numpy()).__abs__().max())
    if max_diff > 1e-12:
        raise AssertionError(f"Variant {main_variant} does not reproduce the main run (max diff {max_diff})")
    au = pd.read_csv(out / "ket_qua/audit_data.csv") if (out / "ket_qua/audit_data.csv").exists() else None
    if au is None:
        raise SystemExit("Run the main scenarios on this --out first")
    _NB.update(n_tr=float(au.n_train_orig.mean()), n_te=float(au.n_test.mean()))
    q = lambda g, v: df[(df.group == g) & (df.variant == v)].set_index("fold")
    L = ["# Kiểm toán FISA-GM: FKG-MM so với FKG-UM và các toán tử fusion", "",
         f"{df.fold.nunique()} fold; biến thể {main_variant} tái lập đúng lần chạy chính (sai khác lớn nhất {max_diff:.1e}).",
         "GM-w: trọng số bộ ba w_t = max(AUC_t − 0,5; 0) ước lượng chéo theo bệnh nhân trên ảnh gốc của tập "
         "huấn luyện; +bin: biến nhị phân giữ 2 mức. Áp dụng như nhau cho mọi nhóm; tập kiểm tra không dùng để chọn.", "",
         "AP = average precision; PR-AUC = quy tắc hình thang; mốc ngẫu nhiên của cả hai = tỉ lệ dương.", "",
         "| Nhóm | Biến thể | AUC | AP | PR-AUC | BalAcc | Sens. |", "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for v in VARIANTS:
            L.append(f"| {NAMES[g] if v == 'GM' else ''} | {v} | " + " | ".join(
                _ms(q(g, v), m) for m in ("auc_roc", "auc_pr", "pr_auc_trap", "balacc", "sens")) + " |")
    L += ["", "## So sánh cặp (Δ = điểm %, p Nadeau–Bengio với kích thước fold thực tế)", ""]
    for v in VARIANTS:
        L.append(f"**{v}**")
        for a, b in (("fusion", "table"), ("fusion", "image"), ("fusion", "fusion_tensor"), ("fusion_tensor", "table")):
            L.append(f"- {NAMES[a]} − {NAMES[b]}: AUC {_txt(_pair(q(a, v), q(b, v)))}; "
                     f"BalAcc {_txt(_pair(q(a, v), q(b, v), 'balacc'))}")
        L.append("")
    L += ["## Hiệu quả của sửa đổi trên cùng nhóm (GM-w − GM)", ""]
    for g in GROUPS:
        L.append(f"- {NAMES[g]}: AUC {_txt(_pair(q(g, 'GM-w'), q(g, 'GM')))}")
    (res / "ket_qua_kiem_toan.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "run", "report", "all", "audit"])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/chuong3")
    parser.add_argument("--config", type=Path, default=ROOT / "config_ch3.json")
    parser.add_argument("--data", type=Path, default=None,
                        help="Prepared data of another --out (run/report/audit only), default <out>/data")
    parser.add_argument("--modalities", nargs="+", default=ALL_GROUPS, choices=ALL_GROUPS)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--scenarios", nargs="+", default=["M1", "M2", "M3", "M4", "M5", "M6", "EPS"])
    parser.add_argument("--rule-base", choices=["all", "dedup_conf"], default=None,
                        help="dedup_conf: luật trùng giữ 1, luật mâu thuẫn giữ nhãn có độ tin cậy cao hơn")
    parser.add_argument("--fcm-fit", choices=["all", "original"], default=None,
                        help="original: hàm thuộc FCM ước lượng trên ảnh gốc (trước SMOTE), như Mục 3.6.3 của luận án")
    parser.add_argument("--fkg-um-table", choices=["table", "table_sel"], default=None,
                        help="table_sel: FKG-UM (Bảng) dùng các thuộc tính bảng FKG-MM chọn (k = k_table_um); "
                             "table: 13 cột lâm sàng như bản cũ")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    out = a.out.resolve()
    data = a.data.resolve() if a.data else None
    if data and a.stage in ("prepare", "all"):
        parser.error("--data is for run/report/audit; prepare always writes <out>/data")
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    if a.smoke:
        cfg.update(cfg.get("smoke_overrides", {}))
    cfg.pop("smoke_overrides", None)
    if a.rule_base:
        cfg["rule_base"] = a.rule_base
    if a.fcm_fit:
        cfg["fcm_fit"] = a.fcm_fit
    if a.fkg_um_table:
        cfg["fkg_um_table"] = a.fkg_um_table
    if a.stage in ("prepare", "all"):
        prepare(out, a.modalities, a.folds, cfg, a.smoke, a.resume)
    if a.stage in ("run", "all"):
        run_scenarios(out, a.folds, a.scenarios, cfg, a.resume, data)
    if a.stage == "report":
        report(out, data)
    if a.stage == "audit":
        audit(out, a.folds, cfg, data)


if __name__ == "__main__":
    main()
