"""Five image-table fusion strategies and unimodal branches of Chapter 3, fold-local.

GENERATED VERBATIM (ast source segments) from
  reference/preprocess_source/Preprocess_kfold_feature_selection.py   sha256 2374bf5ee6790da8cdbba7b142695f3ab179026457bf492cf043262493b40f86
  (Source_code/main/diabetic_retinopathy, working copy of 02/11/2026: fusion_tensor = Algorithm 3.5)
and, for the optional legacy tensor only,
  reference/preprocess_source/Preprocess_kfold_feature_selection_HEAD_43a784c.py   sha256 0699e16a302d37ff5afcd5f9b7d97e2f6f3a0755ed9cf268c7e75f38ffcc0958
  (prepare_fusion_tensor_fold renamed prepare_fusion_tensor_cross_svd_fold: 3 x rank cross-SVD columns,
  the 30-column fusion_tensor of FKGMM_BRSET3_nested_20261102).
Do not edit by hand: tests/test_ch3.py checks every function against the reference file.
Package-specific code (inputs with missing values, fold driver) lives in chuong3.py.
"""
import numpy as np
import pandas as pd
from functools import partial
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler

LABEL_COLUMN = "diabetic_retinopathy"

IMAGE_FEATURE_COLUMNS = [
    "Contrast Feature",
    "Dissimilarity Feature",
    "Homogeneity Feature",
    "Energy Feature",
    "Correlation Feature",
    "ASM Feature",
    "Mean Feature",
    "Variance Feature",
    "Standard Deviation Feature",
    "RMS Feature",
]

TABLE_FEATURE_COLUMNS = [
    "patient_age",
    "patient_sex",
    "diabetes_time_y",
    "insuline",
    "diabetes",
    "exam_eye",
    "optic_disc",
    "vessels",
    "macula",
    "focus",
    "Illuminaton",
    "image_field",
    "quality",
]

TABLE_CLUSTER_BY_COLUMN = {
    "patient_age": 3,
    "patient_sex": 2,
    "diabetes_time_y": 4,
    "insuline": 2,
    "diabetes": 2,
    "exam_eye": 2,
    "optic_disc": 2,
    "vessels": 2,
    "macula": 2,
    "focus": 2,
    "Illuminaton": 2,
    "image_field": 2,
    "quality": 2,
}

TENSOR_EPSILON = 1e-8


def clean_numeric_split(train_df, test_df):
    medians = train_df.median(numeric_only=True).fillna(0)
    return train_df.fillna(medians), test_df.fillna(medians)


def scale_split(train_df, test_df, scaler):
    train_clean, test_clean = clean_numeric_split(train_df, test_df)
    train_scaled = scaler.fit_transform(train_clean)
    test_scaled = scaler.transform(test_clean)
    return (
        pd.DataFrame(train_scaled, columns=train_df.columns),
        pd.DataFrame(test_scaled, columns=test_df.columns),
    )


def selector_to_feature_rows(selector, source_columns, selected_columns, branch, output_start):
    scores = getattr(selector, "scores_", None)
    pvalues = getattr(selector, "pvalues_", None)
    selected_positions = {
        column: output_start + selected_columns.index(column)
        for column in selected_columns
    }

    rows = []
    for idx, column in enumerate(source_columns):
        rows.append(
            {
                "branch": branch,
                "selected": column in selected_positions,
                "source_column": str(column),
                "selected_output_column": (
                    selected_positions[column] if column in selected_positions else ""
                ),
                "score": None if scores is None else float(scores[idx]),
                "p_value": None if pvalues is None else float(pvalues[idx]),
            }
        )
    return rows


def select_k_best(train_df, test_df, labels, k, score_func, branch, output_start):
    if train_df.empty:
        raise ValueError(f"No feature columns available for branch '{branch}'.")
    actual_k = min(k, train_df.shape[1])
    if actual_k < 1:
        raise ValueError(f"k for branch '{branch}' must be at least 1.")

    selector = SelectKBest(score_func=score_func, k=actual_k)
    train_selected = selector.fit_transform(train_df, labels)
    test_selected = selector.transform(test_df)
    selected_columns = list(train_df.columns[selector.get_support()])
    feature_rows = selector_to_feature_rows(
        selector=selector,
        source_columns=list(train_df.columns),
        selected_columns=selected_columns,
        branch=branch,
        output_start=output_start,
    )
    return (
        pd.DataFrame(train_selected),
        pd.DataFrame(test_selected),
        selected_columns,
        feature_rows,
    )


def build_fis_frame(selected_features, labels):
    out = pd.DataFrame(selected_features).reset_index(drop=True)
    out.columns = [str(i) for i in range(out.shape[1])]
    out[LABEL_COLUMN] = pd.Series(labels).reset_index(drop=True)
    return out


def drop_train_constant_columns(train_selected, test_selected, feature_rows):
    train_df = pd.DataFrame(train_selected).reset_index(drop=True)
    test_df = pd.DataFrame(test_selected).reset_index(drop=True)
    keep_positions = [
        idx
        for idx in range(train_df.shape[1])
        if train_df.iloc[:, idx].nunique(dropna=False) > 1
    ]
    if len(keep_positions) == train_df.shape[1]:
        return train_df, test_df, feature_rows, []
    if not keep_positions:
        raise ValueError("All selected columns are constant on the train fold.")

    position_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_positions)}
    dropped_positions = [
        idx
        for idx in range(train_df.shape[1])
        if idx not in position_map
    ]
    updated_rows = []
    for row in feature_rows:
        updated = dict(row)
        selected_output = updated.get("selected_output_column", "")
        try:
            selected_output = int(float(selected_output))
        except (TypeError, ValueError):
            selected_output = None
        updated["dropped_constant_train"] = False
        if selected_output is not None:
            if selected_output in position_map:
                updated["selected_output_column"] = position_map[selected_output]
            else:
                updated["selected"] = False
                updated["selected_output_column"] = ""
                updated["dropped_constant_train"] = True
        updated_rows.append(updated)

    return (
        train_df.iloc[:, keep_positions].reset_index(drop=True),
        test_df.iloc[:, keep_positions].reset_index(drop=True),
        updated_rows,
        dropped_positions,
    )


def apply_train_smote(train_df, seed, max_k_neighbors):
    y = train_df[LABEL_COLUMN]
    x = train_df.drop(columns=[LABEL_COLUMN])
    counts = y.value_counts()
    min_count = int(counts.min())
    if min_count < 2:
        return train_df, "skipped_minority_class_too_small"

    k_neighbors = min(max_k_neighbors, min_count - 1)
    try:
        from imblearn.over_sampling import BorderlineSMOTE
    except ImportError:
        print("[WARN] imblearn is not installed; SMOTE skipped for this train fold.")
        return train_df, "skipped_imblearn_missing"

    sampler = BorderlineSMOTE(
        random_state=seed,
        sampling_strategy="auto",
        k_neighbors=k_neighbors,
    )
    x_resampled, y_resampled = sampler.fit_resample(x, y)
    resampled = pd.DataFrame(x_resampled, columns=x.columns)
    resampled[LABEL_COLUMN] = pd.Series(y_resampled).reset_index(drop=True)
    return resampled, f"borderline_smote_k_{k_neighbors}"


def prepare_image_fold(train_x, test_x, train_y, args, fold_seed):
    train_scaled, test_scaled = clean_numeric_split(train_x, test_x)
    train_selected, test_selected, selected_columns, rows = select_k_best(
        train_scaled,
        test_scaled,
        train_y,
        args.k_img,
        f_classif,
        branch="image",
        output_start=0,
    )
    cluster = [5] * len(selected_columns) + [2]
    return train_selected, test_selected, cluster, rows


def prepare_table_fold(train_x, test_x, train_y, args, fold_seed):
    train_scaled, test_scaled = scale_split(train_x, test_x, MinMaxScaler())
    score_func = partial(mutual_info_classif, random_state=fold_seed)
    train_selected, test_selected, selected_columns, rows = select_k_best(
        train_scaled,
        test_scaled,
        train_y,
        args.k_table,
        score_func,
        branch="table",
        output_start=0,
    )
    cluster = [TABLE_CLUSTER_BY_COLUMN.get(str(column), 5) for column in selected_columns] + [2]
    return train_selected, test_selected, cluster, rows


def fallback_fusion_split(columns):
    midpoint = len(columns) // 2
    return list(columns[:midpoint]), list(columns[midpoint:])


def split_fusion_columns(columns):
    table_columns = [col for col in TABLE_FEATURE_COLUMNS if col in columns]
    image_columns = [col for col in IMAGE_FEATURE_COLUMNS if col in columns]
    if not table_columns or not image_columns:
        table_columns, image_columns = fallback_fusion_split(list(columns))
    return table_columns, image_columns


def selected_index_feature_rows(source_columns, selected_indices, branch, output_start, scores=None):
    selected_indices = list(selected_indices)
    selected_positions = {
        idx: output_start + pos
        for pos, idx in enumerate(selected_indices)
    }
    rows = []
    for idx, column in enumerate(source_columns):
        rows.append(
            {
                "branch": branch,
                "selected": idx in selected_positions,
                "source_column": str(column),
                "selected_output_column": selected_positions[idx] if idx in selected_positions else "",
                "score": None if scores is None else float(scores[idx]),
                "p_value": None,
            }
        )
    return rows


def generated_component_feature_rows(method_name, feature_count, scores=None):
    return [
        {
            "branch": method_name,
            "selected": True,
            "source_column": f"{method_name}_component_{idx}",
            "selected_output_column": idx,
            "score": None if scores is None else float(scores[idx]),
            "p_value": None,
        }
        for idx in range(feature_count)
    ]


def prepare_fusion_fold(train_x, test_x, train_y, args, fold_seed):
    table_columns, image_columns = split_fusion_columns(train_x.columns)

    train_img, test_img = scale_split(
        train_x[image_columns],
        test_x[image_columns],
        StandardScaler(),
    )
    train_tab, test_tab = scale_split(
        train_x[table_columns],
        test_x[table_columns],
        StandardScaler(),
    )

    train_img_selected, test_img_selected, img_columns, img_rows = select_k_best(
        train_img,
        test_img,
        train_y,
        args.k_img,
        f_classif,
        branch="image",
        output_start=0,
    )

    train_tab_selected, test_tab_selected, tab_columns, tab_rows = select_k_best(
        train_tab,
        test_tab,
        train_y,
        args.k_tab,
        f_classif,
        branch="table",
        output_start=len(img_columns),
    )

    train_selected = pd.concat([train_img_selected, train_tab_selected], axis=1, ignore_index=True)
    test_selected = pd.concat([test_img_selected, test_tab_selected], axis=1, ignore_index=True)

    # The legacy FT-selection fusion scenario uses [5] for every selected feature.
    cluster = [5] * (len(img_columns) + len(tab_columns)) + [2]
    return train_selected, test_selected, cluster, img_rows + tab_rows


def compute_filter_scores(train_values, train_y, seed):
    from sklearn.ensemble import RandomForestClassifier

    x_scaled = MinMaxScaler().fit_transform(train_values)
    mi_scores = mutual_info_classif(x_scaled, train_y, random_state=seed)
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(x_scaled, train_y)
    return (mi_scores + rf.feature_importances_) / 2


def absolute_correlation(left, right):
    corr = np.corrcoef(left, right)[0, 1]
    return abs(corr) if np.isfinite(corr) else 0.0


def select_filter_multimodal_indices(img_values, tab_values, img_scores, tab_scores, k_img, k_tab, threshold):
    """Greedy multimodal filter selection with intra- and inter-modal corr checks."""
    if threshold <= 0 or threshold > 1:
        raise ValueError("--filter-corr must be in the interval (0, 1].")

    selected_img = []
    selected_tab = []
    selected_vectors = []
    candidates = []
    candidates.extend(("image", int(idx), float(img_scores[idx])) for idx in np.argsort(img_scores)[::-1])
    candidates.extend(("table", int(idx), float(tab_scores[idx])) for idx in np.argsort(tab_scores)[::-1])
    candidates.sort(key=lambda item: item[2], reverse=True)

    for modality, idx, _score in candidates:
        if modality == "image":
            if len(selected_img) >= k_img:
                continue
            vector = img_values[:, idx]
        else:
            if len(selected_tab) >= k_tab:
                continue
            vector = tab_values[:, idx]

        if any(absolute_correlation(vector, selected_vector) > threshold for selected_vector in selected_vectors):
            continue
        if modality == "image":
            selected_img.append(idx)
        else:
            selected_tab.append(idx)
        selected_vectors.append(vector)
        if len(selected_img) >= k_img and len(selected_tab) >= k_tab:
            break

    return selected_img, selected_tab


def prepare_fusion_filter_fold(train_x, test_x, train_y, args, fold_seed):
    table_columns, image_columns = split_fusion_columns(train_x.columns)
    train_img, test_img = scale_split(train_x[image_columns], test_x[image_columns], StandardScaler())
    train_tab, test_tab = scale_split(train_x[table_columns], test_x[table_columns], StandardScaler())

    img_values = train_img.to_numpy()
    tab_values = train_tab.to_numpy()
    img_scores = compute_filter_scores(img_values, train_y, fold_seed)
    tab_scores = compute_filter_scores(tab_values, train_y, fold_seed + 1)
    selected_img, selected_tab = select_filter_multimodal_indices(
        img_values,
        tab_values,
        img_scores,
        tab_scores,
        args.k_img,
        args.k_tab,
        args.filter_corr,
    )

    train_selected = pd.concat(
        [train_img.iloc[:, selected_img], train_tab.iloc[:, selected_tab]],
        axis=1,
        ignore_index=True,
    )
    test_selected = pd.concat(
        [test_img.iloc[:, selected_img], test_tab.iloc[:, selected_tab]],
        axis=1,
        ignore_index=True,
    )
    rows = selected_index_feature_rows(image_columns, selected_img, "image_filter", 0, img_scores)
    rows += selected_index_feature_rows(table_columns, selected_tab, "table_filter", len(selected_img), tab_scores)
    cluster = [5] * len(selected_img) + [
        TABLE_CLUSTER_BY_COLUMN.get(str(table_columns[idx]), 5)
        for idx in selected_tab
    ] + [2]
    return train_selected, test_selected, cluster, rows


def normalize_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True) + 1e-8
    return values / norms


def fit_cross_modal_svd(train_img_values, train_tab_values, requested_rank):
    """Fit W_img and W_tab from the train-only cross relation F_img^T F_tab.

    This solves the orthogonal cross-covariance objective
    max trace(W_img.T @ C @ W_tab), C = F_img.T @ F_tab / (n - 1),
    and avoids taking an SVD over flattened Kronecker features whose component
    matrices cannot be multiplied back with the original modalities.
    """
    max_rank = min(train_img_values.shape[1], train_tab_values.shape[1])
    rank = min(int(requested_rank), max_rank)
    if rank < 1:
        raise ValueError("Cross-modal SVD rank must be at least 1.")

    cross_cov = train_img_values.T @ train_tab_values / max(1, train_img_values.shape[0] - 1)
    u, singular_values, vt = np.linalg.svd(cross_cov, full_matrices=False)
    return u[:, :rank], vt[:rank, :].T, singular_values[:rank]


def apply_cross_modal_projection(img_values, tab_values, w_img, w_tab, singular_values):
    weights = np.sqrt(np.maximum(singular_values, 1e-12))
    img_latent = normalize_rows((img_values @ w_img) * weights)
    tab_latent = normalize_rows((tab_values @ w_tab) * weights)
    return img_latent, tab_latent


def prepare_fusion_hadamard_fold(train_x, test_x, train_y, args, fold_seed):
    """Hadamard fusion with train-only cross-SVD projections."""
    table_columns, image_columns = split_fusion_columns(train_x.columns)
    train_img, test_img = scale_split(train_x[image_columns], test_x[image_columns], StandardScaler())
    train_tab, test_tab = scale_split(train_x[table_columns], test_x[table_columns], StandardScaler())

    common_dim = int(args.hadamard_dim)
    if common_dim < 1:
        raise ValueError("--hadamard-dim must be at least 1.")

    w_img, w_tab, singular_values = fit_cross_modal_svd(
        train_img.to_numpy(),
        train_tab.to_numpy(),
        common_dim,
    )
    train_img_proj, train_tab_proj = apply_cross_modal_projection(
        train_img.to_numpy(),
        train_tab.to_numpy(),
        w_img,
        w_tab,
        singular_values,
    )
    test_img_proj, test_tab_proj = apply_cross_modal_projection(
        test_img.to_numpy(),
        test_tab.to_numpy(),
        w_img,
        w_tab,
        singular_values,
    )
    train_selected = train_img_proj * train_tab_proj
    test_selected = test_img_proj * test_tab_proj
    feature_count = train_selected.shape[1]
    cluster = [5] * feature_count + [2]
    return (
        pd.DataFrame(train_selected),
        pd.DataFrame(test_selected),
        cluster,
        generated_component_feature_rows("hadamard", feature_count, singular_values),
    )


def fit_tensor_fusion(train_img_values, train_tab_values, requested_rank, epsilon=TENSOR_EPSILON):
    """Fit the tensor-product fusion operator Phi_ten (Algorithm 3.5) on train rows only.

    M = (1/N) F~img^T F~tab is the sample mean of the tensor products x~_n (x) y~_n.
    Its rank-rho truncated SVD gives P_img = U_rho Sigma_rho^(1/2) and
    P_tab = V_rho Sigma_rho^(1/2), so (x~ P_img) . (y~ P_tab) is the best rank-rho
    reconstruction of the bilinear interaction x~^T M y~.
    """
    train_img_values = np.asarray(train_img_values, dtype=float)
    train_tab_values = np.asarray(train_tab_values, dtype=float)
    if int(requested_rank) < 1:
        raise ValueError("--tensor-rank must be at least 1.")

    # z-score with the population standard deviation, as in Example 3.2.
    mean_img = train_img_values.mean(axis=0)
    std_img = train_img_values.std(axis=0)
    mean_tab = train_tab_values.mean(axis=0)
    std_tab = train_tab_values.std(axis=0)
    norm_img = (train_img_values - mean_img) / (std_img + epsilon)
    norm_tab = (train_tab_values - mean_tab) / (std_tab + epsilon)

    interaction = norm_img.T @ norm_tab / norm_img.shape[0]
    u, singular_values, vt = np.linalg.svd(interaction, full_matrices=False)
    tolerance = singular_values.max(initial=0.0) * max(interaction.shape) * np.finfo(float).eps
    matrix_rank = int((singular_values > tolerance).sum())
    if matrix_rank < 1:
        raise ValueError("Tensor fusion interaction matrix M is zero on the train fold.")
    rank = min(int(requested_rank), matrix_rank)

    u = u[:, :rank]
    v = vt[:rank, :].T
    singular_values = singular_values[:rank]
    # SVD signs are arbitrary; fix them so the first non-zero entry of each U
    # column is positive (the convention of Example 3.2) and FRBs are reproducible.
    for component in range(rank):
        column = u[:, component]
        first = np.flatnonzero(np.abs(column) > 1e-12)
        if first.size and column[first[0]] < 0:
            u[:, component] *= -1.0
            v[:, component] *= -1.0

    root_sigma = np.sqrt(singular_values)
    return {
        "mean_img": mean_img,
        "std_img": std_img,
        "mean_tab": mean_tab,
        "std_tab": std_tab,
        "epsilon": float(epsilon),
        "p_img": u * root_sigma,
        "p_tab": v * root_sigma,
        "singular_values": singular_values,
        "matrix_rank": matrix_rank,
    }


def apply_tensor_fusion(img_values, tab_values, operator):
    """Return [F~img P_img || F~tab P_tab] using the train-fold statistics."""
    norm_img = (np.asarray(img_values, dtype=float) - operator["mean_img"]) / (
        operator["std_img"] + operator["epsilon"]
    )
    norm_tab = (np.asarray(tab_values, dtype=float) - operator["mean_tab"]) / (
        operator["std_tab"] + operator["epsilon"]
    )
    return np.concatenate([norm_img @ operator["p_img"], norm_tab @ operator["p_tab"]], axis=1)


def prepare_fusion_tensor_fold(train_x, test_x, train_y, args, fold_seed):
    """Tensor-product fusion Phi_ten (Algorithm 3.5): 2*rho columns, rho <= rank(M)."""
    table_columns, image_columns = split_fusion_columns(train_x.columns)
    operator = fit_tensor_fusion(
        train_x[image_columns].to_numpy(),
        train_x[table_columns].to_numpy(),
        args.tensor_rank,
    )
    train_selected = apply_tensor_fusion(
        train_x[image_columns].to_numpy(), train_x[table_columns].to_numpy(), operator
    )
    test_selected = apply_tensor_fusion(
        test_x[image_columns].to_numpy(), test_x[table_columns].to_numpy(), operator
    )
    rank = len(operator["singular_values"])
    cluster = [5] * (2 * rank) + [2]
    rows = [
        {
            "branch": f"tensor_{branch}",
            "selected": True,
            "source_column": f"tensor_{branch}_component_{component}",
            "selected_output_column": offset + component,
            "score": float(operator["singular_values"][component]),
            "p_value": None,
        }
        for offset, branch in ((0, "img"), (rank, "tab"))
        for component in range(rank)
    ]
    return pd.DataFrame(train_selected), pd.DataFrame(test_selected), cluster, rows


def evaluate_wrapper_feature_set(values, target, seed, cv, estimators):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    label_counts = pd.Series(target).value_counts()
    actual_cv = min(int(cv), int(label_counts.min()))
    if actual_cv < 2:
        model = RandomForestClassifier(n_estimators=estimators, random_state=seed, n_jobs=-1)
        model.fit(values, target)
        return float(model.score(values, target))
    model = RandomForestClassifier(n_estimators=estimators, random_state=seed, n_jobs=-1)
    scores = cross_val_score(model, values, target, cv=actual_cv, scoring="accuracy")
    return float(scores.mean())


def best_wrapper_feature(values, selected_indices, target, remaining_indices, seed, cv, estimators, cache, cache_prefix):
    best_score = None
    best_feature = None
    for idx in sorted(remaining_indices):
        trial = selected_indices + [idx]
        cache_key = (cache_prefix, tuple(trial))
        if cache_key not in cache:
            cache[cache_key] = evaluate_wrapper_feature_set(values[:, trial], target, seed, cv, estimators)
        score = cache[cache_key]
        if best_score is None or score > best_score:
            best_score = score
            best_feature = idx
    return best_feature, best_score


def select_wrapper_indices(img_values, tab_values, target, args, seed):
    selected_img = []
    selected_tab = []
    img_indices = set(range(img_values.shape[1]))
    tab_indices = set(range(tab_values.shape[1]))
    score_cache = {}
    cv = max(2, int(args.wrapper_cv))
    estimators = max(1, int(args.wrapper_rf_estimators))

    for _ in range(min(args.wrapper_min_img, args.wrapper_max_img, img_values.shape[1])):
        best_feature, _ = best_wrapper_feature(
            img_values,
            selected_img,
            target,
            img_indices - set(selected_img),
            seed,
            cv,
            estimators,
            score_cache,
            "min_image",
        )
        if best_feature is not None:
            selected_img.append(best_feature)

    for _ in range(min(args.wrapper_min_tab, args.wrapper_max_tab, tab_values.shape[1])):
        best_feature, _ = best_wrapper_feature(
            tab_values,
            selected_tab,
            target,
            tab_indices - set(selected_tab),
            seed,
            cv,
            estimators,
            score_cache,
            "min_table",
        )
        if best_feature is not None:
            selected_tab.append(best_feature)

    fused = np.concatenate([img_values[:, selected_img], tab_values[:, selected_tab]], axis=1)
    best_score = evaluate_wrapper_feature_set(fused, target, seed, cv, estimators)

    while len(selected_img) < min(args.wrapper_max_img, img_values.shape[1]) or len(selected_tab) < min(args.wrapper_max_tab, tab_values.shape[1]):
        best_new_score = None
        best_new_feature = None
        best_modality = None

        if len(selected_img) < min(args.wrapper_max_img, img_values.shape[1]):
            for idx in sorted(img_indices - set(selected_img)):
                fused_trial = np.concatenate(
                    [img_values[:, selected_img + [idx]], tab_values[:, selected_tab]],
                    axis=1,
                )
                cache_key = ("image", tuple(selected_img + [idx]), tuple(selected_tab))
                if cache_key not in score_cache:
                    score_cache[cache_key] = evaluate_wrapper_feature_set(
                        fused_trial,
                        target,
                        seed,
                        cv,
                        estimators,
                    )
                score = score_cache[cache_key]
                if best_new_score is None or score > best_new_score:
                    best_new_score = score
                    best_new_feature = idx
                    best_modality = "image"

        if len(selected_tab) < min(args.wrapper_max_tab, tab_values.shape[1]):
            for idx in sorted(tab_indices - set(selected_tab)):
                fused_trial = np.concatenate(
                    [img_values[:, selected_img], tab_values[:, selected_tab + [idx]]],
                    axis=1,
                )
                cache_key = ("table", tuple(selected_img), tuple(selected_tab + [idx]))
                if cache_key not in score_cache:
                    score_cache[cache_key] = evaluate_wrapper_feature_set(
                        fused_trial,
                        target,
                        seed,
                        cv,
                        estimators,
                    )
                score = score_cache[cache_key]
                if best_new_score is None or score > best_new_score:
                    best_new_score = score
                    best_new_feature = idx
                    best_modality = "table"

        if best_new_feature is None or best_new_score <= best_score:
            break
        best_score = best_new_score
        if best_modality == "image":
            selected_img.append(best_new_feature)
        else:
            selected_tab.append(best_new_feature)

    return selected_img, selected_tab


def prepare_fusion_wrapper_fold(train_x, test_x, train_y, args, fold_seed):
    table_columns, image_columns = split_fusion_columns(train_x.columns)
    train_img, test_img = scale_split(train_x[image_columns], test_x[image_columns], StandardScaler())
    train_tab, test_tab = scale_split(train_x[table_columns], test_x[table_columns], StandardScaler())
    img_values = train_img.to_numpy()
    tab_values = train_tab.to_numpy()
    selected_img, selected_tab = select_wrapper_indices(img_values, tab_values, train_y.to_numpy(), args, fold_seed)

    train_selected = pd.concat(
        [train_img.iloc[:, selected_img], train_tab.iloc[:, selected_tab]],
        axis=1,
        ignore_index=True,
    )
    test_selected = pd.concat(
        [test_img.iloc[:, selected_img], test_tab.iloc[:, selected_tab]],
        axis=1,
        ignore_index=True,
    )
    rows = selected_index_feature_rows(image_columns, selected_img, "image_wrapper", 0)
    rows += selected_index_feature_rows(table_columns, selected_tab, "table_wrapper", len(selected_img))
    cluster = [5] * len(selected_img) + [
        TABLE_CLUSTER_BY_COLUMN.get(str(table_columns[idx]), 5)
        for idx in selected_tab
    ] + [2]
    return train_selected, test_selected, cluster, rows


def prepare_fusion_tensor_cross_svd_fold(train_x, test_x, train_y, args, fold_seed):
    """Tensor-style fusion using cross-SVD latent factors, not raw Kronecker SVD."""
    table_columns, image_columns = split_fusion_columns(train_x.columns)
    train_img, test_img = scale_split(train_x[image_columns], test_x[image_columns], StandardScaler())
    train_tab, test_tab = scale_split(train_x[table_columns], test_x[table_columns], StandardScaler())

    if int(args.tensor_rank) < 1:
        raise ValueError("--tensor-rank must be at least 1.")
    w_img, w_tab, singular_values = fit_cross_modal_svd(
        train_img.to_numpy(),
        train_tab.to_numpy(),
        args.tensor_rank,
    )
    train_img_proj, train_tab_proj = apply_cross_modal_projection(
        train_img.to_numpy(),
        train_tab.to_numpy(),
        w_img,
        w_tab,
        singular_values,
    )
    test_img_proj, test_tab_proj = apply_cross_modal_projection(
        test_img.to_numpy(),
        test_tab.to_numpy(),
        w_img,
        w_tab,
        singular_values,
    )
    train_selected = np.concatenate(
        [train_img_proj, train_tab_proj, train_img_proj * train_tab_proj],
        axis=1,
    )
    test_selected = np.concatenate(
        [test_img_proj, test_tab_proj, test_img_proj * test_tab_proj],
        axis=1,
    )
    feature_count = train_selected.shape[1]
    cluster = [5] * feature_count + [2]
    rows = generated_component_feature_rows(
        "tensor_cross_svd",
        feature_count,
        np.tile(singular_values, 3),
    )
    return pd.DataFrame(train_selected), pd.DataFrame(test_selected), cluster, rows
