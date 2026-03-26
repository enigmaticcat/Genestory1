import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_PROCESSED_DIR, RESULTS_DIR,
    TRAIN_RATIO, RANDOM_SEED, NUM_CV_FOLDS,
)
from src.data_preprocessing import preprocess_scenario
from src.dataset import prepare_profile_datasets

from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("XGBoost chưa được cài. Chạy: pip install xgboost")
    sys.exit(1)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    print("[WARN] SHAP chưa được cài — SHAP analysis sẽ bị bỏ qua. Chạy: pip install shap")
    HAS_SHAP = False


# ============================================================
# Cấu hình XGBoost
# ============================================================
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    eval_metric="mlogloss",
    verbosity=0,
)


# ============================================================
# Feature Name Mapping
# ============================================================
# 17 per-marker features (in order from _extract_marker_features)
_MARKER_FEAT_NAMES = [
    "n_alleles",      # 0
    "h1",             # 1
    "h2",             # 2
    "h3",             # 3
    "sum_h",          # 4
    "mean_h",         # 5
    "std_h",          # 6
    "h_ratio",        # 7
    "h_range",        # 8
    "n_ol",           # 9
    "n_missing",      # 10
    "stutter_ratio",  # 11
    "snr_top2",       # 12
    "log1p_h1",       # 13
    "log1p_sum_h",    # 14
    "missing_marker", # 15
    "marker_index",   # 16
]

# 13 aggregate (profile-level) features
_AGGREGATE_FEAT_NAMES = [
    "Global_MAC",              # max allele count across markers
    "Global_mean_allele_cnt",  # mean allele count
    "Global_std_allele_cnt",   # std allele count
    "Global_markers_3plus",    # markers with 3+ alleles
    "Global_markers_5plus",    # markers with 5+ alleles
    "Global_total_OL",         # total out-of-ladder peaks
    "Global_mean_max_height",  # mean of per-marker max height
    "Global_std_max_height",   # std of per-marker max height
    "Global_total_peaks",      # sum of allele counts
    "Global_total_height",     # sum of all heights
    "Global_n_missing_markers",# number of padded/missing markers
    "Global_multiplex_id",     # kit encoding
    "Global_injection_time_id",# injection time encoding
]


def get_feature_names(scenario_name: str, n_features: int) -> list:
    """
    Tạo danh sách tên feature mô tả cho scenario đã cho.
    Format: {MARKER}_{feature} cho per-marker features,
            Global_{name} cho aggregate features.
    Nếu n_features không khớp, fallback về F{i:03d}.
    """
    from src.config import SCENARIOS, MARKERS_TO_REMOVE

    try:
        markers = sorted(SCENARIOS[scenario_name]["markers_to_keep"])
    except KeyError:
        return [f"F{i:03d}" for i in range(n_features)]

    n_per_marker = len(_MARKER_FEAT_NAMES)  # 17
    n_markers    = len(markers)
    expected_n   = n_markers * n_per_marker + len(_AGGREGATE_FEAT_NAMES)

    if n_features != expected_n:
        # The actual number doesn’t match — fall back to generic names
        print(f"  [WARN] get_feature_names: expected {expected_n} features for "
              f"scenario '{scenario_name}', got {n_features}. Using generic names.")
        return [f"F{i:03d}" for i in range(n_features)]

    names = []
    for marker in markers:
        for feat in _MARKER_FEAT_NAMES:
            names.append(f"{marker}_{feat}")
    names.extend(_AGGREGATE_FEAT_NAMES)
    return names

def run_shap_analysis(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    scenario_name: str,
    results_dir: str,
    feature_names: list = None,
    max_display: int = 20,
    n_background: int = 100,
):
    """
    Chạy SHAP TreeExplainer trên best_model và lưu 4 loại biểu đồ:
      1. Summary plot (beeswarm)   — toàn bộ test set, tất cả class
      2. Bar plot (mean |SHAP|)    — feature importance tổng hợp
      3. Waterfall plot            — 1 mẫu đại diện mỗi class
      4. Decision plot             — 20 mẫu ngẫu nhiên theo class
    """
    print(f"\n[SHAP] Running SHAP TreeExplainer for scenario '{scenario_name}'...")
    shap_dir = os.path.join(results_dir, f"{scenario_name}_shap")
    os.makedirs(shap_dir, exist_ok=True)

    # ── Explainer & values ──────────────────────────────────────
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)          # shape: (n_samples, n_features, n_classes)
    print(f"  SHAP values computed. Shape: {shap_values.values.shape}")

    n_classes = len(class_names)
    n_features = X_test.shape[1]
    # Use descriptive names if provided, else fall back to generic
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f"F{i:03d}" for i in range(n_features)]

    # ── 1. Summary plot (beeswarm, per class) ───────────────────
    print("  [1/4] Summary beeswarm plots...")
    for cls_idx, cls_name in enumerate(class_names):
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values.values[:, :, cls_idx],
            X_test,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title(f"SHAP Beeswarm — {scenario_name} | Class: {cls_name}", fontsize=12)
        plt.tight_layout()
        path = os.path.join(shap_dir, f"shap_beeswarm_class{cls_idx+1}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    # ── 2. Global Bar plot (mean |SHAP| across all classes) ─────
    print("  [2/4] Global bar plot (mean |SHAP|)...")
    # Average absolute SHAP across classes
    mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))  # (n_features,)
    top_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in reversed(top_idx)],
        mean_abs_shap[list(reversed(top_idx))],
        color="#2196F3",
        edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Global Feature Importance — {scenario_name}", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(shap_dir, "shap_bar_global.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Top 5 features: {[feature_names[i] for i in top_idx[:5]]}")

    # ── 3. Waterfall plot — 1 representative sample per class ───
    print("  [3/4] Waterfall plots (1 sample/class)...")
    for cls_idx, cls_name in enumerate(class_names):
        mask = np.where(y_test == cls_idx)[0]
        if len(mask) == 0:
            continue
        sample_idx = mask[0]  # first sample of this class
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values.values[sample_idx, :, cls_idx],
                base_values=shap_values.base_values[sample_idx, cls_idx],
                data=X_test[sample_idx],
                feature_names=feature_names,
            ),
            max_display=15,
            show=False,
        )
        plt.title(f"SHAP Waterfall — {cls_name} (sample #{sample_idx})", fontsize=11)
        plt.tight_layout()
        path = os.path.join(shap_dir, f"shap_waterfall_class{cls_idx+1}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    # ── 4. Decision plot — 20 random samples ────────────────────
    print("  [4/4] Decision plot (20 random samples)...")
    np.random.seed(42)
    sample_size = min(20, len(X_test))
    rand_idx = np.random.choice(len(X_test), size=sample_size, replace=False)
    # Use class 0 (NOC=1) perspective for decision plot
    for cls_idx, cls_name in enumerate(class_names):
        base_val = shap_values.base_values[rand_idx, cls_idx].mean()
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.decision_plot(
            base_val,
            shap_values.values[rand_idx, :, cls_idx],
            feature_names=feature_names,
            feature_display_range=slice(-1, -max_display - 1, -1),
            show=False,
        )
        plt.title(f"SHAP Decision Plot — {cls_name} ({sample_size} samples)", fontsize=11)
        plt.tight_layout()
        path = os.path.join(shap_dir, f"shap_decision_class{cls_idx+1}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  SHAP plots saved to: {shap_dir}/")
    return shap_dir


def run_xgb_scenario(scenario_name: str, skip_preprocessing: bool = False, run_shap: bool = True):
    print(f"\n{'#'*60}")
    print(f"# TAWSEEM XGBoost — Scenario: {scenario_name.upper()}")
    print(f"{'#'*60}")

    processed_path = os.path.join(DATA_PROCESSED_DIR, f"{scenario_name}_processed.csv")

    if skip_preprocessing and os.path.exists(processed_path):
        print(f"\n[1/4] Loading preprocessed data: {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        print(f"\n[1/4] Preprocessing scenario '{scenario_name}'...")
        df = preprocess_scenario(scenario_name)

    print(f"\n[2/4] Feature engineering (profile-level)...")
    train_dataset, test_dataset, _, _, cnn_data, group_data, full_data = prepare_profile_datasets(
        df, train_ratio=TRAIN_RATIO, random_seed=RANDOM_SEED
    )

    X_train_raw = train_dataset.features.numpy()   
    y_train = train_dataset.labels.numpy()      
    X_test_raw  = test_dataset.features.numpy()
    y_test  = test_dataset.labels.numpy()
    groups_train = group_data[0]
    
    X_full_raw, y_full, groups_full, _ = full_data

    print(f"  Train: {X_train_raw.shape[0]} profiles × {X_train_raw.shape[1]} features")
    print(f"  Test:  {X_test_raw.shape[0]} profiles × {X_test_raw.shape[1]} features")

    # ── Feature names (descriptive) ────────────────────────────
    feat_names = get_feature_names(scenario_name, X_train_raw.shape[1])
    print(f"  Feature names: {feat_names[0]} ... {feat_names[-1]}")

    # Use X_train_raw (90%), y_train, groups_train for CV
    try:
        skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        splits = list(skf.split(X_train_raw, groups_train))
        print(f"  CV Stratified Strategy: Strata (NOC x MUX x INJ)")
    except ValueError:
        skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        splits = list(skf.split(X_train_raw, y_train))
        print(f"  CV Stratified Strategy: NOC (Due to small strata < 5 samples)")

    cv_scores = []
    best_acc = 0
    best_model = None
    best_scaler = None
    
    from sklearn.preprocessing import MinMaxScaler
    import copy
    
    cv_start = time.time()
    
    for fold, (trn_idx, val_idx) in enumerate(splits):
        # FIT scaler riêng trên tập train-fold để avoid data leakage vào val-fold
        fold_scaler = MinMaxScaler()
        X_fold_train = fold_scaler.fit_transform(X_train_raw[trn_idx])
        X_fold_val   = fold_scaler.transform(X_train_raw[val_idx])
        
        y_fold_train = y_train[trn_idx]
        y_fold_val   = y_train[val_idx]
        
        # Bug fix 1: khởi tạo model mới mỗi fold (tránh NameError + state leak)
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)
        acc = accuracy_score(y_fold_val, preds)
        
        cv_scores.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            best_scaler = copy.deepcopy(fold_scaler)
            
    cv_scores = np.array(cv_scores)
    cv_time = time.time() - cv_start

    print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Per fold    : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV time     : {cv_time:.1f}s")
    
    print(f"\n[4/4] Testing Best Fold Model (Val Acc: {best_acc:.4f}) on 10% Test Set...")
    eval_start = time.time()
    
    X_train_scaled = best_scaler.transform(X_train_raw)
    X_test_scaled  = best_scaler.transform(X_test_raw)
    
    train_preds = best_model.predict(X_train_scaled)
    test_preds  = best_model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test,  test_preds)
    eval_elapsed = time.time() - eval_start

    print(f"\n  Train (90%) Accuracy : {train_acc:.4f}")
    print(f"  Test  (10%) Accuracy : {test_acc:.4f}")
    print(f"  Eval time            : {eval_elapsed:.1f}s")

    class_names = [f"{i+1}-Person" for i in range(5)]
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, test_preds, target_names=class_names, digits=4))

    # Bug fix: dùng best_model thay vì model (model chỉ là fold cuối)
    importances = best_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print(f"  Top 15 Feature Importances:")
    for rank, idx in enumerate(top_idx, 1):
        fname = feat_names[idx] if idx < len(feat_names) else f"F{idx:03d}"
        print(f"    {rank:2d}. {fname:<35s}: {importances[idx]:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, test_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"XGBoost — {scenario_name} (Test Acc: {test_acc:.4f})")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"{scenario_name}_xgb_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n  Confusion matrix saved: {cm_path}")

    # ── SHAP Analysis ──────────────────────────────────────────────
    if HAS_SHAP and run_shap:
        run_shap_analysis(
            model=best_model,
            X_test=X_test_scaled,
            y_test=y_test,
            class_names=class_names,
            scenario_name=scenario_name,
            results_dir=RESULTS_DIR,
            feature_names=feat_names,
        )

    return {
        "scenario": scenario_name,
        "cv_acc": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "elapsed_time": cv_time + eval_elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="TAWSEEM — XGBoost SOTA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario", type=str, default="single",
        choices=["single", "three", "four", "four_union", "all"],
        help="Scenario to run (default: single)",
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Bỏ qua tiền xử lý nếu CSV đã tồn tại trong data/processed/",
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Tắt SHAP analysis (mặc định: bật nếu shap được cài)",
    )
    args = parser.parse_args()

    scenarios = ["single", "three", "four", "four_union"] if args.scenario == "all" else [args.scenario]

    all_results = []
    for scenario in scenarios:
        result = run_xgb_scenario(
            scenario,
            skip_preprocessing=args.skip_preprocessing,
            run_shap=not args.no_shap,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("TỔNG HỢP KẾT QUẢ XGBOOST")
        print(f"{'='*60}")
        print(f"  {'Scenario':<10} {'CV Acc':>10} {'Test Acc':>10} {'Time':>8}")
        print(f"  {'-'*42}")
        for r in all_results:
            print(
                f"  {r['scenario']:<10} "
                f"{r['cv_acc']:.4f}±{r['cv_std']:.4f}  "
                f"{r['test_acc']:.4f}    "
                f"{r['elapsed_time']:.1f}s"
            )

    print("\n XGBoost training hoàn thành!")


if __name__ == "__main__":
    main()
