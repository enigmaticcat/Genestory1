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


def run_xgb_scenario(scenario_name: str, skip_preprocessing: bool = False):
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

    print(f"\n  Train (90%) Accuracy : {train_acc:.4f}")
    print(f"  Test  (10%) Accuracy : {test_acc:.4f}")
    print(f"  Eval time            : {time.time() - eval_start:.1f}s")

    class_names = [f"{i+1}-Person" for i in range(5)]
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, test_preds, target_names=class_names, digits=4))

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print(f"  Top 15 Feature Importances:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank:2d}. Feature {idx:4d}: {importances[idx]:.4f}")

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

    return {
        "scenario": scenario_name,
        "cv_acc": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "elapsed_time": elapsed,
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
    args = parser.parse_args()

    scenarios = ["single", "three", "four", "four_union"] if args.scenario == "all" else [args.scenario]

    all_results = []
    for scenario in scenarios:
        result = run_xgb_scenario(scenario, skip_preprocessing=args.skip_preprocessing)
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
