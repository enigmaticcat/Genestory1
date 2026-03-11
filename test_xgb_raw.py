"""
TEMPORARY TEST: XGBoost on raw concatenated features (1540 features).
Run: python test_xgb_raw.py
Delete this file after testing.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

from src.config import RANDOM_SEED, TRAIN_RATIO
from src.data_preprocessing import preprocess_scenario

try:
    from xgboost import XGBClassifier
except ImportError:
    print("pip install xgboost first!")
    sys.exit(1)


def raw_concat_profile(df):
    """Concatenate all markers into 1 vector per profile (raw features)."""
    profile_col = 'Sample File'
    per_marker_cols = [c for c in df.columns if c not in ('NOC', 'Sample File', 'Dye', 'Marker')]
    
    df_sorted = df.sort_values([profile_col, 'Marker']).reset_index(drop=True)
    df_sorted = df_sorted.drop_duplicates(subset=[profile_col, 'Marker'], keep='first').reset_index(drop=True)
    
    # Keep only profiles with most common marker count
    mpp = df_sorted.groupby(profile_col).size()
    n_markers = mpp.mode().iloc[0]
    valid = mpp[mpp == n_markers].index
    df_sorted = df_sorted[df_sorted[profile_col].isin(valid)].reset_index(drop=True)
    
    has_inj = 'injection_time' in df_sorted.columns
    
    profiles, labels, metas = [], [], []
    for sf, group in df_sorted.groupby(profile_col, sort=False):
        profiles.append(group[per_marker_cols].values.flatten())
        labels.append(group['NOC'].iloc[0])
        metas.append((
            group['NOC'].iloc[0], 
            group['injection_time'].iloc[0] if has_inj else group['NOC'].iloc[0]
        ))
        
    # Return profiles, labels, and metadata for stratified splitting
    return np.array(profiles, dtype=np.float32), np.array(labels), pd.DataFrame(metas, columns=['NOC', 'inj'])


def main():
    print("=" * 60)
    print("TEST: XGBoost on raw concatenated features")
    print("=" * 60)
    
    # Preprocess
    import argparse
    parser = argparse.ArgumentParser(description="TAWSEEM Raw XGBoost")
    parser.add_argument('--scenario', type=str, default='single',
                        choices=['single', 'three', 'four', 'four_union'],
                        help='Scenario to run')
    args = parser.parse_args()
    
    df = preprocess_scenario(args.scenario)
    X, y, df_meta = raw_concat_profile(df)
    n_features = X.shape[1]
    print(f"\nData: {X.shape[0]} profiles × {n_features} features")
    
    # Stratified split by (NOC x inj)
    np.random.seed(RANDOM_SEED)
    df_meta['strata'] = df_meta['NOC'].astype(str) + '_' + df_meta['inj'].astype(str)
    
    train_idx, test_idx = [], []
    for sv in sorted(df_meta['strata'].unique()):
        idx = df_meta[df_meta['strata'] == sv].index.tolist()
        np.random.shuffle(idx)
        n_train = int(len(idx) * TRAIN_RATIO)
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])
    
    X_train, y_train = X[train_idx], y[train_idx] - 1  # NOC 1-5 -> 0-4
    X_test, y_test = X[test_idx], y[test_idx] - 1
    groups_train = df_meta['strata'].values[train_idx]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # --- XGBoost on raw features ---
    print(f"\n--- XGBoost (raw {n_features} features) ---")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.3,  # lower colsample for high-dim
        min_child_weight=3, reg_alpha=0.5, reg_lambda=2.0,
        random_state=RANDOM_SEED, eval_metric='mlogloss', verbosity=0,
    )
    
    gkf = GroupKFold(n_splits=5)
    
    # Custom CV with scaling per fold to avoid leakage
    cv_scores_xgb = []
    cv_scores_rf = []
    
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=1000, max_depth=None, min_samples_split=3,
        max_features=0.1, random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced',
    )
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        scaler = MinMaxScaler()
        X_fold_train = scaler.fit_transform(X_train[trn_idx])
        X_fold_val = scaler.transform(X_train[val_idx])
        
        y_f_train = y_train[trn_idx]
        y_f_val = y_train[val_idx]
        
        # XGBoost
        xgb.fit(X_fold_train, y_f_train)
        preds_xgb = xgb.predict(X_fold_val)
        cv_scores_xgb.append(accuracy_score(y_f_val, preds_xgb))
        
        # RF
        rf.fit(X_fold_train, y_f_train)
        preds_rf = rf.predict(X_fold_val)
        cv_scores_rf.append(accuracy_score(y_f_val, preds_rf))

    cv_xgb = np.array(cv_scores_xgb)
    cv_rf = np.array(cv_scores_rf)
    
    # Scale full train for final models
    final_scaler = MinMaxScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    
    print(f"XGB CV: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}  {[f'{s:.4f}' for s in cv_xgb]}")
    xgb.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, xgb.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, xgb.predict(X_test_scaled))
    print(f"XGB Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    print(f"\n--- RandomForest (raw {n_features} features) ---")
    print(f"RF CV: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}  {[f'{s:.4f}' for s in cv_rf]}")
    rf.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
    print(f"RF Train: {train_acc:.4f}, Test: {test_acc:.4f}")



if __name__ == "__main__":
    main()
