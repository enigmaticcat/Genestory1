"""
TAWSEEM Tree-based Models (XGBoost + Random Forest)
Alternative to MLP for profile-level NOC prediction.
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import NUM_CV_FOLDS, RANDOM_SEED, RESULTS_DIR

# Try to import XGBoost, fall back to GradientBoosting if not available
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("  XGBoost not installed, using sklearn GradientBoosting instead")


def get_models():
    """Return dict of tree-based models to try."""
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight='balanced',
        ),
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            eval_metric='mlogloss',
            verbosity=0,
        )
    else:
        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            random_state=RANDOM_SEED,
        )
    
    return models


def train_tree_models(train_dataset, test_dataset, groups_train, full_data, scenario_name):
    """
    Train and evaluate tree-based models (RF + XGBoost).
    
    Args:
        train_dataset: DNAProfileDataset (already scaled)
        test_dataset: DNAProfileDataset (already scaled)
        scenario_name: for printing
    
    Returns: dict of {model_name: {cv_acc, test_acc, model, metrics}}
    """
    print(f"\n{'='*50}")
    print(f"Tree-based Models")
    print(f"{'='*50}")
    
    # Extract numpy arrays
    X_train = train_dataset.features.numpy()
    y_train = train_dataset.labels.numpy()
    X_test = test_dataset.features.numpy()
    y_test = test_dataset.labels.numpy()
    
    print(f"  Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples × {X_test.shape[1]} features")
    print(f"  Classes: {np.unique(y_train)}")
    
    models = get_models()
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        start_time = time.time()
        
        X_cv, y_cv, groups_cv = X_train, y_train, groups_train
        try:
            skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            splits = list(skf.split(X_cv, groups_cv))
        except ValueError:
            skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            splits = list(skf.split(X_cv, y_cv))
            
        cv_scores = []
        best_acc = 0
        best_model = None
        best_scaler = None
        
        from sklearn.preprocessing import MinMaxScaler
        import copy
        
        for trn_idx, val_idx in splits:
            fold_scaler = MinMaxScaler()
            X_fold_train = fold_scaler.fit_transform(X_cv[trn_idx])
            X_fold_val   = fold_scaler.transform(X_cv[val_idx])
            y_fold_train = y_cv[trn_idx]
            y_fold_val   = y_cv[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            acc = accuracy_score(y_fold_val, preds)
            cv_scores.append(acc)
            
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model)
                best_scaler = copy.deepcopy(fold_scaler)
        
        cv_scores = np.array(cv_scores)
        
        cv_time = time.time() - start_time
        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Per fold: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"  CV time: {cv_time:.1f}s")
        
        print(f"\n  [Testing Best Fold Model (Val Acc: {best_acc:.4f}) on 10% Test Set]")
        start_time = time.time()
        
        # Evaluate Best Model
        X_train_scaled = best_scaler.transform(X_train)
        X_test_scaled = best_scaler.transform(X_test)
        
        
        train_preds = best_model.predict(X_train_scaled)
        test_preds = best_model.predict(X_test_scaled)
        
        test_preds_proba = None
        if hasattr(best_model, 'predict_proba'):
            test_preds_proba = best_model.predict_proba(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        
        print(f"\n  Train (90%) Accuracy: {train_acc:.4f}")
        print(f"  Test  (10%) Accuracy: {test_acc:.4f}")
        print(f"  Eval time: {time.time() - start_time:.1f}s")
        
        # Detailed test metrics
        class_names = [f"{i+1}-Person" for i in range(len(np.unique(y_train)))]
        print(f"\n  Test Classification Report:")
        print(classification_report(y_test, test_preds, target_names=class_names, digits=4))
        
        # Feature importance (top 10)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            print(f"  Top 10 features:")
            for i, idx in enumerate(top_idx):
                print(f"    {i+1}. Feature {idx}: {importances[idx]:.4f}")
        
        results[name] = {
            'cv_acc': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_acc': test_acc,
            'train_acc': train_acc,
            'model': best_model,
            'scaler': best_scaler,
            'time': start_time,
            'test_preds': test_preds,
            'test_preds_proba': test_preds_proba
        }
    
    return results
