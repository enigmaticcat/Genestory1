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


def train_tree_models(train_dataset, test_dataset, groups_train, scenario_name):
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
        
        # 5-Fold Group CV
        gkf = GroupKFold(n_splits=NUM_CV_FOLDS)
        cv_scores = cross_val_score(model, X_train, y_train, groups=groups_train, cv=gkf, scoring='accuracy')
        
        cv_time = time.time() - start_time
        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Per fold: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"  CV time: {cv_time:.1f}s")
        
        # Train on full training set
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        
        print(f"\n  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Train time: {train_time:.1f}s")
        
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
            'model': model,
            'time': train_time,
        }
    
    return results
