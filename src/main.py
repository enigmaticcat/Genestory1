"""
TAWSEEM: Main Entry Point
Run the full pipeline: preprocess → train (with CV) → evaluate → plot.
"""

import os
import sys
import argparse
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, TRAIN_RATIO, RANDOM_SEED
from src.data_preprocessing import preprocess_scenario
from src.dataset import prepare_datasets, prepare_profile_datasets
from src.train import cross_validate, train_final_model
from src.tree_models import train_tree_models
from src.evaluate import (
    generate_all_plots, plot_accuracy_comparison, generate_comprehensive_evaluation,
    print_unified_test_results, plot_all_confusion_matrices
)


def run_scenario(scenario_name, skip_preprocessing=False, skip_cv=False):
    """Run full pipeline for one scenario."""
    
    print(f"\n{'#'*60}")
    print(f"# TAWSEEM — Scenario: {scenario_name}")
    print(f"{'#'*60}")
    
    # --- Step 1: Preprocessing ---
    processed_path = os.path.join(DATA_PROCESSED_DIR, f"{scenario_name}_processed.csv")
    
    if skip_preprocessing and os.path.exists(processed_path):
        print(f"\nLoading preprocessed data from {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        df = preprocess_scenario(scenario_name)
    
    # --- Step 2: Prepare PyTorch datasets (profile-level) ---
    print(f"\nPreparing PyTorch datasets (profile-level)...")
    train_dataset, test_dataset, scaler, train_profile_ids, cnn_data, group_data, full_data = prepare_profile_datasets(
        df, train_ratio=TRAIN_RATIO, random_seed=RANDOM_SEED
    )
    
    groups_train, groups_test = group_data
    n_features = train_dataset.n_features
    
    # --- Step 3: Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    print(f"Using device: {device}")
    
    _, _, y_train_raw, _, class_weights = cnn_data

    # --- Step 4: Cross-validation (MLP) ---
    if not skip_cv:
        cv_accuracies = cross_validate(
            full_data, n_features, device, scenario_name,
            class_weights=class_weights,
        )
    
    # --- Step 5: Train final MLP model ---
    model, train_metrics, test_metrics, elapsed = train_final_model(
        train_dataset, test_dataset, n_features, device, scenario_name,
        class_weights=class_weights,
    )
    
    # --- Step 6: Train tree-based models (RF + XGBoost) ---
    tree_results = train_tree_models(train_dataset, test_dataset, groups_train, full_data, scenario_name)
    
    # --- Step 7: Generate comprehensive evaluation ---
    model_probs = {
        'MLP': test_metrics.get('probs'),
        'Random Forest': tree_results.get('RandomForest', {}).get('test_preds_proba'),
        'XGBoost': tree_results.get('XGBoost', {}).get('test_preds_proba')
    }
    
    # Use comprehensive evaluation để tạo đầy đủ plots và metrics
    generate_comprehensive_evaluation(
        train_metrics, test_metrics, scenario_name, 
        df=df, model_probs=model_probs, tree_results=tree_results
    )
    
    # --- Step 8: Print unified test results for all 3 models ---
    print_unified_test_results(test_metrics, tree_results, scenario_name)
    
    # --- Step 9: Plot all confusion matrices together ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_all_confusion_matrices(test_metrics, tree_results, scenario_name)
    
    # Find best model
    best_tree_name = max(tree_results, key=lambda k: tree_results[k]['test_acc'])
    best_tree_acc = tree_results[best_tree_name]['test_acc']
    mlp_acc = test_metrics['accuracy']
    
    print(f"\n{'='*50}")
    print(f"MODEL COMPARISON — {scenario_name}")
    print(f"{'='*50}")
    print(f"  MLP:           Test Acc = {mlp_acc:.4f}")
    for name, res in tree_results.items():
        marker = " ← BEST" if res['test_acc'] == max(mlp_acc, best_tree_acc) else ""
        print(f"  {name:15s} Test Acc = {res['test_acc']:.4f} (CV: {res['cv_acc']:.4f} ± {res['cv_std']:.4f}){marker}")
    
    return {
        'train_acc': train_metrics['accuracy'],
        'test_acc': test_metrics['accuracy'],
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'elapsed_time': elapsed,
        'tree_results': tree_results,
    }


def main():
    parser = argparse.ArgumentParser(description='TAWSEEM DNA Profiling Tool')
    parser.add_argument('--scenario', type=str, default='single',
                        choices=['single', 'three', 'four', 'four_union', 'all'],
                        help='Scenario to run (default: single)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing if processed CSV exists')
    parser.add_argument('--skip-cv', action='store_true',
                        help='Skip cross-validation (faster)')
    args = parser.parse_args()
    
    if args.scenario == 'all':
        scenarios = ['single', 'three', 'four', 'four_union']
    else:
        scenarios = [args.scenario]
    
    all_results = {}
    
    for scenario in scenarios:
        results = run_scenario(
            scenario, 
            skip_preprocessing=args.skip_preprocessing,
            skip_cv=args.skip_cv,
        )
        all_results[scenario] = results
    
    # If multiple scenarios, create comparison plot
    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print("COMPARISON ACROSS SCENARIOS")
        print(f"{'='*50}")
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plot_accuracy_comparison(
            all_results,
            os.path.join(RESULTS_DIR, 'accuracy_comparison.png')
        )
        
        for name, res in all_results.items():
            print(f"  {name}: Train Acc={res['train_acc']:.4f}, "
                  f"Test Acc={res['test_acc']:.4f}, "
                  f"Time={res['elapsed_time']:.1f}s")
    
    print("\n✅ TAWSEEM pipeline completed!")
    return all_results


if __name__ == "__main__":
    main()
