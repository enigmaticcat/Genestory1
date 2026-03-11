#!/usr/bin/env python3
"""
TAWSEEM Evaluation Only Script
Chạy evaluation và tạo plots mà không cần train models.
Dùng cho việc test các evaluation functions mới.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import RESULTS_DIR, NUM_CLASSES
from src.evaluate import (
    save_auc_metrics_to_csv,
    plot_three_model_roc_comparison,
    plot_dataset_distribution,
    generate_comprehensive_evaluation
)


def create_mock_data():
    """
    Tạo mock data để test các evaluation functions.
    """
    print("Creating mock data for evaluation testing...")
    
    # Mock true labels (500 samples)
    np.random.seed(42)
    n_samples = 500
    y_true = np.random.randint(0, NUM_CLASSES, n_samples)
    
    # Mock probabilities cho 3 models
    model_probs = {}
    
    # MLP probabilities (slightly better performance)
    mlp_probs = np.random.dirichlet(np.ones(NUM_CLASSES) * 3, n_samples)
    # Boost correct class probabilities
    for i in range(n_samples):
        mlp_probs[i, y_true[i]] *= 1.5
        mlp_probs[i] = mlp_probs[i] / mlp_probs[i].sum()
    model_probs['MLP'] = mlp_probs
    
    # Random Forest probabilities  
    rf_probs = np.random.dirichlet(np.ones(NUM_CLASSES) * 2.5, n_samples)
    for i in range(n_samples):
        rf_probs[i, y_true[i]] *= 1.3
        rf_probs[i] = rf_probs[i] / rf_probs[i].sum()
    model_probs['RandomForest'] = rf_probs
    
    # XGBoost probabilities
    xgb_probs = np.random.dirichlet(np.ones(NUM_CLASSES) * 2.8, n_samples)
    for i in range(n_samples):
        xgb_probs[i, y_true[i]] *= 1.4
        xgb_probs[i] = xgb_probs[i] / xgb_probs[i].sum()
    model_probs['XGBoost'] = xgb_probs
    
    # Mock tree results
    tree_results = {
        'RandomForest': {
            'test_acc': 0.78,
            'cv_acc': 0.76,
            'cv_std': 0.03
        },
        'XGBoost': {
            'test_acc': 0.82,
            'cv_acc': 0.80,
            'cv_std': 0.025
        }
    }
    
    # Mock test metrics
    test_metrics = {
        'accuracy': 0.85,
        'y_true': y_true,
        'y_pred': np.argmax(mlp_probs, axis=1)
    }
    
    # Mock dataset for distribution
    noc_distribution = np.random.choice(range(1, 6), n_samples, 
                                      p=[0.3, 0.25, 0.2, 0.15, 0.1])
    df_mock = pd.DataFrame({
        'NOC': noc_distribution,
        'Sample File': [f'sample_{i}.csv' for i in range(n_samples)]
    })
    
    return y_true, model_probs, tree_results, test_metrics, df_mock


def test_evaluation_functions():
    """
    Test tất cả evaluation functions với mock data.
    """
    print(f"\n{'='*60}")
    print("TAWSEEM Evaluation Functions Testing")
    print(f"{'='*60}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create mock data
    y_true, model_probs, tree_results, test_metrics, df_mock = create_mock_data()
    
    scenario_name = "test_scenario"
    
    print(f"\nMock data created:")
    print(f"  - Samples: {len(y_true)}")
    print(f"  - Models: {list(model_probs.keys())}")
    print(f"  - Classes: {NUM_CLASSES}")
    
    # Test 1: Dataset Distribution Plot
    print(f"\n[1/4] Testing dataset distribution plot...")
    plot_dataset_distribution(
        df_mock,
        "TAWSEEM Mock Dataset Distribution",
        os.path.join(RESULTS_DIR, f'{scenario_name}_mock_distribution.png')
    )
    
    # Test 2: Three-Model ROC Comparison
    print(f"\n[2/4] Testing 3-model ROC curves...")
    plot_three_model_roc_comparison(
        model_probs, y_true, scenario_name, tree_results
    )
    
    # Test 3: AUC Metrics CSV
    print(f"\n[3/4] Testing AUC metrics CSV export...")
    auc_df = save_auc_metrics_to_csv(
        model_probs, y_true, scenario_name, tree_results, test_metrics
    )
    print("AUC Metrics Preview:")
    print(auc_df.round(4))
    
    # Test 4: Comprehensive Evaluation
    print(f"\n[4/4] Testing comprehensive evaluation...")
    generate_comprehensive_evaluation(
        test_metrics,  # train_metrics (dùng test_metrics để đơn giản)
        test_metrics,  # test_metrics
        scenario_name,
        df=df_mock,
        model_probs=model_probs,
        tree_results=tree_results
    )
    
    print(f"\n✅ All evaluation functions tested successfully!")
    print(f"   Results saved in: {RESULTS_DIR}")
    print(f"\n📊 Generated files:")
    
    # List generated files
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith(scenario_name)]
    for i, filename in enumerate(sorted(result_files), 1):
        print(f"   {i}. {filename}")


def main():
    """Main function để chạy evaluation testing."""
    
    print("TAWSEEM Evaluation Testing Script")
    print("This script tests evaluation functions without running the full pipeline.")
    print("Useful for debugging and testing new evaluation features.")
    
    try:
        test_evaluation_functions()
        
        print(f"\n{'='*60}")
        print("🎉 EVALUATION TESTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation testing:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())