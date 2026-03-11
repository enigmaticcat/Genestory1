"""
TAWSEEM Plot Generator Utility
Tạo specific plots mà không cần chạy toàn bộ pipeline.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import RESULTS_DIR, DATA_PROCESSED_DIR
from src.evaluate import (
    plot_dataset_distribution,
    plot_three_model_roc_comparison,
    save_auc_metrics_to_csv
)


def load_processed_data(scenario_name):
    """Load processed data cho scenario."""
    
    processed_file = os.path.join(DATA_PROCESSED_DIR, f"{scenario_name}_processed.csv")
    
    if not os.path.exists(processed_file):
        print(f"❌ Processed data not found: {processed_file}")
        print(f"   Available files in {DATA_PROCESSED_DIR}:")
        if os.path.exists(DATA_PROCESSED_DIR):
            for file in os.listdir(DATA_PROCESSED_DIR):
                if file.endswith('.csv'):
                    print(f"     - {file}")
        return None
    
    try:
        df = pd.read_csv(processed_file)
        print(f"✅ Loaded data: {processed_file}")
        print(f"   Shape: {df.shape}")
        print(f"   NOC distribution: {df['NOC'].value_counts().sort_index().to_dict()}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def create_distribution_plot(scenario_name):
    """Tạo dataset distribution plot cho scenario."""
    
    print(f"\n{'='*50}")
    print(f"Creating Dataset Distribution Plot for {scenario_name}")
    print(f"{'='*50}")
    
    df = load_processed_data(scenario_name)
    if df is None:
        return False
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    plot_dataset_distribution(
        df,
        f"TAWSEEM Dataset Distribution - {scenario_name.replace('_', ' ').title()}",
        os.path.join(RESULTS_DIR, f'{scenario_name}_dataset_distribution.png')
    )
    
    print(f"✅ Distribution plot created successfully!")
    return True


def create_mock_roc_plots(scenario_name):
    """Tạo mock ROC plots để demo visualization."""
    
    print(f"\n{'='*50}")
    print(f"Creating Mock ROC Plots for {scenario_name}")
    print(f"{'='*50}")
    
    # Create realistic mock data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # Generate realistic class distribution
    class_probs = [0.3, 0.25, 0.2, 0.15, 0.1]  # 1-Person most common
    y_true = np.random.choice(range(n_classes), n_samples, p=class_probs)
    
    print(f"Mock data:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Class distribution: {np.bincount(y_true)}")
    
    # Generate model probabilities với realistic performance differences
    model_probs = {}
    
    # MLP - best performance
    mlp_probs = np.random.dirichlet(np.ones(n_classes) * 3, n_samples)
    for i in range(n_samples):
        mlp_probs[i, y_true[i]] *= 2.0  # Boost correct class
        mlp_probs[i] = mlp_probs[i] / mlp_probs[i].sum()
    model_probs['MLP'] = mlp_probs
    
    # Random Forest - good performance  
    rf_probs = np.random.dirichlet(np.ones(n_classes) * 2.5, n_samples)
    for i in range(n_samples):
        rf_probs[i, y_true[i]] *= 1.7
        rf_probs[i] = rf_probs[i] / rf_probs[i].sum()
    model_probs['RandomForest'] = rf_probs
    
    # XGBoost - medium performance
    xgb_probs = np.random.dirichlet(np.ones(n_classes) * 2.0, n_samples)  
    for i in range(n_samples):
        xgb_probs[i, y_true[i]] *= 1.5
        xgb_probs[i] = xgb_probs[i] / xgb_probs[i].sum()
    model_probs['XGBoost'] = xgb_probs
    
    # Mock tree results for accuracy display
    tree_results = {
        'RandomForest': {'test_acc': 0.82},
        'XGBoost': {'test_acc': 0.78}
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create ROC comparison plot
    plot_three_model_roc_comparison(
        model_probs, y_true, scenario_name, tree_results
    )
    
    # Create AUC metrics CSV
    save_auc_metrics_to_csv(
        model_probs, y_true, scenario_name, tree_results, 
        test_metrics={'accuracy': 0.85}
    )
    
    print(f"✅ Mock ROC plots và AUC metrics created!")
    return True


def main():
    """Main function với command line arguments."""
    
    parser = argparse.ArgumentParser(description='TAWSEEM Plot Generator Utility')
    
    parser.add_argument('--scenario', type=str, default='single',
                        help='Scenario name (default: single)')
    
    parser.add_argument('--plot-type', type=str, choices=['distribution', 'roc', 'both'],
                        default='both', help='Type of plot to generate')
    
    parser.add_argument('--mock-data', action='store_true',
                        help='Use mock data instead of real processed data')
    
    args = parser.parse_args()
    
    print("TAWSEEM Plot Generator Utility")
    print(f"Scenario: {args.scenario}")
    print(f"Plot type: {args.plot_type}")
    print(f"Use mock data: {args.mock_data}")
    
    success_count = 0
    total_tasks = 0
    
    try:
        if args.plot_type in ['distribution', 'both']:
            total_tasks += 1
            if args.mock_data:
                # Create mock distribution
                np.random.seed(42)
                noc_dist = np.random.choice(range(1, 6), 1000, p=[0.3, 0.25, 0.2, 0.15, 0.1])
                df_mock = pd.DataFrame({'NOC': noc_dist})
                
                os.makedirs(RESULTS_DIR, exist_ok=True)
                plot_dataset_distribution(
                    df_mock,
                    f"TAWSEEM Mock Dataset Distribution - {args.scenario.title()}",
                    os.path.join(RESULTS_DIR, f'{args.scenario}_mock_distribution.png')
                )
                print(f"✅ Mock distribution plot created!")
                success_count += 1
            else:
                if create_distribution_plot(args.scenario):
                    success_count += 1
        
        if args.plot_type in ['roc', 'both']:
            total_tasks += 1
            if create_mock_roc_plots(args.scenario):
                success_count += 1
        
        print(f"\n{'='*60}")
        if success_count == total_tasks:
            print("🎉 ALL PLOTS GENERATED SUCCESSFULLY!")
        else:
            print(f"⚠️  PARTIAL SUCCESS: {success_count}/{total_tasks} tasks completed")
        print(f"{'='*60}")
        
        print(f"\n📁 Results saved in: {RESULTS_DIR}")
        
        # List created files
        if os.path.exists(RESULTS_DIR):
            scenario_files = [f for f in os.listdir(RESULTS_DIR) 
                            if f.startswith(args.scenario)]
            if scenario_files:
                print(f"\nGenerated files:")
                for i, filename in enumerate(sorted(scenario_files), 1):
                    print(f"  {i}. {filename}")
        
    except Exception as e:
        print(f"\n❌ Error during plot generation:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0 if success_count == total_tasks else 1


if __name__ == "__main__":
    exit(main())