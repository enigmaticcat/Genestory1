"""
TAWSEEM Results Summary Generator
Tạo summary report từ tất cả results đã có.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import RESULTS_DIR, NUM_CLASSES


def collect_auc_metrics():
    """Thu thập tất cả AUC metrics từ CSV files."""
    auc_files = []
    results_path = Path(RESULTS_DIR)
    
    if not results_path.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return None
        
    for file in results_path.glob("*_auc_metrics.csv"):
        auc_files.append(file)
    
    if not auc_files:
        print("No AUC metrics files found.")
        return None
        
    all_auc_data = []
    for file in auc_files:
        try:
            df = pd.read_csv(file)
            all_auc_data.append(df)
            print(f"Loaded: {file.name}")
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
            
    if all_auc_data:
        combined_df = pd.concat(all_auc_data, ignore_index=True)
        return combined_df
    
    return None


def generate_performance_summary():
    """Tạo summary về performance của tất cả models."""
    
    print(f"\n{'='*60}")
    print("TAWSEEM PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    # Collect AUC metrics
    auc_df = collect_auc_metrics()
    
    if auc_df is None:
        print("No AUC data available for summary.")
        return
        
    print(f"\nData overview:")
    print(f"  - Total entries: {len(auc_df)}")
    print(f"  - Scenarios: {auc_df['Scenario'].unique().tolist()}")
    print(f"  - Models: {auc_df['Model'].unique().tolist()}")
    
    # Summary by model
    print(f"\n📊 PERFORMANCE BY MODEL")
    print(f"{'='*50}")
    
    model_summary = auc_df.groupby('Model').agg({
        'Macro_AUC': ['mean', 'std', 'min', 'max'],
        'Accuracy': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(model_summary)
    
    # Best performing model per scenario
    print(f"\n🏆 BEST MODEL PER SCENARIO")
    print(f"{'='*50}")
    
    for scenario in auc_df['Scenario'].unique():
        scenario_data = auc_df[auc_df['Scenario'] == scenario]
        best_auc_idx = scenario_data['Macro_AUC'].idxmax()
        best_acc_idx = scenario_data['Accuracy'].idxmax()
        
        best_auc_model = scenario_data.loc[best_auc_idx]
        best_acc_model = scenario_data.loc[best_acc_idx]
        
        print(f"\n{scenario.upper()}:")
        print(f"  Best AUC:      {best_auc_model['Model']} (AUC={best_auc_model['Macro_AUC']:.4f})")
        print(f"  Best Accuracy: {best_acc_model['Model']} (Acc={best_acc_model['Accuracy']:.4f})")
    
    # Per-class AUC analysis
    print(f"\n📈 PER-CLASS AUC ANALYSIS")
    print(f"{'='*50}")
    
    class_columns = ['AUC_1Person', 'AUC_2Person', 'AUC_3Person', 'AUC_4Person', 'AUC_5Person']
    
    for model in auc_df['Model'].unique():
        model_data = auc_df[auc_df['Model'] == model]
        print(f"\n{model}:")
        
        for col in class_columns:
            if col in model_data.columns:
                mean_auc = model_data[col].mean()
                std_auc = model_data[col].std()
                class_name = col.replace('AUC_', '')
                print(f"  {class_name:<10}: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Save summary to file
    summary_file = os.path.join(RESULTS_DIR, 'performance_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("TAWSEEM PERFORMANCE SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("MODEL PERFORMANCE OVERVIEW\n")
        f.write(model_summary.to_string())
        f.write("\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write(auc_df.to_string(index=False))
        
    print(f"\n💾 Summary saved to: {summary_file}")
    
    # Create comparison CSV
    comparison_file = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    auc_df.to_csv(comparison_file, index=False)
    print(f"💾 Comparison data saved to: {comparison_file}")


def list_generated_files():
    """List tất cả files đã được tạo."""
    
    results_path = Path(RESULTS_DIR)
    if not results_path.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return
        
    print(f"\n📁 FILES IN RESULTS DIRECTORY")
    print(f"{'='*50}")
    print(f"Location: {RESULTS_DIR}\n")
    
    file_types = {
        'AUC Metrics': '*_auc_metrics.csv',
        'ROC Curves (3-Model)': '*_three_model_roc_curves.png', 
        'Dataset Distribution': '*_dataset_distribution.png',
        'Confusion Matrix': '*_confusion_matrix_*.png',
        'Performance Metrics': '*_performance_metrics.png',
        'Individual ROC': '*_roc_individual.png',
        'Summary Reports': 'performance_summary.txt'
    }
    
    total_files = 0
    
    for file_type, pattern in file_types.items():
        files = list(results_path.glob(pattern))
        if files:
            print(f"{file_type}:")
            for file in sorted(files):
                print(f"  ✓ {file.name}")
                total_files += 1
            print()
    
    print(f"Total files: {total_files}")


def main():
    """Main function."""
    
    print("TAWSEEM Results Summary Generator")
    print("Generates comprehensive summary from all evaluation results.")
    
    try:
        # Generate performance summary
        generate_performance_summary()
        
        # List all files
        list_generated_files()
        
        print(f"\n{'='*60}")
        print("✅ SUMMARY GENERATION COMPLETED!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Error during summary generation:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())