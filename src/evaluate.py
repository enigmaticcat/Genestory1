"""
TAWSEEM Evaluation & Visualization
Metrics computation, confusion matrix, and performance plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import NUM_CLASSES, RESULTS_DIR


CLASS_NAMES = ['1-Person', '2-Person', '3-Person', '4-Person', '5-Person']


def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, F1-score (overall and per-class).
    
    Labels should be 0-indexed (0-4) for classes (1-5 contributors).
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0),
        'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0),
        'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
    }
    return metrics


def print_metrics(metrics):
    """Print metrics in a readable format."""
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f} (macro)")
    print(f"  Recall:    {metrics['recall_macro']:.4f} (macro)")
    print(f"  F1-Score:  {metrics['f1_macro']:.4f} (macro)")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print(f"  {'-'*44}")
    for i in range(len(metrics['precision_per_class'])):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
        print(f"  {name:<12} {metrics['precision_per_class'][i]:>10.4f} "
              f"{metrics['recall_per_class'][i]:>10.4f} "
              f"{metrics['f1_per_class'][i]:>10.4f}")


def plot_confusion_matrix(metrics, title, save_path):
    """
    Plot confusion matrix (replicates Figure 22, 25, 27 from paper).
    """
    cm = metrics['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_precision_recall_f1(metrics, title, save_path):
    """
    Plot precision, recall, F1-score per class (replicates Figure 23, 26, 28).
    """
    x = np.arange(NUM_CLASSES)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', color='#2196F3')
    bars2 = ax.bar(x, metrics['recall_per_class'], width, label='Recall', color='#4CAF50')
    bars3 = ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score', color='#FF9800')
    
    ax.set_xlabel('Number of Contributors', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_prediction_errors(metrics, title, save_path):
    """
    Plot prediction error visualization (replicates Figure 24).
    Shows misclassification patterns per class.
    """
    cm = metrics['confusion_matrix']
    n_classes = cm.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    
    for true_class in range(n_classes):
        errors = {}
        total = cm[true_class].sum()
        for pred_class in range(n_classes):
            if pred_class != true_class and cm[true_class, pred_class] > 0:
                errors[pred_class] = cm[true_class, pred_class] / total * 100
        
        if errors:
            x_positions = list(errors.keys())
            heights = list(errors.values())
            x_offset = (true_class - n_classes / 2 + 0.5) * 0.15
            ax.bar([x + x_offset for x in x_positions], heights, 0.14,
                   label=f'True: {CLASS_NAMES[true_class]}', color=colors[true_class], alpha=0.8)
    
    ax.set_xlabel('Predicted As', fontsize=12)
    ax.set_ylabel('Error Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_comparison(results_dict, save_path):
    """
    Plot accuracy comparison across scenarios (replicates Figure 29).
    
    Args:
        results_dict: {scenario_name: {'train_acc': float, 'test_acc': float}}
    """
    scenarios = list(results_dict.keys())
    train_accs = [results_dict[s]['train_acc'] for s in scenarios]
    test_accs = [results_dict[s]['test_acc'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, train_accs, width, label='Training', color='#2196F3')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Testing', color='#FF9800')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('TAWSEEM: Accuracy Comparison Across Scenarios', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_all_plots(train_metrics, test_metrics, scenario_name):
    """Generate all plots for a scenario."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    scenario_display = scenario_name.replace('_', ' ').title()
    
    print(f"\nGenerating plots for {scenario_display}...")
    
    # Confusion matrices
    plot_confusion_matrix(
        train_metrics,
        f'TAWSEEM Training Confusion Matrix ({scenario_display})',
        os.path.join(RESULTS_DIR, f'{scenario_name}_cm_train.png')
    )
    plot_confusion_matrix(
        test_metrics,
        f'TAWSEEM Testing Confusion Matrix ({scenario_display})',
        os.path.join(RESULTS_DIR, f'{scenario_name}_cm_test.png')
    )
    
    # Precision/Recall/F1
    plot_precision_recall_f1(
        train_metrics,
        f'TAWSEEM Precision/Recall/F1 - Training ({scenario_display})',
        os.path.join(RESULTS_DIR, f'{scenario_name}_prf_train.png')
    )
    plot_precision_recall_f1(
        test_metrics,
        f'TAWSEEM Precision/Recall/F1 - Testing ({scenario_display})',
        os.path.join(RESULTS_DIR, f'{scenario_name}_prf_test.png')
    )
    
    # Prediction errors
    plot_prediction_errors(
        test_metrics,
        f'TAWSEEM Prediction Errors ({scenario_display})',
        os.path.join(RESULTS_DIR, f'{scenario_name}_errors.png')
    )
