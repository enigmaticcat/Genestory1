"""
TAWSEEM Training Pipeline
Training loop with 5-fold cross-validation.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CV_FOLDS, 
    RANDOM_SEED, RESULTS_DIR, EARLY_STOPPING_PATIENCE,
)
from src.model import TAWSEEM_MLP
from src.evaluate import compute_metrics, print_metrics


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(pt) = -αt * (1 - pt)^γ * log(pt)
    
    Đầy là tiêu chuẩn tốt hơn CrossEntropyLoss cho dữ liệu mất cân bằng cực đoan.
    - γ (gamma): tập trung vào hard examples. γ=2 là giá trị phổ biến.
    - alpha (class weights): đư ợc truyền từ class_weights dict.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # Tensor [n_classes]

    def forward(self, inputs, targets):
        # inputs: (N, C), targets: (N,)
        log_pt = nn.functional.log_softmax(inputs, dim=1)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        pt = log_pt.exp()

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += features.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model. Returns loss, accuracy, all predictions and labels."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_samples += features.size(0)
    
    avg_loss = total_loss / total_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    
    return avg_loss, accuracy, all_preds, all_labels


def cross_validate(train_dataset, n_features, device, scenario_name, profile_ids=None, class_weights=None):
    """
    Perform 5-fold cross-validation.
    
    Returns: list of fold accuracies
    """
    print(f"\n{'='*50}")
    print(f"5-Fold Cross-Validation")
    print(f"{'='*50}")

    # Build Focal Loss alpha (NOC 1-5 → index 0-4)
    if class_weights is not None:
        alpha_tensor = torch.tensor(
            [class_weights.get(i + 1, 1.0) for i in range(5)],
            dtype=torch.float32
        ).to(device)
        print(f"  Focal Loss alpha: {[f'{class_weights.get(i+1, 1.0):.3f}' for i in range(5)]}")
    else:
        alpha_tensor = None
    
    labels = train_dataset.labels.numpy()
    fold_accuracies = []
    
    skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Fold {fold + 1}/{NUM_CV_FOLDS} ---")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = TAWSEEM_MLP(input_dim=n_features).to(device)
        criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{EPOCHS}: "
                      f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                      f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f} "
                      f"[patience {patience_counter}/{EARLY_STOPPING_PATIENCE}]")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
                break

        fold_accuracies.append(best_val_acc)
        print(f"  Fold {fold + 1} Best Val Accuracy: {best_val_acc:.4f}")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\nCV Results: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Per fold: {[f'{a:.4f}' for a in fold_accuracies]}")
    
    return fold_accuracies


def train_final_model(train_dataset, test_dataset, n_features, device, scenario_name, class_weights=None):
    """
    Train the final model on the full training set and evaluate on test set.
    
    Returns: model, train_metrics, test_metrics, elapsed_time
    """
    print(f"\n{'='*50}")
    print(f"Training Final Model")
    print(f"{'='*50}")

    # Build Focal Loss với class weights (NOC 1-5 → index 0-4)
    if class_weights is not None:
        alpha_tensor = torch.tensor(
            [class_weights.get(i + 1, 1.0) for i in range(5)],
            dtype=torch.float32
        ).to(device)
        print(f"  Focal Loss alpha: {[f'{class_weights.get(i+1, 1.0):.3f}' for i in range(5)]}")
    else:
        alpha_tensor = None
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    model = TAWSEEM_MLP(input_dim=n_features).to(device)
    model.summary()
    
    criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    start_time = time.time()
    
    best_test_acc = 0
    patience_counter = 0
    model_path = os.path.join(RESULTS_DIR, f"{scenario_name}_best_model.pth")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

        # Save best model & early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Test Loss={test_loss:.4f}, Acc={test_acc:.4f} "
                  f"[patience {patience_counter}/{EARLY_STOPPING_PATIENCE}]")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1} (best test acc: {best_test_acc:.4f})")
            break
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Final evaluation on train and test
    _, train_acc, train_preds, train_labels = evaluate(model, train_loader, criterion, device)
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    # Compute detailed metrics
    train_metrics = compute_metrics(train_labels, train_preds)
    test_metrics = compute_metrics(test_labels, test_preds)
    
    print(f"\n--- Training Set Metrics ---")
    print_metrics(train_metrics)
    
    print(f"\n--- Test Set Metrics ---")
    print_metrics(test_metrics)
    
    return model, train_metrics, test_metrics, elapsed_time

