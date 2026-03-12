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
        # Ensure targets are within valid range [0, num_classes-1]
        targets = targets.long()
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
        
        # Ensure labels are valid (0-4)
        assert (labels >= 0).all() and (labels < 5).all(), f"Invalid label range: {labels.min()}-{labels.max()}"
        
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
    """Evaluate model. Returns loss, accuracy, all predictions, labels, and probabilities."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * features.size(0)
            
            # Predict
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_samples += features.size(0)
    
    avg_loss = total_loss / total_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    accuracy = (all_preds == all_labels).mean()
    
    return avg_loss, accuracy, all_preds, all_labels, all_probs


def cross_validate(full_data, n_features, device, scenario_name, class_weights=None):
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
    
    X_full, y_full, groups_full, _ = full_data
    fold_accuracies = []
    
    try:
        skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        splits = list(skf.split(X_full, groups_full))
        print(f"  CV Stratified Strategy: Strata (NOC x MUX x INJ)")
    except ValueError:
        skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        splits = list(skf.split(X_full, y_full))
        print(f"  CV Stratified Strategy: NOC (Due to small strata < 5 samples)")
    
    from sklearn.preprocessing import MinMaxScaler
    from src.dataset import DNAProfileDataset
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold + 1}/{NUM_CV_FOLDS} ---")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        # Custom scaler per fold to avoid data leakage
        fold_scaler = MinMaxScaler()
        X_fold_train = fold_scaler.fit_transform(X_full[train_idx])
        X_fold_val   = fold_scaler.transform(X_full[val_idx])
        
        # Build datasets directly from pre-scaled numpy arrays
        train_fold_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_fold_train, dtype=torch.float32),
            torch.tensor(y_full[train_idx] - 1, dtype=torch.long),   # NOC 1-5 → 0-4
        )
        val_fold_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_fold_val, dtype=torch.float32),
            torch.tensor(y_full[val_idx] - 1, dtype=torch.long),
        )
        
        train_loader = DataLoader(train_fold_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_fold_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = TAWSEEM_MLP(input_dim=n_features).to(device)
        criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

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
    Train the final model on the full training set WITHOUT touching test set.
    
    IMPORTANT: Test set is ONLY used for final evaluation AFTER training.
    Early stopping uses a held-out validation split from training data.
    
    Returns: model, train_metrics, test_metrics, elapsed_time
    """
    print(f"\n{'='*50}")
    print(f"Training Final Model")
    print(f"{'='*50}")
    print(f"  ⚠️  Data Split Principle:")
    print(f"      • Train on: 90% data (full training set)")
    print(f"      • Validation: Hold-out from 90% for early stopping")
    print(f"      • Test on: 10% data (NEVER touched during training)")
    
    # Extract data from datasets for splitting
    # NOTE: train_dataset.labels are already 0-4 (from DNAProfileDataset)
    X_train = train_dataset.features.numpy()
    y_train = train_dataset.labels.numpy()  # Already 0-4
    
    # Create a validation split from training data (8% of original = 11% of 90%)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.111, random_state=RANDOM_SEED)
    for fit_idx, val_idx in sss.split(X_train, y_train):  # y_train is already 0-4
        X_train_only = X_train[fit_idx]
        y_train_only = y_train[fit_idx]
        X_val_only = X_train[val_idx]
        y_val_only = y_train[val_idx]
    
    # Normalize training-only data
    from sklearn.preprocessing import MinMaxScaler
    final_scaler = MinMaxScaler()
    X_train_only_scaled = final_scaler.fit_transform(X_train_only)
    X_val_only_scaled = final_scaler.transform(X_val_only)
    
    # Create dataloaders
    train_only_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train_only_scaled, dtype=torch.float32),
        torch.tensor(y_train_only, dtype=torch.long),  # Already 0-4
    )
    val_only_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val_only_scaled, dtype=torch.float32),
        torch.tensor(y_val_only, dtype=torch.long),  # Already 0-4
    )
    
    train_loader = DataLoader(train_only_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_only_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train only: {len(X_train_only)}, Val (hold-out): {len(X_val_only)}")

    # Build Focal Loss with class weights (NOC 1-5 → index 0-4)
    if class_weights is not None:
        alpha_tensor = torch.tensor(
            [class_weights.get(i + 1, 1.0) for i in range(5)],
            dtype=torch.float32
        ).to(device)
        print(f"  Focal Loss alpha: {[f'{class_weights.get(i+1, 1.0):.3f}' for i in range(5)]}")
    else:
        alpha_tensor = None
    
    # Create model
    model = TAWSEEM_MLP(input_dim=n_features).to(device)
    model.summary()
    
    criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    start_time = time.time()
    
    best_val_acc = 0
    patience_counter = 0
    model_path = os.path.join(RESULTS_DIR, f"{scenario_name}_best_model.pth")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n  Early stopping based on VALIDATION set (NOT test set)")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        # Early stopping based on validation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f} "
                  f"[patience {patience_counter}/{EARLY_STOPPING_PATIENCE}]")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
            break
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds")
    
    # Load best model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # === FINAL EVALUATION (ONLY NOW, after training) ===
    print(f"\n{'='*50}")
    print(f"Final Evaluation on Test Set (10% - NEVER SEEN)")
    print(f"{'='*50}")
    
    # Scale test set using the same scaler from training
    X_test = test_dataset.features.numpy()
    y_test = test_dataset.labels.numpy()  # Already 0-4
    X_test_scaled = final_scaler.transform(X_test)
    
    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),  # Already 0-4
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Also evaluate on full training set for reference
    X_train_scaled = final_scaler.transform(X_train)
    train_full_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),  # Already 0-4
    )
    train_full_loader = DataLoader(train_full_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    _, train_acc, train_preds, train_labels, train_probs = evaluate(model, train_full_loader, criterion, device)
    _, test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
    
    # Compute detailed metrics
    train_metrics = compute_metrics(train_labels, train_preds)
    test_metrics = compute_metrics(test_labels, test_preds)
    
    train_metrics['probs'] = train_probs
    test_metrics['probs']  = test_probs
    
    print(f"\n--- Training Set (90%) Metrics ---")
    print_metrics(train_metrics)
    
    print(f"\n--- Test Set (10%) Metrics ---")
    print_metrics(test_metrics)
    
    return model, train_metrics, test_metrics, elapsed_time

