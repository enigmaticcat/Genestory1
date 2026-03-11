"""
TAWSEEM: CNN Training Script
Script to train and evaluate the 1D-CNN model.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_PROCESSED_DIR, RESULTS_DIR, TRAIN_RATIO, RANDOM_SEED,
    BATCH_SIZE, EPOCHS, LEARNING_RATE
)
from src.data_preprocessing import preprocess_scenario
from src.dataset import prepare_profile_datasets, DNAProfileCNNDataset
from src.model import TAWSEEM_CNN
from src.evaluate import compute_metrics, print_metrics, generate_all_plots

def run_cnn_pipeline(scenario_name):
    print(f"\n{'#'*60}")
    print(f"# TAWSEEM CNN — Scenario: {scenario_name}")
    print(f"{'#'*60}")
    
    # --- Step 1: Data ---
    processed_path = os.path.join(DATA_PROCESSED_DIR, f"{scenario_name}_processed.csv")
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
    else:
        df = preprocess_scenario(scenario_name)
        
    _, _, _, _, cnn_data = prepare_profile_datasets(
        df, train_ratio=TRAIN_RATIO, random_seed=RANDOM_SEED
    )
    X_train_2d, X_test_2d, y_train, y_test, class_weights = cnn_data
    
    # Create CNN Datasets
    # Note: prepare_profile_datasets already handles scaling for 2D data internally if we used its logic,
    # but here we use the raw 2D arrays and the DNAProfileCNNDataset wrapper.
    train_dataset = DNAProfileCNNDataset(X_train_2d, y_train, fit_scaler=True)
    test_dataset = DNAProfileCNNDataset(X_test_2d, y_test, scaler=train_dataset.scaler, fit_scaler=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Step 2: Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TAWSEEM_CNN(
        n_features_per_marker=train_dataset.n_features_per_marker,
        n_markers=train_dataset.n_markers
    ).to(device)
    model.summary()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Step 3: Training ---
    print("\nTraining CNN...")
    start_time = time.time()
    best_acc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            acc = (np.array(all_preds) == np.array(all_labels)).mean()
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Test Acc: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                os.makedirs(RESULTS_DIR, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{scenario_name}_cnn_best.pth"))
                
    elapsed = time.time() - start_time
    print(f"\nCNN Training completed in {elapsed:.1f}s. Best Accuracy: {best_acc:.4f}")
    
    # Final Evaluation
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"{scenario_name}_cnn_best.pth")))
    model.eval()
    
    # Helper to get all metrics
    def get_metrics(loader):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return compute_metrics(np.array(all_labels), np.array(all_preds))

    train_metrics = get_metrics(train_loader)
    test_metrics = get_metrics(test_loader)
    
    print("\nFinal CNN Metrics (Test):")
    print_metrics(test_metrics)
    
    # Generate Plots
    generate_all_plots(train_metrics, test_metrics, f"{scenario_name}_cnn")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='single')
    args = parser.parse_args()
    run_cnn_pipeline(args.scenario)
