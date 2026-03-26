import os
import sys
import pickle
import numpy as np

# Thêm config path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, DATA_PROCESSED_DIR
from data_preprocessing import preprocess_scenario
from dataset import prepare_profile_datasets

def export_scaler():
    print("=" * 50)
    print("Exporting MinMaxScaler for Single Scenario")
    print("=" * 50)
    
    scenario_name = 'single'
    
    # 1. Pipeline returns df
    print(f"1. Preprocessing for scenario: {scenario_name}...")
    df_processed = preprocess_scenario(scenario_name)
    
    # 2. Extract dataset scaler (just like in train.py / dataset.py)
    print("2. Extracting features and scaler via prepare_profile_datasets...")
    train_dataset, test_dataset, final_scaler, train_profile_ids, _, _, _ = prepare_profile_datasets(
        df_processed, train_ratio=0.9, random_seed=42
    )
    
    # 3. Save it
    out_path = os.path.join(DATA_PROCESSED_DIR, f"{scenario_name}_scaler.pkl")
    print(f"3. Saving scaler to {out_path}...")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(final_scaler, f)
        
    print("Done!")

if __name__ == "__main__":
    export_scaler()
