import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import argparse

# Thêm config path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, DATA_PROCESSED_DIR, SCENARIOS
from data_preprocessing import (
    step2_drop_high_alleles, step3_handle_ol_values, step4_handle_missing_values,
    step5_remove_markers, step6_encode_dye, step7_encode_marker, step7b_encode_multiplex,
    step7c_encode_injection_time, step8_create_profile_loci, step10_finalize_features
)
from dataset import _extract_marker_features
from model import TAWSEEM_MLP

def extract_inference_features(df_sorted):
    """
    Extract features directly without downsampling (for inference).
    """
    profile_col = 'Sample File'
    
    height_cols = [f'Height {i}' for i in range(1, 11) if f'Height {i}' in df_sorted.columns]
    ol_cols = [f'OL_ind_{i}' for i in range(1, 11) if f'OL_ind_{i}' in df_sorted.columns]
    missing_cols = [f'Missing_Allele_{i}' for i in range(1, 11) if f'Missing_Allele_{i}' in df_sorted.columns]
    
    has_missing_marker = 'Missing_Marker' in df_sorted.columns
    has_multiplex = 'multiplex' in df_sorted.columns
    has_inj_time = 'injection_time' in df_sorted.columns
    
    flat_profiles = []
    profile_names = []
    
    for sample_file, group in df_sorted.groupby(profile_col, sort=False):
        marker_feats = []
        for _, row in group.iterrows():
            marker_feats.append(_extract_marker_features(row, height_cols, ol_cols, missing_cols))
            
        marker_feats = np.array(marker_feats, dtype=np.float32)
        allele_counts = marker_feats[:, 0]
        max_heights = marker_feats[:, 1]
        ol_counts = marker_feats[:, 9]
        sum_heights = marker_feats[:, 4]
        
        aggregates = np.array([
            np.max(allele_counts),               # MAC
            np.mean(allele_counts),
            np.std(allele_counts),
            np.sum(allele_counts >= 3),          # markers with 3+ alleles
            np.sum(allele_counts >= 5),          # markers with 5+ alleles
            np.sum(ol_counts),                   # total OL
            np.mean(max_heights),
            np.std(max_heights),
            np.sum(allele_counts),               # total peaks
            np.sum(sum_heights),                 # total height signal
            float(group['Missing_Marker'].sum()) if has_missing_marker else 0.0,
            float(group['multiplex'].iloc[0])    if has_multiplex  else -1.0,
            float(group['injection_time'].iloc[0]) if has_inj_time else -1.0,
        ], dtype=np.float32)
        
        flat_profiles.append(np.concatenate([marker_feats.flatten(), aggregates]))
        profile_names.append(sample_file)
        
    return np.array(flat_profiles, dtype=np.float32), profile_names

def run_inference(input_csv, model_path, scaler_path, scenario_name='single', output_csv='predictions.csv'):
    print("=" * 60)
    print(f"TAWSEEM Inference Pipeline - {scenario_name}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"\n1. Loading input data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if 'NOC' not in df.columns:
        df['NOC'] = 1  
    if 'multiplex' not in df.columns:
        df['multiplex'] = 'GF29' 
    if 'injection_time' not in df.columns:
        df['injection_time'] = '25 sec' 
        
    print(f"   Loaded {df['Sample File'].nunique()} profiles, {len(df)} rows.")
    
    # 2. Preprocess exactly like training
    print("\n2. Applying preprocessing pipeline...")
    scenario = SCENARIOS[scenario_name]
    
    df = step2_drop_high_alleles(df)
    df = step3_handle_ol_values(df)
    df = step4_handle_missing_values(df)
    df = step5_remove_markers(df, scenario['markers_to_keep'])
    
    # Force padding missing markers for inference so the feature count is exactly 387
    from data_preprocessing import step5b_pad_missing_markers
    df = step5b_pad_missing_markers(df, scenario['markers_to_keep'])
        
    df = step6_encode_dye(df)
    df = step7_encode_marker(df, scenario['markers_to_keep'])
    
    from config import MULTIPLEX_ENCODING
    df['multiplex'] = df['multiplex'].map(MULTIPLEX_ENCODING).fillna(-1).astype(int)
    
    time_encoding = {'25 sec': 0}
    df['injection_time'] = df['injection_time'].map(time_encoding).fillna(-1).astype(int)
    
    df = step8_create_profile_loci(df)
    df = step10_finalize_features(df)
    
    # Sort exactly like dataset.py
    df_sorted = df.sort_values(['Sample File', 'Marker']).reset_index(drop=True)
    df_sorted = df_sorted.drop_duplicates(subset=['Sample File', 'Marker'], keep='first').reset_index(drop=True)
    
    # 3. Extract profile features
    print("\n3. Extracting profile-level features...")
    X_flat, profile_names = extract_inference_features(df_sorted)
    
    print(f"   Extracted {len(X_flat)} profiles with {X_flat.shape[1]} features each.")
    
    # 4. Scale features
    print(f"\n4. Loading scaler from {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    X_scaled = scaler.transform(X_flat)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # 5. Model Inference
    print(f"\n5. Loading model from {model_path}...")
    model = TAWSEEM_MLP(input_dim=X_flat.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("\n6. Running predictions...")
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1) + 1  # 0-4 mapped back to 1-5 NOC
        
    # 7. Collect Results
    results = []
    for i, profile in enumerate(profile_names):
        res = {
            'Sample File': profile,
            'Predicted NOC': preds[i],
            'Confidence (%)': round(np.max(probs[i]) * 100, 2)
        }
        for cls in range(5):
            res[f'Prob NOC={cls+1}'] = round(probs[i][cls] * 100, 2)
        results.append(res)
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    
    print(f"\nTesting finished! Predictions saved to {output_csv}")
    print(res_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAWSEEM Inference")
    parser.add_argument("--data", default="data/processed/20251011_HID360_PROVEDIt.csv")
    parser.add_argument("--model", default="data/processed/single_best_model.pth")
    parser.add_argument("--scaler", default="data/processed/single_scaler.pkl")
    parser.add_argument("--scenario", default="single")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    
    run_inference(args.data, args.model, args.scaler, args.scenario, args.output)
