"""
TAWSEEM PyTorch Dataset
Custom Dataset class with MinMaxScaler normalization.
Supports both flat (MLP/XGBoost) and 2D (CNN) profile-level data.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class DNAProfileDataset(Dataset):
    """
    PyTorch Dataset for DNA mixture profiles.
    Features are normalized to [0, 1] using MinMaxScaler.
    """
    
    def __init__(self, features, labels, scaler=None, fit_scaler=True):
        if scaler is None:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = scaler
        
        if fit_scaler:
            self.features = self.scaler.fit_transform(features)
        else:
            self.features = self.scaler.transform(features)
        
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(labels - 1)  # NOC 1-5 → 0-4
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    @property
    def n_features(self):
        return self.features.shape[1]


class DNAProfileCNNDataset(Dataset):
    """
    PyTorch Dataset for 1D CNN — returns (n_markers, n_features_per_marker) tensors.
    Each marker's features are scaled independently.
    """
    
    def __init__(self, features_2d, labels, scaler=None, fit_scaler=True):
        """
        Args:
            features_2d: (N, n_markers, n_features_per_marker)
            labels: (N,) with NOC 1-5
        """
        N, M, F = features_2d.shape
        
        # Scale each feature across all samples and markers
        flat = features_2d.reshape(N * M, F)
        if scaler is None:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = scaler
        
        if fit_scaler:
            flat_scaled = self.scaler.fit_transform(flat)
        else:
            flat_scaled = self.scaler.transform(flat)
        
        self.features = torch.FloatTensor(flat_scaled.reshape(N, M, F))
        self.labels = torch.LongTensor(labels - 1)
        self.n_markers = M
        self.n_features_per_marker = F
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Return (n_features_per_marker, n_markers) for Conv1d — channels first
        return self.features[idx].T, self.labels[idx]


def prepare_datasets(df, train_ratio=0.7, random_seed=42):
    """Split DataFrame into train/test datasets at profile level (legacy)."""
    np.random.seed(random_seed)
    profile_col = 'Sample File'
    profiles = df.groupby(profile_col)['NOC'].first()
    
    train_profiles, test_profiles = [], []
    for noc in sorted(profiles.unique()):
        noc_profiles = profiles[profiles == noc].index.tolist()
        np.random.shuffle(noc_profiles)
        n_train = int(len(noc_profiles) * train_ratio)
        train_profiles.extend(noc_profiles[:n_train])
        test_profiles.extend(noc_profiles[n_train:])
    
    train_df = df[df[profile_col].isin(train_profiles)]
    test_df = df[df[profile_col].isin(test_profiles)]
    
    unique_train_profiles = train_df[profile_col].unique()
    profile_id_map = {name: idx for idx, name in enumerate(unique_train_profiles)}
    train_profile_ids = train_df[profile_col].map(profile_id_map).values.astype(np.int64)
    
    feature_cols = [c for c in df.columns if c not in ('NOC', 'Sample File')]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['NOC'].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['NOC'].values.astype(np.int64)
    
    train_dataset = DNAProfileDataset(X_train, y_train, fit_scaler=True)
    test_dataset = DNAProfileDataset(X_test, y_test, scaler=train_dataset.scaler, fit_scaler=False)
    
    print(f"  Train: {len(train_profiles)} profiles, {len(train_dataset)} rows")
    print(f"  Test:  {len(test_profiles)} profiles, {len(test_dataset)} rows")
    return train_dataset, test_dataset, train_dataset.scaler, train_profile_ids


def _extract_marker_features(row, height_cols, ol_cols, missing_cols):
    """Extract rich features from a single marker row. Returns 17 features."""
    heights = row[height_cols].values.astype(float)
    ol_flags = row[ol_cols].values.astype(float)
    missing_flags = row[missing_cols].values.astype(float)
    
    n_alleles = int((missing_flags == 0).sum())
    n_missing = int(missing_flags.sum())
    n_ol = int(ol_flags.sum())
    
    valid_heights = heights[missing_flags == 0]
    
    if len(valid_heights) > 0:
        sorted_h = np.sort(valid_heights)[::-1]
        h1 = sorted_h[0]
        h2 = sorted_h[1] if len(sorted_h) > 1 else 0
        h3 = sorted_h[2] if len(sorted_h) > 2 else 0
        sum_h = np.sum(valid_heights)
        mean_h = np.mean(valid_heights)
        std_h = np.std(valid_heights) if len(valid_heights) > 1 else 0
        h_ratio = h1 / (h2 + 1e-6) if h2 > 0 else 0
        h_range = h1 - np.min(valid_heights)
        
        # Stutter ratio (smallest / largest peak)
        stutter_ratio = np.min(valid_heights) / (h1 + 1e-6)
        
        # SNR (Top 2 peaks / Rest of the peaks)
        top2_h = h1 + h2
        rest_h = sum_h - top2_h
        snr_top2 = top2_h / (rest_h + 1e-6)
        
        # Log transformations for stability
        log1p_h1 = np.log1p(h1)
        log1p_sum_h = np.log1p(sum_h)
    else:
        h1 = h2 = h3 = sum_h = mean_h = std_h = h_ratio = h_range = 0
        stutter_ratio = snr_top2 = log1p_h1 = log1p_sum_h = 0
    
    # Feature 16: missing_marker flag (1 = padded locus, kit does not measure this)
    missing_marker = float(row.get('Missing_Marker', 0))
    
    # Feature 17: marker identity index (0-based; tells model which locus this is)
    marker_index = float(row.get('Marker', -1))
    
    # 17 features per marker
    return [
        n_alleles, h1, h2, h3, sum_h, mean_h, std_h, h_ratio, h_range, n_ol, n_missing,
        stutter_ratio, snr_top2, log1p_h1, log1p_sum_h,
        missing_marker,   # feature 16
        marker_index,     # feature 17
    ]


def prepare_profile_datasets(df, train_ratio=0.9, random_seed=42):
    """
    Convert per-row data to per-PROFILE data with rich feature engineering.
    
    Returns:
        train_dataset, test_dataset, scaler, train_profile_ids,
        (X_train_2d, X_test_2d, y_train, y_test, class_weights),
        (groups_train, groups_test),
        (X_flat, y, strata_labels, X_matrix) # full balanced data
    """
    np.random.seed(random_seed)
    profile_col = 'Sample File'
    
    # Sort and deduplicate
    df_sorted = df.sort_values([profile_col, 'Marker']).reset_index(drop=True)
    before = len(df_sorted)
    df_sorted = df_sorted.drop_duplicates(subset=[profile_col, 'Marker'], keep='first').reset_index(drop=True)
    if len(df_sorted) < before:
        print(f"  Removed {before - len(df_sorted)} duplicate rows")
    
    # Keep profiles with most common marker count
    mpp = df_sorted.groupby(profile_col).size()
    n_markers = mpp.mode().iloc[0]
    valid = mpp[mpp == n_markers].index
    if len(valid) < len(mpp):
        print(f"  Removed {len(mpp) - len(valid)} profiles with inconsistent markers")
        df_sorted = df_sorted[df_sorted[profile_col].isin(valid)].reset_index(drop=True)
    
    # Column groups
    height_cols = [f'Height {i}' for i in range(1, 11) if f'Height {i}' in df.columns]
    ol_cols = [f'OL_ind_{i}' for i in range(1, 11) if f'OL_ind_{i}' in df.columns]
    missing_cols = [f'Missing_Allele_{i}' for i in range(1, 11) if f'Missing_Allele_{i}' in df.columns]
    
    N_PER_MARKER = 17   # 15 peak features + missing_marker flag + marker_index
    N_AGGREGATE = 13    # 10 original + n_missing_markers + multiplex_id + injection_time_id
    
    flat_profiles = []
    matrix_profiles = []
    labels = []
    
    has_missing_marker = 'Missing_Marker' in df_sorted.columns
    has_multiplex = 'multiplex' in df_sorted.columns
    has_inj_time = 'injection_time' in df_sorted.columns

    for sample_file, group in df_sorted.groupby(profile_col, sort=False):
        marker_feats = []
        for _, row in group.iterrows():
            marker_feats.append(_extract_marker_features(row, height_cols, ol_cols, missing_cols))
        
        marker_feats = np.array(marker_feats, dtype=np.float32)
        allele_counts = marker_feats[:, 0]
        max_heights = marker_feats[:, 1]
        ol_counts = marker_feats[:, 9]
        sum_heights = marker_feats[:, 4]
        
        # Profile aggregates
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
            # NEW: context features
            float(group['Missing_Marker'].sum()) if has_missing_marker else 0.0,  # n_missing_markers
            float(group['multiplex'].iloc[0])    if has_multiplex  else -1.0,     # kit id
            float(group['injection_time'].iloc[0]) if has_inj_time else -1.0,    # injection_time id
        ], dtype=np.float32)
        
        flat_profiles.append(np.concatenate([marker_feats.flatten(), aggregates]))
        matrix_profiles.append(marker_feats)
        labels.append(group['NOC'].iloc[0])
    
    X_flat = np.array(flat_profiles, dtype=np.float32)
    X_matrix = np.array(matrix_profiles, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    print(f"  Profile-level: {len(X_flat)} profiles")
    print(f"    Flat: {X_flat.shape[1]} features ({n_markers}x{N_PER_MARKER} + {N_AGGREGATE})")
    print(f"    CNN:  {X_matrix.shape} (profiles x markers x features)")
    
    # --- 1. Create rich strata for splitting (NOC x injection_time x multiplex) ---
    profile_meta = df_sorted.groupby(profile_col, sort=False).agg(
        NOC=('NOC', 'first'),
        **({'inj': ('injection_time', 'first')} if has_inj_time else {}),
        **({'mux': ('multiplex', 'first')} if has_multiplex else {}),
    ).reset_index()
    
    # Align row order with X_flat
    profile_order = df_sorted.groupby(profile_col, sort=False).first().index
    profile_meta = profile_meta.set_index(profile_col).loc[profile_order].reset_index()

    strata_cols = ['NOC']
    if has_inj_time: strata_cols.append('inj')
    if has_multiplex: strata_cols.append('mux')
    
    profile_meta['strata'] = profile_meta[strata_cols].astype(str).agg('_'.join, axis=1)

    # --- 2. Dynamic Downsampling on FULL dataset (Class 1 cap = 5x mean of Class 2-5) ---
    counts = profile_meta['NOC'].value_counts()
    other_counts = [counts.get(n, 0) for n in [2, 3, 4, 5]]
    mean_other = sum(other_counts) / 4.0 if sum(other_counts) > 0 else 0
    max_class1 = max(1, int(mean_other * 5))
    
    idx_class1 = profile_meta[profile_meta['NOC'] == 1].index.tolist()
    idx_others = profile_meta[profile_meta['NOC'] != 1].index.tolist()
    
    if len(idx_class1) > max_class1:
        keep_class1 = []
        c1_strata = profile_meta.loc[idx_class1, 'strata'].value_counts(normalize=True)
        for sv, prop in c1_strata.items():
            target_n = int(round(prop * max_class1))
            sv_idx = profile_meta[(profile_meta['NOC'] == 1) & (profile_meta['strata'] == sv)].index.tolist()
            np.random.shuffle(sv_idx)
            keep_class1.extend(sv_idx[:target_n])
            
        if len(keep_class1) > max_class1:
            np.random.shuffle(keep_class1)
            keep_class1 = keep_class1[:max_class1]
        elif len(keep_class1) < max_class1:
            rem = list(set(idx_class1) - set(keep_class1))
            np.random.shuffle(rem)
            keep_class1.extend(rem[:max_class1 - len(keep_class1)])
            
        balanced_idx = keep_class1 + idx_others
    else:
        balanced_idx = profile_meta.index.tolist()
        
    np.random.shuffle(balanced_idx)
    balanced_idx = np.array(balanced_idx)
    
    # Complete filtered Base Dataset
    X_flat_b = X_flat[balanced_idx]
    y_b = y[balanced_idx]
    groups_b = profile_meta['strata'].values[balanced_idx]
    X_matrix_b = X_matrix[balanced_idx]
    
    print(f"\n[Dataset] After balancing class 1 on FULL dataset:")
    print(f"  Total balanced profiles: {len(balanced_idx)}")
    
    # --- 3. Stratified Split 80/20 on the BALANCED dataset ---
    balanced_meta = profile_meta.iloc[balanced_idx].copy().reset_index(drop=True)
    
    train_idx_rel, test_idx_rel = [], []
    for sv in sorted(balanced_meta['strata'].unique()):
        idx = balanced_meta[balanced_meta['strata'] == sv].index.tolist()
        np.random.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        train_idx_rel.extend(idx[:n_train])
        test_idx_rel.extend(idx[n_train:])
        
    np.random.shuffle(train_idx_rel)
    np.random.shuffle(test_idx_rel)

    X_train_flat, y_train = X_flat_b[train_idx_rel], y_b[train_idx_rel]
    X_test_flat, y_test = X_flat_b[test_idx_rel], y_b[test_idx_rel]
    X_train_2d, X_test_2d = X_matrix_b[train_idx_rel], X_matrix_b[test_idx_rel]
    
    train_profile_ids = np.arange(len(train_idx_rel))
    
    # Flat datasets
    train_dataset = DNAProfileDataset(X_train_flat, y_train, fit_scaler=True)
    test_dataset = DNAProfileDataset(X_test_flat, y_test, scaler=train_dataset.scaler, fit_scaler=False)
    
    # Class weights for imbalanced data (by NOC only, not injection_time)
    from collections import Counter
    counts = Counter(y_train.tolist())
    total = len(y_train)
    n_classes = len(counts)
    class_weights = {c: total / (n_classes * n) for c, n in counts.items()}
    
    groups_train = groups_b[train_idx_rel]
    groups_test  = groups_b[test_idx_rel]

    print(f"  Train: {len(train_idx_rel)}, Test: {len(test_idx_rel)}")
    print(f"  Class distribution train: {dict(sorted(counts.items()))}")
    
    return train_dataset, test_dataset, train_dataset.scaler, train_profile_ids, \
           (X_train_2d, X_test_2d, y_train, y_test, class_weights), \
           (groups_train, groups_test), \
           (X_flat_b, y_b, groups_b, X_matrix_b)
