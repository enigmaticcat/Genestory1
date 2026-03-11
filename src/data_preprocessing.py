import os
import re
import csv
import pandas as pd
import numpy as np
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, MULTIPLEX_FOLDERS,
    MARKERS_TO_REMOVE, SCENARIOS, DYE_ENCODING, MULTIPLEX_ENCODING,
    MAX_ALLELES, RANDOM_SEED,
)


def step1_load_csvs_with_noc(multiplex_name, folder_name, injection_times=None):
    """
    Step 1: Read CSV files and assign NOC labels.
    
    NOC is determined by:
    - 1-Person folders: NOC = 1
    - 2-5 Person folders: count person IDs in sample name
      e.g., "RD14-0003-31_32-..." → 2 persons → NOC = 2
    - IDPlus29 has separate 2-Person/, 3-Person/ etc. folders
    
    Returns: pd.DataFrame with all rows and a 'NOC' column
    """
    multiplex_path = os.path.join(DATA_RAW_DIR, folder_name)
    all_dfs = []
    
    print(f"  Loading {multiplex_name} from {folder_name}...")
    
    # --- Load 1-Person data ---
    one_person_dir = os.path.join(multiplex_path, "1-Person")
    if os.path.isdir(one_person_dir):
        for inj_dir in sorted(os.listdir(one_person_dir)):
            inj_path = os.path.join(one_person_dir, inj_dir)
            if not os.path.isdir(inj_path):
                continue
            if injection_times and inj_dir not in injection_times:
                continue
            for f in os.listdir(inj_path):
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(inj_path, f), low_memory=False).copy()
                    df['NOC'] = 1
                    df['multiplex'] = multiplex_name
                    df['injection_time'] = inj_dir
                    all_dfs.append(df)
                    print(f"    1-Person/{inj_dir}: {df['Sample File'].nunique()} profiles, {len(df)} rows")
    
    # --- Load 2-5 Person data ---
    # Check for combined "2-5-Persons" folder or separate "2-Person", "3-Person" etc.
    combined_dir = os.path.join(multiplex_path, "2-5-Persons")
    if os.path.isdir(combined_dir):
        for inj_dir in sorted(os.listdir(combined_dir)):
            inj_path = os.path.join(combined_dir, inj_dir)
            if not os.path.isdir(inj_path):
                continue
            if injection_times and inj_dir not in injection_times:
                continue
            for f in os.listdir(inj_path):
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(inj_path, f), low_memory=False).copy()
                    # Extract NOC from sample filename
                    df['NOC'] = df['Sample File'].apply(_extract_noc_from_filename)
                    df['multiplex'] = multiplex_name
                    df['injection_time'] = inj_dir
                    all_dfs.append(df)
                    noc_dist = df.groupby('NOC')['Sample File'].nunique().to_dict()
                    print(f"    2-5-Persons/{inj_dir}: {df['Sample File'].nunique()} profiles, NOC dist: {noc_dist}")
    else:
        # Separate folders (IDPlus29 style)
        for noc in range(2, 6):
            noc_dir = os.path.join(multiplex_path, f"{noc}-Person")
            if not os.path.isdir(noc_dir):
                continue
            for inj_dir in sorted(os.listdir(noc_dir)):
                inj_path = os.path.join(noc_dir, inj_dir)
                if not os.path.isdir(inj_path):
                    continue
                if injection_times and inj_dir not in injection_times:
                    continue
                for f in os.listdir(inj_path):
                    if f.endswith('.csv'):
                        df = pd.read_csv(os.path.join(inj_path, f), low_memory=False).copy()
                        df['NOC'] = noc
                        df['multiplex'] = multiplex_name
                        df['injection_time'] = inj_dir
                        all_dfs.append(df)
                        print(f"    {noc}-Person/{inj_dir}: {df['Sample File'].nunique()} profiles, {len(df)} rows")
    
    if not all_dfs:
        raise ValueError(f"No CSV files found for {multiplex_name}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {combined['Sample File'].nunique()} profiles, {len(combined)} rows")
    return combined


def _extract_noc_from_filename(sample_name):
    """Extract number of contributors from sample filename by counting person IDs."""
    # Pattern: RD14-0003-31_32-... or RD12-0002-XX_YY-...
    match = re.search(r'RD\d+-\d+-([0-9]+(?:_[0-9]+)*)-', sample_name)
    if match:
        person_ids = match.group(1).split('_')
        return len(person_ids)
    # Fallback: try to find M<N>c pattern (e.g., M2c, M3e)
    match2 = re.search(r'-M(\d)', sample_name)
    if match2:
        return int(match2.group(1))
    return -1  # Unknown


def step2_drop_high_alleles(df):
    """
    Step 2: Drop Allele/Size/Height columns from position 10 to 100.
    
    Reason: From position 10 onwards, data is >77% empty.
    Keep only positions 1-9 (MAX_ALLELES).
    """
    cols_to_keep = ['Sample File', 'Marker', 'Dye', 'NOC', 'multiplex', 'injection_time']
    for i in range(1, MAX_ALLELES + 1):
        cols_to_keep.extend([f'Allele {i}', f'Size {i}', f'Height {i}'])
    
    # Only keep columns that exist
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep].copy()
    
    print(f"  Step 2: Kept {MAX_ALLELES} allele positions → {len(df.columns)} columns")
    return df


def step3_handle_ol_values(df):
    """
    Step 3: Handle Out-of-Ladder (OL) values in Allele columns.
    
    - Create binary OL indicator columns: OL_ind_1 ... OL_ind_10
    - Replace "OL" with 0 in allele columns
    """
    for i in range(1, MAX_ALLELES + 1):
        col = f'Allele {i}'
        if col in df.columns:
            # Create OL indicator
            df[f'OL_ind_{i}'] = (df[col] == 'OL').astype(int)
            # Replace OL with 0
            df[col] = df[col].replace('OL', '0')
    
    ol_count = sum(df[f'OL_ind_{i}'].sum() for i in range(1, MAX_ALLELES + 1))
    print(f"  Step 3: Found {ol_count} OL values, created {MAX_ALLELES} indicator columns")
    return df


def step4_handle_missing_values(df):
    """
    Step 4: Handle missing values (empty cells).
    
    Paper Section 4.2.2: "we created a missing indicator column for each
    column that contains missing values."
    
    - Create binary Missing indicator columns for ALL numeric columns:
      Missing_Allele_1..10, Missing_Size_1..10, Missing_Height_1..10 (30 total)
    - Fill empty cells with the mean of their column
    """
    numeric_cols = []
    for i in range(1, MAX_ALLELES + 1):
        for prefix in ['Allele', 'Size', 'Height']:
            col = f'{prefix} {i}'
            if col in df.columns:
                numeric_cols.append(col)
    
    # Create missing indicators for ALL numeric columns (Allele, Size, Height)
    n_missing_indicators = 0
    for i in range(1, MAX_ALLELES + 1):
        for prefix in ['Allele', 'Size', 'Height']:
            col = f'{prefix} {i}'
            if col in df.columns:
                # For Allele: missing = NaN/empty AND not OL
                if prefix == 'Allele':
                    ol_indicator = df[f'OL_ind_{i}'] if f'OL_ind_{i}' in df.columns else 0
                    df[f'Missing_{prefix}_{i}'] = (
                        (df[col].isna() | (df[col] == '')) & (ol_indicator == 0)
                    ).astype(int)
                else:
                    # For Size/Height: missing = NaN/empty
                    df[f'Missing_{prefix}_{i}'] = (
                        df[col].isna() | (df[col] == '')
                    ).astype(int)
                n_missing_indicators += 1
    
    # Convert to numeric and fill with mean
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    total_missing = df[numeric_cols].isna().sum().sum()
    
    # Fill missing with 0 (since they represent absence of peaks)
    for col in numeric_cols:
        df[col] = df[col].fillna(0)
    
    print(f"  Step 4: Filled {total_missing} missing values with 0, created {n_missing_indicators} missing indicators")
    return df


def step5_remove_markers(df, markers_to_keep):
    """
    Step 5: Remove AMEL, Yindel, DYS391 markers and Sample File column.
    
    Also filter to only keep markers that exist in the target scenario.
    """
    before = len(df)
    df = df[df['Marker'].isin(markers_to_keep)].copy()
    after = len(df)
    
    print(f"  Step 5: Removed {before - after} rows with excluded markers ({before} -> {after})")
    print(f"          Keeping {len(markers_to_keep)} markers: {markers_to_keep}")
    return df


def step5b_pad_missing_markers(df, all_markers):
    """
    Step 5b: Pad missing markers for each profile with zero rows.

    When a GF29 profile lacks 'Penta D' (PP16HS-only marker), adds a row
    with Marker='Penta D', all allele/size/height = 0, Missing_Marker = 1.

    After this step every profile has exactly len(all_markers) rows.
    Missing_Marker = 0: real marker detected by the kit
    Missing_Marker = 1: padded (kit does not measure this locus)
    """
    df = df.copy()
    df['Missing_Marker'] = 0

    all_markers_set = set(all_markers)
    new_rows = []

    # Columns to zero-out in padded rows
    zero_cols = []
    for i in range(1, MAX_ALLELES + 1):
        for prefix in ['Allele', 'Size', 'Height']:
            col = f'{prefix} {i}'
            if col in df.columns:
                zero_cols.append(col)
        for col in [f'OL_ind_{i}',
                    f'Missing_Allele_{i}', f'Missing_Size_{i}', f'Missing_Height_{i}']:
            if col in df.columns:
                zero_cols.append(col)

    for sample_file, group in df.groupby('Sample File', sort=False):
        existing = set(group['Marker'].unique())
        missing = all_markers_set - existing
        if not missing:
            continue

        # Use first row as template (carries Sample File, NOC, multiplex, injection_time)
        template = group.iloc[0].copy()
        for marker in missing:
            pad = template.copy()
            pad['Marker'] = marker
            pad['Dye'] = -1   # no dye for padded marker
            for col in zero_cols:
                pad[col] = 0
            pad['Missing_Marker'] = 1
            new_rows.append(pad)

    if new_rows:
        pad_df = pd.DataFrame(new_rows)
        df = pd.concat([df, pad_df], ignore_index=True)

    df = df.sort_values(['Sample File', 'Marker']).reset_index(drop=True)
    n_profiles = df['Sample File'].nunique()
    avg_pad = len(new_rows) / n_profiles if n_profiles else 0
    print(f"  Step 5b: Added {len(new_rows)} padded rows "
          f"({avg_pad:.1f} avg/profile). All profiles now have {len(all_markers)} markers.")
    return df


def step6_encode_dye(df):
    """
    Step 6: Encode Dye column (categorical → numerical).
    B→0, G→1, P→2, R→3, Y→4
    """
    df['Dye'] = df['Dye'].map(DYE_ENCODING).fillna(-1).astype(int)
    print(f"  Step 6: Encoded Dye values: {DYE_ENCODING}")
    return df


def step7_encode_marker(df, markers_to_keep):
    """
    Step 7: Encode Marker name (categorical → numerical).
    """
    marker_encoding = {m: i for i, m in enumerate(sorted(markers_to_keep))}
    df['Marker'] = df['Marker'].map(marker_encoding)
    print(f"  Step 7: Encoded {len(marker_encoding)} markers")
    return df


def step7b_encode_multiplex(df):
    """
    Step 7b: Encode multiplex column (categorical → numerical).
    IDPlus28→0, IDPlus29→1, GF29→2, PP16HS32→3
    """
    df['multiplex'] = df['multiplex'].map(MULTIPLEX_ENCODING).fillna(-1).astype(int)
    print(f"  Step 7b: Encoded multiplex values: {MULTIPLEX_ENCODING}")
    return df


def step7c_encode_injection_time(df):
    """
    Step 7c: Encode injection_time column (categorical → numerical).
    Automatically assigns integer labels based on sorted unique values.
    """
    unique_times = sorted(df['injection_time'].unique())
    time_encoding = {t: i for i, t in enumerate(unique_times)}
    df['injection_time'] = df['injection_time'].map(time_encoding).fillna(-1).astype(int)
    print(f"  Step 7c: Encoded injection_time values: {time_encoding}")
    return df


def step8_create_profile_loci(df):
    """
    Step 8: Create Profile Loci feature.
    
    Assigns the index of each locus (marker) within its profile.
    For a profile with 22 markers, values go from 0 to 21.
    This tells the model "this is the Nth marker in the profile."
    
    Paper Section 4.2.6: "We created this feature because of its
    importance in specifying which markers are together."
    """
    df = df.copy()
    df['profile_loci'] = df.groupby('Sample File').cumcount()
    
    n_profiles = df['Sample File'].nunique()
    max_loci = df['profile_loci'].max() + 1
    print(f"  Step 8: Created profile_loci (0-{max_loci-1}) for {n_profiles} unique profiles")
    return df


    # Remove Step 9: Dataset balancing will be done in dataset.py during Train/Test Split
    # to guarantee proportional downsampling on both Train and Test set independently.




def step10_finalize_features(df):
    """
    Step 10: Select final feature columns and label.
    
    Removed only truly problematic features:
      - profile_loci:   redundant with Marker / causes leakage if unique ID
      - multiplex:      removed if constant (single scenario)
      - injection_time: removed if constant (single scenario)
    
    Kept features (~72 for single scenario):
      - Allele/Size/Height 1-10:         30  (peak data)
      - OL_ind 1-10:                     10  (out-of-ladder indicators)
      - Missing_Allele/Size/Height 1-10: 30  (missing value indicators)
      - Dye, Marker:                      2  (categorical)
      - multiplex, injection_time:       0-2  (only if non-constant)
    
    Also keeps 'Sample File' for profile-level splitting (not a feature).
    """
    feature_cols = []
    
    # Values: Allele, Size, Height 1-10
    for i in range(1, MAX_ALLELES + 1):
        feature_cols.extend([f'Allele {i}', f'Size {i}', f'Height {i}'])
    
    # OL indicators
    for i in range(1, MAX_ALLELES + 1):
        feature_cols.append(f'OL_ind_{i}')
    
    # Missing indicators (Allele + Size + Height)
    for i in range(1, MAX_ALLELES + 1):
        for prefix in ['Allele', 'Size', 'Height']:
            feature_cols.append(f'Missing_{prefix}_{i}')
    
    # Categorical: always keep Dye and Marker
    feature_cols.extend(['Dye', 'Marker'])
    
    # Only keep multiplex/injection_time if they have >1 unique value
    for cat_col in ['multiplex', 'injection_time']:
        if cat_col in df.columns and df[cat_col].nunique() > 1:
            feature_cols.append(cat_col)
    
    # Missing_Marker flag (only present in four_union scenario)
    if 'Missing_Marker' in df.columns:
        feature_cols.append('Missing_Marker')
    
    # NOTE: profile_loci intentionally excluded (redundant with Marker)
    
    # Only keep existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Keep Sample File for profile-level splitting (not a model feature)
    keep_cols = feature_cols + ['NOC']
    if 'Sample File' in df.columns:
        keep_cols.append('Sample File')
    
    result = df[keep_cols].copy()
    
    n_values = sum(1 for c in feature_cols if c.startswith(('Allele ', 'Size ', 'Height ')))
    n_ol = sum(1 for c in feature_cols if c.startswith('OL_ind'))
    n_missing = sum(1 for c in feature_cols if c.startswith('Missing_'))
    n_cat = len(feature_cols) - n_values - n_ol - n_missing
    
    print(f"  Step 10: Final dataset: {result.shape[0]} rows × {len(feature_cols)} features + 1 label")
    print(f"           Feature groups: {n_values} values + {n_ol} OL + {n_missing} missing + {n_cat} categorical = {len(feature_cols)}")
    return result


def preprocess_scenario(scenario_name):
    """
    Run the full 10-step preprocessing pipeline for a given scenario.
    
    Args:
        scenario_name: "single", "three", or "four"
    
    Returns:
        pd.DataFrame: processed dataset ready for training
    """
    scenario = SCENARIOS[scenario_name]
    print(f"\n{'='*60}")
    print(f"Preprocessing Scenario: {scenario['name']}")
    print(f"{'='*60}")
    
    # Step 1: Load all CSVs
    print("\n[Step 1] Loading CSV files and assigning NOC labels...")
    all_dfs = []
    for mpx_name in scenario['multiplexes']:
        folder_name = MULTIPLEX_FOLDERS[mpx_name]
        df = step1_load_csvs_with_noc(
            mpx_name, folder_name, 
            injection_times=scenario.get('injection_times')
        )
        all_dfs.append(df)
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Combined: {df['Sample File'].nunique()} profiles, {len(df)} rows")
    
    # Step 2: Drop high alleles
    print("\n[Step 2] Dropping Allele 10-100...")
    df = step2_drop_high_alleles(df)
    
    # Step 3: Handle OL values
    print("\n[Step 3] Handling OL values...")
    df = step3_handle_ol_values(df)
    
    # Step 4: Handle missing values
    print("\n[Step 4] Handling missing values...")
    df = step4_handle_missing_values(df)
    
    # Step 5: Remove markers
    print("\n[Step 5] Removing AMEL, Yindel, DYS391...")
    df = step5_remove_markers(df, scenario['markers_to_keep'])

    # Step 5b: Pad missing markers to full union (only for union scenarios)
    if scenario.get('use_union_padding', False):
        print("\n[Step 5b] Padding missing markers with zeros...")
        df = step5b_pad_missing_markers(df, scenario['markers_to_keep'])

    # Step 6: Encode Dye
    print("\n[Step 6] Encoding Dye...")
    df = step6_encode_dye(df)
    
    # Step 7: Encode Marker
    print("\n[Step 7] Encoding Marker names...")
    df = step7_encode_marker(df, scenario['markers_to_keep'])
    
    # Step 7b: Encode Multiplex
    print("\n[Step 7b] Encoding Multiplex...")
    df = step7b_encode_multiplex(df)
    
    # Step 7c: Encode Injection Time
    print("\n[Step 7c] Encoding Injection Time...")
    df = step7c_encode_injection_time(df)
    
    # Step 8: Create profile_loci
    print("\n[Step 8] Creating profile_loci feature...")
    df = step8_create_profile_loci(df)
    
    # Step 9 is now handled inside dataset.py during Train/Test Splitting
    
    # Step 10: Finalize features
    print("\n[Step 10] Finalizing features...")
    df = step10_finalize_features(df)
    
    # Save processed data
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED_DIR, f"{scenario_name}_processed.csv")
    df.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*40}")
    print(f"SUMMARY: {scenario['name']}")
    print(f"{'='*40}")
    print(f"  Profiles: {df['Sample File'].nunique()}")
    print(f"  Rows: {len(df)}")
    print(f"  Features: {df.shape[1] - 2}")  # -2 for NOC and Sample File
    print(f"  NOC distribution:")
    for noc in sorted(df['NOC'].unique()):
        count = (df['NOC'] == noc).sum()
        profiles = df[df['NOC'] == noc]['Sample File'].nunique()
        print(f"    NOC={noc}: {profiles} profiles, {count} rows")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TAWSEEM Data Preprocessing")
    parser.add_argument('--scenario', type=str, default='single',
                        choices=['single', 'three', 'four', 'four_union', 'all'],
                        help='Which scenario to preprocess')
    args = parser.parse_args()
    
    if args.scenario == 'all':
        for s in ['single', 'three', 'four', 'four_union']:
            preprocess_scenario(s)
    else:
        preprocess_scenario(args.scenario)

