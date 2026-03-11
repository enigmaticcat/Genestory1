"""
TAWSEEM — Google Colab Runner
Run this file in Google Colab with GPU runtime.

Usage on Colab:
  1. Upload the TAWSEEM folder to Google Drive (or clone from GitHub)
  2. Upload PROVEDIt dataset to Google Drive
  3. Open this file in Colab, set Runtime → GPU
  4. Run all cells
"""

# ==============================================================
# CELL 1: Setup — Mount Drive & Install Dependencies
# ==============================================================
# Run this cell first

import subprocess, sys, os

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "torch", "pandas", "numpy", "scikit-learn", 
                       "matplotlib", "seaborn", "openpyxl"])

# Mount Google Drive (uncomment if using Drive)
# from google.colab import drive
# drive.mount('/content/drive')

# Set project root — CHANGE THIS PATH to match your setup
# Option A: Cloned from GitHub to Colab
PROJECT_ROOT = "/content/TAWSEEM"

# Option B: From Google Drive (uncomment and modify)
# PROJECT_ROOT = "/content/drive/MyDrive/TAWSEEM"

# Clone from GitHub if not exists (uncomment and add your repo URL)
# if not os.path.exists(PROJECT_ROOT):
#     !git clone https://github.com/YOUR_USERNAME/TAWSEEM.git {PROJECT_ROOT}

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print(f"Working directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')}")

# ==============================================================
# CELL 2: Upload Dataset
# ==============================================================
# Upload PROVEDIt_1-5-Person CSVs Filtered.zip to Colab

DATASET_DIR = os.path.join(PROJECT_ROOT, "PROVEDIt_1-5-Person CSVs Filtered")

if not os.path.exists(DATASET_DIR):
    print("Dataset not found! Please upload it.")
    print("Option 1: Upload ZIP file directly:")
    print("  - Upload PROVEDIt_1-5-Person CSVs Filtered.zip to Colab")
    print("  - Then run: !unzip 'PROVEDIt_1-5-Person CSVs Filtered.zip' -d .")
    print("")
    print("Option 2: Upload to Google Drive first, then copy:")
    print("  - !cp -r '/content/drive/MyDrive/PROVEDIt_1-5-Person CSVs Filtered' .")
else:
    print(f"Dataset found: {DATASET_DIR}")
    print(f"Contents: {os.listdir(DATASET_DIR)}")

# ==============================================================
# CELL 3: Override config paths for Colab
# ==============================================================

import src.config as config

# Override paths to work in Colab
config.PROJECT_ROOT = PROJECT_ROOT
config.DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "PROVEDIt_1-5-Person CSVs Filtered")
config.DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
config.RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==============================================================
# CELL 4: Preprocessing
# ==============================================================

from src.data_preprocessing import preprocess_scenario

# Choose scenario: "single", "three", or "four"
SCENARIO = "single"  # Start with single (fastest)

df = preprocess_scenario(SCENARIO)
print(f"\nShape: {df.shape}")
print(f"NOC distribution:\n{df['NOC'].value_counts().sort_index()}")

# ==============================================================
# CELL 5: Prepare Datasets
# ==============================================================

from src.dataset import prepare_datasets
from src.config import TRAIN_RATIO, RANDOM_SEED

train_dataset, test_dataset, scaler = prepare_datasets(
    df, train_ratio=TRAIN_RATIO, random_seed=RANDOM_SEED
)
n_features = train_dataset.n_features
print(f"Features: {n_features}")

# ==============================================================
# CELL 6: Train Model
# ==============================================================

from src.train import train_final_model, cross_validate

# Optional: Cross-validation (takes longer)
# cv_accs = cross_validate(train_dataset, n_features, device, SCENARIO)

# Train final model
model, train_metrics, test_metrics, elapsed = train_final_model(
    train_dataset, test_dataset, n_features, device, SCENARIO
)
print(f"\nTraining time: {elapsed:.1f}s")

# ==============================================================
# CELL 7: Visualize Results
# ==============================================================

from src.evaluate import generate_all_plots
import matplotlib.pyplot as plt
from IPython.display import display, Image

generate_all_plots(train_metrics, test_metrics, SCENARIO)

# Display plots inline
for plot_file in sorted(os.listdir(config.RESULTS_DIR)):
    if plot_file.endswith('.png'):
        print(f"\n--- {plot_file} ---")
        display(Image(filename=os.path.join(config.RESULTS_DIR, plot_file)))

# ==============================================================
# CELL 8: Run All Scenarios (Optional)
# ==============================================================

# Uncomment to run all 3 scenarios
"""
all_results = {}
for scenario in ['single', 'three', 'four']:
    print(f"\n{'#'*60}")
    print(f"# SCENARIO: {scenario}")
    print(f"{'#'*60}")
    
    df = preprocess_scenario(scenario)
    train_ds, test_ds, scaler = prepare_datasets(df, TRAIN_RATIO, RANDOM_SEED)
    model, tr_m, te_m, elapsed = train_final_model(
        train_ds, test_ds, train_ds.n_features, device, scenario
    )
    generate_all_plots(tr_m, te_m, scenario)
    all_results[scenario] = {'train_acc': tr_m['accuracy'], 'test_acc': te_m['accuracy']}

from src.evaluate import plot_accuracy_comparison
plot_accuracy_comparison(all_results, os.path.join(config.RESULTS_DIR, 'comparison.png'))
display(Image(filename=os.path.join(config.RESULTS_DIR, 'comparison.png')))
"""
