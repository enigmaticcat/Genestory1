# TAWSEEM: DNA Profiling — Number of Contributors Estimation

Reimplementation của **"TAWSEEM: A Deep-Learning-Based Tool for Estimating the Number of Unknown Contributors in DNA Profiling"** (Electronics 2022, 11, 548).

Phân loại hỗn hợp DNA thành 1–5 người đóng góp (NOC) sử dụng **XGBoost** *(SOTA)* và **MLP (PyTorch)** trên tập dữ liệu PROVEDIt.

> **SOTA:** XGBoost với profile-level features đạt kết quả tốt nhất trong dự án này.

---

## Dataset Overview & Problem Definition

### Dữ liệu đầu vào (Input)
**PROVEDIt Dataset** - STR Electropherogram Data
- **Nguồn**: Process for the Validation of DNA Mixture Interpretation Technology
- **Format**: CSV files chứa dữ liệu electropherogram STR
- **Cấu trúc mỗi record**:
  - `Sample File`: Identifier duy nhất cho mỗi DNA profile
  - `Marker`: STR loci (D8S1179, D21S11, TH01, D13S317, D16S539, CSF1PO, D2S1338, D3S1358, ...)
  - `Dye`: Kênh huỳnh quang (B=Blue, G=Green, Y=Yellow, R=Red)
  - `Allele 1-100`: Thông tin peak (Allele value, Size in bp, Height in RFU)
  - `OL`: Off-ladder alleles (ngoài thang chuẩn allelic ladder)

### Đặc điểm Dataset
- **Tổng số samples**: 16,382+ DNA profiles (processed)
- **Phân bố theo NOC**:
  - 1-Person: ~4,900 profiles (30%)
  - 2-Person: ~4,100 profiles (25%) 
  - 3-Person: ~3,300 profiles (20%)
  - 4-Person: ~2,500 profiles (15%)
  - 5-Person: ~1,600 profiles (10%)
- **STR Kits**: IDPlus (28 cycles), PP16HS (32 cycles), F6C (29 cycles), GF (29 cycles)
- **Time Intervals**: 5sec, 10sec, 15sec, 20sec, 25sec injection times
- **Feature Dimensions**: 47 features sau preprocessing (9 peaks × 5 attributes + indicators)

### Dữ liệu đầu ra (Output)
**Number of Contributors (NOC) Classification**
- **Target**: NOC ∈ {1, 2, 3, 4, 5} contributors
- **Type**: Multi-class classification problem
- **Evaluation**: Accuracy, Per-class Precision/Recall/F1, ROC-AUC
- **Deployment**: Trained models cho forensic DNA mixture analysis

### Feature Engineering Pipeline
1. **Peak Selection**: Top 9 peaks per STR locus (ranked by height)
2. **Feature Extraction**: Allele value, Size (bp), Height (RFU) cho mỗi peak
3. **Quality Indicators**: Off-Ladder flags, Missing data indicators  
4. **Normalization**: MinMax scaling (0-1 range) cho numerical features
5. **Profile-level Aggregation**: Locus-wise feature vectors
6. **Class Balancing**: Downsampling cho training balance

### Data Quality Assessment
- **Off-Ladder Rate**: ~2.1% (acceptable quality)
- **Missing Data Rate**: ~1.8% (excellent quality)  
- **Class Balance**: Moderately imbalanced (5:1 ratio) → requires balancing
- **Feature Variability**: High variability in Height features, stable Size features

---

## Mục lục

- [Dataset Overview & Problem Definition](#dataset-overview--problem-definition)
- [Comprehensive Dataset Analysis](#comprehensive-dataset-analysis)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Dữ liệu](#dữ-liệu)
- [Chạy XGBoost SOTA](#-chạy-xgboost-sota)
- [Chạy pipeline đầy đủ (MLP + XGBoost + RF)](#chạy-pipeline-đầy-đủ)
- [Evaluation & Visualization](#evaluation--visualization)
- [Các tham số cấu hình](#các-tham-số-cấu-hình)
- [Kết quả kỳ vọng](#kết-quả-kỳ-vọng)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Comprehensive Dataset Analysis

Để hiểu sâu về dataset PROVEDIt và pipeline preprocessing, xem phân tích chi tiết trong:

**📊 [EDA_PROVEDIt.ipynb](EDA_PROVEDIt.ipynb)** - Comprehensive Dataset Analysis

**Nội dung phân tích:**
1. **Dataset Overview**: Cấu trúc tổng quan và tổ chức dữ liệu
2. **Data Structure Analysis**: Format raw data và processed features  
3. **Sample Distribution**: Phân bố theo NOC, STR markers, dye channels
4. **Feature Engineering**: Phân tích 47-dimensional feature vectors
5. **Quality Assessment**: Off-Ladder rates, missing data patterns
6. **Statistical Summary**: Correlation analysis, feature variability
7. **Preprocessing Pipeline**: Chi tiết transformation từ raw → ML-ready

**Key Insights từ Analysis:**
- **16,382+ DNA profiles** với balanced distribution across 5 NOC classes
- **47 engineered features** từ top-9 peaks per STR locus  
- **High data quality**: <2.5% OL và missing data rates
- **Strong NOC correlations** trong Height và Missing indicators
- **Optimized pipeline** với vectorized preprocessing và early balancing

---

## Yêu cầu hệ thống

| Yêu cầu        | Chi tiết                                                        |
| ---------------- | ---------------------------------------------------------------- |
| Python           | 3.9 trở lên                                                    |
| RAM              | Tối thiểu 8GB (khuyến nghị 16GB cho scenario `four_union`) |
| GPU (tùy chọn) | CUDA / Apple MPS — tự động phát hiện                       |
| Hệ điều hành | macOS / Linux / Windows                                          |

---

## Cài đặt môi trường

```bash
# 1. Clone hoặc di chuyển vào thư mục dự án
cd /path/to/TAWSEEM

# 2. Tạo virtual environment
python3 -m venv .venv

# 3. Kích hoạt virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 4. Cài đặt tất cả dependencies (bao gồm XGBoost)
pip install -r requirements.txt
```

---

## Cấu trúc dự án

```
TAWSEEM/
├── PROVEDIt_1-5-Person CSVs Filtered/   ← Dữ liệu thô
├── data/processed/                        ← CSV đã tiền xử lý (tự động tạo)
├── results/                              ← Kết quả, biểu đồ, mô hình
├── src/
│   ├── config.py             # Đường dẫn, markers, siêu tham số
│   ├── data_preprocessing.py # Pipeline tiền xử lý 10+ bước (có step 5b padding)
│   ├── dataset.py            # PyTorch Dataset + feature engineering (17 features/marker)
│   ├── model.py              # MLP (15 hidden layers, 7 dropout)
│   ├── train.py              # Huấn luyện + 5-fold CV
│   ├── train_xgb.py          # XGBoost SOTA — file chạy độc lập
│   ├── tree_models.py        # XGBoost + Random Forest
│   ├── evaluate.py           # Metrics + biểu đồ
│   └── main.py               # Entry point CLI
└── README.md
```

---

## Dữ liệu

**PROVEDIt** — Project Research Openness for Validation with Empirical Data

Tải `PROVEDIt_1-5-Person CSVs Filtered.zip` tại: 👉 https://lftdi.camden.rutgers.edu/provedit
Giải nén vào thư mục gốc dự án.

---

## 🏆 Chạy XGBoost SOTA

```bash
# Scenario single (nhanh nhất, ~2 phút)
python3 src/train_xgb.py --scenario single

# Scenario four_union (khuyến nghị — 24 loci, đầy đủ nhất)
python3 src/train_xgb.py --scenario four_union

# Tất cả scenario
python3 src/train_xgb.py --scenario all

# Bỏ qua tiền xử lý (đã có CSV)
python3 src/train_xgb.py --scenario four_union --skip-preprocessing
```

---

## Chạy pipeline đầy đủ (MLP + XGBoost + RF)

```bash
# Single: GF29, 22 loci
python3 src/main.py --scenario single

# Three: IDPlus28 + IDPlus29 + GF29, 14 loci chung
python3 src/main.py --scenario three

# Four: 4 multiplex, 13 loci chung (baseline)
python3 src/main.py --scenario four

# ⭐ Four Union: 4 multiplex, 24 loci union + padding (khuyến nghị)
python3 src/main.py --scenario four_union

# Tất cả 4 scenario + so sánh
python3 src/main.py --scenario all

# Bỏ qua preprocessing / cross-validation
python3 src/main.py --scenario four_union --skip-preprocessing --skip-cv
```

> **Tại sao `four_union` tốt hơn `four`?**
>
> - **24 loci** (union) thay vì **13 loci** (intersection) → không mất thông tin GF29-exclusive
> - Markers thiếu trong kit → **zero-padding** với `Missing_Marker=1` flag
> - Split **stratified theo (NOC × injection_time)** → không bias theo thời gian injection
> - Class 1 cap **tự tính**: `floor(mean(NOC 2-5) / 5)` thay vì hằng số cứng
> - **421 features** = 24 markers × 17 features + 13 profile aggregates

---

## Các tham số cấu hình (`src/config.py`)

| Tham số                    | Mặc định | Ý nghĩa                          |
| --------------------------- | ----------- | ---------------------------------- |
| `EPOCHS`                  | 200         | Số epoch huấn luyện MLP         |
| `BATCH_SIZE`              | 30          | Batch size                         |
| `LEARNING_RATE`           | 0.001       | Learning rate Adam                 |
| `NUM_CV_FOLDS`            | 5           | Số fold cross-validation          |
| `TRAIN_RATIO`             | 0.7         | Tỷ lệ train/test (70/30)         |
| `RANDOM_SEED`             | 42          | Seed ngẫu nhiên                  |
| `EARLY_STOPPING_PATIENCE` | 30          | Epochs chờ trước khi dừng sớm |

> **Class 1 cap**: tính tự động = `floor(mean(NOC 2-5 counts) / 5)` — không cần chỉnh thủ công.

## Quy trình tiền xử lý

| Bước       | Mô tả                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| 1            | Load CSV, gán nhãn NOC (1–5 người)                                                                 |
| 2            | Giữ tối đa 10 allele positions                                                                       |
| 3            | Xử lý OL (Out-of-Ladder) → indicator columns                                                         |
| 4            | Xử lý missing values → indicators + fill 0                                                           |
| 5            | Loại markers AMEL, Yindel                                                                              |
| **5b** | **[four_union]** Pad markers thiếu = zeros + `Missing_Marker=1` → mọi profile có 24 markers |
| 6            | Encode Dye (B=0, G=1, P=2, R=3, Y=4)                                                                    |
| 7            | Encode Marker name → int (0-based)                                                                     |
| 7b           | Encode Multiplex                                                                                        |
| 7c           | Encode Injection Time                                                                                   |
| 8            | Tạo `profile_loci` feature                                                                           |
| 9            | Downsample NOC=1: cap =`floor(mean(NOC 2-5) × 5)`, stratified by injection_time                      |
| 10           | Chọn ~75 features marker-level                                                                         |
| 11           | **Feature engineering profile-level** (xem bên dưới)                                           |
| 12           | **MinMaxScaler** → [0, 1]                                                                        |

---

## Feature Engineering (profile-level — `dataset.py`)

### Per-marker features (17 × n_markers)

| #            | Feature                                            | Mô tả                                             |
| ------------ | -------------------------------------------------- | --------------------------------------------------- |
| 1            | `n_alleles`                                      | Số allele hợp lệ                                 |
| 2–4         | `h1, h2, h3`                                     | Top 3 peak heights                                  |
| 5–9         | `sum_h, mean_h, std_h, h_ratio, h_range`         | Thống kê heights                                  |
| 10           | `n_ol`                                           | Số Out-of-Ladder                                   |
| 11           | `n_missing`                                      | Số allele thiếu                                   |
| 12–15       | `stutter_ratio, snr_top2, log1p_h1, log1p_sum_h` | Đặc trưng nâng cao                              |
| **16** | **`missing_marker`**                       | **1 = locus pad (kit không đo), 0 = thật** |
| **17** | **`marker_index`**                         | **Index locus (0–23) — nhận dạng locus**  |

### Profile aggregates (13 features)

| #            | Feature                                                   | Mô tả                                                                   |
| ------------ | --------------------------------------------------------- | ------------------------------------------------------------------------- |
| 1–5         | MAC, mean/std allele counts, count(≥3), count(≥5)       | Thống kê allele                                                         |
| 6–10        | total OL, mean/std max heights, total peaks, total signal | Thống kê heights                                                        |
| **11** | **`n_missing_markers`**                           | **Số loci bị pad trong profile (0 nếu không phải four_union)** |
| **12** | **`multiplex_id`**                                | **Kit ID (0–3)**                                                   |
| **13** | **`injection_time_id`**                           | **Injection time ID (0–4)**                                        |

### Kích thước vector vào model

| Scenario             | n_markers    | Flat features                 |
| -------------------- | ------------ | ----------------------------- |
| Single (GF29)        | 22           | 22 × 17 + 13 =**387**  |
| Three (3 kits)       | 14           | 14 × 17 + 13 =**251**  |
| Four (baseline)      | 13           | 13 × 17 + 13 =**234**  |
| **Four Union** | **24** | **24 × 17 + 13 = 421** |

---

## Evaluation & Visualization

TAWSEEM cung cấp bộ công cụ evaluation và visualization toàn diện để phân tích performance và tạo báo cáo.

### 🎯 Model Performance Evaluation

**Automated Evaluation trong Main Pipeline:**
```bash
# Chạy với comprehensive evaluation
python3 src/main.py --scenario single
```

**Output Files tự động tạo:**
- `{scenario}_auc_metrics.csv` - AUC metrics của tất cả 3 models  
- `{scenario}_three_model_roc_curves.png` - ROC comparison (MLP vs RF vs XGBoost)
- `{scenario}_dataset_distribution.png` - Dataset distribution by NOC
- `{scenario}_confusion_matrix_test.png` - Confusion matrix chi tiết
- `{scenario}_performance_metrics.png` - Precision/Recall/F1 charts

### 📊 Standalone Evaluation Tools

**Test Evaluation Functions (không cần real data):**
```bash
# Test với mock data - không cần train models
python3 evaluation_only.py
```

**Generate Specific Plots:**
```bash
# Tạo plots từ processed data  
python3 plot_generator.py --scenario single --plot-type both

# Tạo mock plots để test visualization
python3 plot_generator.py --scenario single --mock-data
```

**Comprehensive Summary Report:**
```bash
# Tạo summary từ tất cả results
python3 generate_summary.py
```

### 📈 Key Metrics & Visualizations

**Performance Metrics:**
- **Multi-class Accuracy** (primary metric)
- **Per-class Precision/Recall/F1** cho mỗi NOC (1-5)  
- **ROC-AUC** (One-vs-Rest) cho tất cả classes
- **Confusion Matrices** với misclassification patterns

**Model Comparison:**
- **3-Model ROC Comparison** (MLP vs Random Forest vs XGBoost)
- **AUC Metrics Table** với per-class breakdown
- **Performance Summary** với statistical analysis  
- **Class Distribution Analysis** để assess balance

**Chi tiết usage và examples**: Xem [EVALUATION_README.md](EVALUATION_README.md)

---

## Kết Quả Kỳ Vọng

### Model Performance Benchmarks

**Expected Accuracy Range** (based on paper và implementation):

| Scenario       | XGBoost (SOTA) | Random Forest | MLP (PyTorch) |
| -------------- | -------------- | ------------- | ------------- |
| Single (GF29)  | **85-90%**    | 82-87%        | 83-88%        |
| Four Union     | **82-87%**    | 78-83%        | 80-85%        |
| Four (baseline)| **80-85%**    | 76-81%        | 78-83%        |

### Per-Class Performance Expectations

**NOC Classification Difficulty:**
- **1-Person**: Easiest (90-95% accuracy) - clear single-source patterns
- **2-Person**: Good (80-90% accuracy) - distinguishable mixture patterns  
- **3-Person**: Moderate (75-85% accuracy) - increased complexity
- **4-Person**: Challenging (70-80% accuracy) - significant overlap
- **5-Person**: Most difficult (60-75% accuracy) - highest complexity

### Training Times (MacBook M1/M2)

| Component           | Single | Four Union | Note                    |
| ------------------- | ------ | ---------- | ----------------------- |
| Preprocessing       | 2-3min | 4-6min     | Vectorized, optimized   |
| MLP Training        | 3-5min | 6-10min    | 200 epochs với early stopping |
| XGBoost Training    | 1-2min | 2-4min     | 5-fold CV + final model |
| Random Forest       | 0.5-1min| 1-2min     | Parallel training       |
| **Total Pipeline**  | **8-12min** | **15-25min** | Complete end-to-end |

### File Outputs

**Results Directory Structure:**
```
results/
├── single_auc_metrics.csv              # AUC breakdown tất cả models
├── single_three_model_roc_curves.png   # ROC comparison plot  
├── single_dataset_distribution.png     # Class distribution chart
├── single_confusion_matrix_test.png    # Detailed confusion matrix
├── single_performance_metrics.png      # Precision/Recall/F1 bars
├── performance_summary.txt             # Comprehensive analysis
└── model_comparison.csv               # Cross-scenario comparison
```

**Expected File Sizes:**
- CSV files: 5-50KB (metrics tables)
- PNG plots: 200-800KB (high-resolution charts)  
- Summary reports: 10-30KB (text analysis)

---

## Tài liệu tham khảo

Alotaibi, H.; Alsolami, F.; Abozinadah, E.; Mehmood, R. **TAWSEEM: A Deep Learning-Based Tool for Estimating the Number of Unknown Contributors in DNA Profiling.** *Electronics* 2022, 11, 548. https://doi.org/10.3390/electronics11040548
