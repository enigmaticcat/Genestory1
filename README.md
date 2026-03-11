# TAWSEEM: DNA Profiling — Number of Contributors Estimation

Reimplementation of **"TAWSEEM: A Deep-Learning-Based Tool for Estimating the Number of Unknown Contributors in DNA Profiling"** (Electronics 2022, 11, 548).

Phân loại hỗn hợp DNA thành 1–5 người đóng góp (NOC) sử dụng **XGBoost** *(SOTA)* và **MLP (PyTorch)** trên tập dữ liệu PROVEDIt.

> **SOTA:** XGBoost với profile-level features đạt kết quả tốt nhất trong dự án này.

---

## Mục lục

- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Dữ liệu](#dữ-liệu)
- [Chạy XGBoost SOTA](#-chạy-xgboost-sota)
- [Chạy pipeline đầy đủ (MLP + XGBoost + RF)](#chạy-pipeline-đầy-đủ)
- [Các tham số cấu hình](#các-tham-số-cấu-hình)
- [Kết quả kỳ vọng](#kết-quả-kỳ-vọng)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

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

## Tài liệu tham khảo

Alotaibi, H.; Alsolami, F.; Abozinadah, E.; Mehmood, R. **TAWSEEM: A Deep Learning-Based Tool for Estimating the Number of Unknown Contributors in DNA Profiling.** *Electronics* 2022, 11, 548. https://doi.org/10.3390/electronics11040548
