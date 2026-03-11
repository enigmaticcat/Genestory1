# TAWSEEM Evaluation & Visualization Tools

Bộ công cụ evaluation và visualization cho TAWSEEM DNA Profiling project, được thiết kế để tạo ra các plots và metrics mà không cần chạy toàn bộ pipeline training.

## 📁 Files Structure

```
HH/
├── src/
│   └── evaluate.py              # Core evaluation functions (updated)
├── evaluation_only.py           # Test evaluation functions với mock data  
├── generate_summary.py          # Tạo summary report từ results
├── plot_generator.py           # Utility để tạo specific plots
├── EVALUATION_README.md        # This file
└── results/                    # Output directory (tự động tạo)
    ├── *_auc_metrics.csv       # AUC metrics của các models
    ├── *_three_model_roc_curves.png  # ROC comparison 3 models
    ├── *_dataset_distribution.png     # Dataset distribution  
    └── performance_summary.txt # Tổng hợp performance
```

## 🔧 Core Functions (src/evaluate.py)

### Các functions mới được thêm:

#### 1. `save_auc_metrics_to_csv()`
- Lưu AUC metrics của tất cả models vào CSV file
- Bao gồm: Macro AUC, Accuracy, Per-class AUC cho 5 classes
- Output: `{scenario}_auc_metrics.csv`

#### 2. `plot_three_model_roc_comparison()`  
- Tạo ROC curves comparison cho 3 models (MLP, Random Forest, XGBoost)
- Hiển thị AUC và Accuracy trên plot
- Output: `{scenario}_three_model_roc_curves.png`

#### 3. `generate_comprehensive_evaluation()`
- Tạo đầy đủ tất cả plots và metrics
- Bao gồm: ROC curves, dataset distribution, AUC metrics, confusion matrix
- Thay thế `generate_all_plots()` trong main pipeline

## 🚀 Usage

### 1. Test Evaluation Functions (Không cần data thật)

```bash
# Test tất cả evaluation functions với mock data
python evaluation_only.py
```

**Output:**
- `test_scenario_mock_distribution.png` - Dataset distribution plot
- `test_scenario_three_model_roc_curves.png` - 3-model ROC comparison  
- `test_scenario_auc_metrics.csv` - AUC metrics table
- Các plots khác (confusion matrix, performance metrics)

### 2. Tạo Plots từ Data Có Sẵn

```bash
# Tạo distribution plot từ processed data
python plot_generator.py --scenario single --plot-type distribution

# Tạo ROC plots với mock data (để test visualization)
python plot_generator.py --scenario single --plot-type roc --mock-data

# Tạo cả hai loại plots
python plot_generator.py --scenario single --plot-type both
```

**Arguments:**
- `--scenario`: Tên scenario (single, three, four, four_union)
- `--plot-type`: Loại plot (distribution, roc, both)  
- `--mock-data`: Sử dụng mock data thay vì real data

### 3. Tạo Summary Report

```bash
# Tạo summary từ tất cả AUC metrics có sẵn
python generate_summary.py
```

**Output:**
- `performance_summary.txt` - Text report chi tiết
- `model_comparison.csv` - Comparison table
- Console output với analysis

### 4. Chạy Full Pipeline (Updated)

```bash
# Main pipeline với comprehensive evaluation
python src/main.py --scenario single --skip-cv
```

Pipeline bây giờ sẽ tự động tạo:
- 3-model ROC comparison
- AUC metrics CSV  
- Dataset distribution
- Tất cả plots truyền thống

## 📊 Output Files

### AUC Metrics CSV Format:
```csv
Scenario,Model,Macro_AUC,Accuracy,AUC_1Person,AUC_2Person,AUC_3Person,AUC_4Person,AUC_5Person
single,MLP,0.8543,0.8500,0.9012,0.8765,0.8321,0.7890,0.7654
single,RandomForest,0.8234,0.8200,0.8876,0.8543,0.8098,0.7654,0.7321
single,XGBoost,0.8112,0.8000,0.8654,0.8321,0.7987,0.7543,0.7098
```

### Performance Summary Format:
```
MODEL PERFORMANCE OVERVIEW
                 Macro_AUC              Accuracy
                mean  std  min   max   mean  std  min   max
Model
MLP            0.854 0.02 0.834 0.874  0.850 0.01 0.840 0.860
RandomForest   0.823 0.03 0.793 0.853  0.820 0.02 0.800 0.840  
XGBoost        0.811 0.02 0.791 0.831  0.800 0.01 0.790 0.810
```

## 🎯 Use Cases

### 1. **Máy không đủ mạnh để train**
```bash
# Chỉ test evaluation functions
python evaluation_only.py

# Tạo mock plots để xem visualization  
python plot_generator.py --scenario single --mock-data
```

### 2. **Có data processed, muốn tạo plots**
```bash
# Tạo distribution plot từ processed data
python plot_generator.py --scenario single --plot-type distribution

# Tạo summary từ existing results
python generate_summary.py
```

### 3. **Development & Testing**
```bash
# Test new evaluation functions
python evaluation_only.py

# Generate specific plots cho presentation
python plot_generator.py --scenario single --plot-type both
```

### 4. **Full Analysis**
```bash
# Chạy full pipeline với new evaluation
python src/main.py --scenario single

# Tạo comprehensive summary
python generate_summary.py
```

## 🔍 Key Features

1. **Mock Data Testing** - Test evaluation functions mà không cần real data
2. **Modular Plot Generation** - Tạo specific plots theo nhu cầu  
3. **Comprehensive Metrics** - AUC cho tất cả models và classes
4. **Performance Summary** - Tự động analysis và comparison
5. **Flexible Usage** - Có thể chạy riêng lẻ hoặc trong pipeline

## 📈 Visualization Examples

### 1. Three-Model ROC Comparison
- So sánh ROC curves của MLP, Random Forest, XGBoost
- Hiển thị AUC và Accuracy cho mỗi model
- Macro-average AUC cho multi-class classification

### 2. Dataset Distribution  
- Bar chart hiển thị số lượng profiles cho mỗi class (1-5 Person)
- Annotations với exact numbers
- Consistent styling với paper figures

### 3. AUC Metrics Table
- Per-class AUC cho tất cả 5 classes
- Macro AUC và overall Accuracy
- Easy comparison across models và scenarios

## ⚙️ Configuration

Tất cả settings được định nghĩa trong `src/config.py`:
- `RESULTS_DIR`: Output directory cho plots và metrics
- `NUM_CLASSES`: Số classes (default: 5)
- Plot styling: Colors, fonts, figure sizes

## 🐛 Troubleshooting

### Common Issues:

1. **"Results directory not found"**
   ```bash
   mkdir -p results
   ```

2. **"Processed data not found"**  
   - Chạy preprocessing trước hoặc use `--mock-data`

3. **"No AUC metrics files found"**
   - Chạy evaluation trước hoặc use `evaluation_only.py`

### Dependencies:
- pandas, numpy, matplotlib, seaborn  
- scikit-learn (for metrics)
- Tất cả dependencies trong `requirements.txt`

---

**Note:** Các scripts này được thiết kế để hoạt động độc lập và không require full TAWSEEM pipeline để chạy. Perfect cho testing và development!