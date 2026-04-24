# AnomalyNet-ml · v1.1

**CatBoost-based IDS model collection for network anomaly detection**  
IoT & General Network · 71 CICFlowMeter features · Binary + multiclass pipelines

> **v1.1** adds `general_network/` — models trained on CICIDS 2017 for PC/home network traffic (FPR=0.08%, Macro F1=0.9777)

---

## Model performance

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Val   | 97.96 %  | 99.95 %   | 97.97 % | 98.95 % | 99.39 % |
| Test  | 98.85 %  | 99.97 %   | 98.87 % | 99.41 % | 99.63 % |

- **Threshold**: 0.70 (tuned on validation set)
- **Best iteration**: 2916
- **Training time**: 36 s (i7-12700F, 32 GB RAM)
- FP rate on benign: 2.6 % (val) / 3.3 % (test)

> **Cross-dataset note**: When evaluated on CIC-IDS 2018 (enterprise traffic), benign FP rate rises to ~94 % — expected domain shift between IoT and enterprise traffic patterns. Attack recall remains ~95 %.

## Models overview

| Model | Dataset | F1 | FPR | Classes |
|-------|---------|-----|-----|---------|
| `model/model.cbm` (IoT Stage1) | CIC IoT-DIAD 2024 | 0.9941 | 0.1% | binary |
| `stage2_multiclass/` (IoT Stage2) | CIC IoT-DIAD 2024 + ext | 0.9309 | — | 8 |
| `stage3_cic2023/` (IoT Stage3) | CIC IoT 2023 | 0.819 | — | 8 |
| **`general_network/general_stage1`** | **CICIDS 2017** | **0.9984** | **0.08%** | **binary** |
| **`general_network/general_stage2`** | **CICIDS 2017** | **0.9777** | — | **7** |

---

## Repository structure

```
AnomalyNet-ml/
├── general_network/              # NEW v1.1 — PC/home network models
│   ├── artifacts/
│   │   ├── feature_contract.json
│   │   ├── preprocessing_params.json  # CICIDS 2017 medians
│   │   ├── threshold.json
│   │   ├── class_names.json
│   │   └── metrics.json
│   └── models/
│       ├── general_stage1/catboost/model.cbm
│       └── general_stage2/catboost/
│           ├── model_mc.cbm
│           └── class_mapping.json
├── model/
│   ├── model.cbm                 # Trained CatBoost model (12 MB)
│   ├── metrics.json              # Full metrics (val + test)
│   ├── best_threshold.json       # Optimal threshold
│   ├── feature_importance.json   # Per-feature importance
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curve.png
├── artifacts/
│   ├── feature_contract.json     # 71 feature names + order
│   ├── preprocessing_params.json # Inf→NaN fill values
│   ├── scaler.joblib             # StandardScaler (fitted)
│   └── inf_nan_report.json
├── src/
│   ├── preprocessing.py          # Feature pipeline
│   ├── evaluation.py             # Metrics computation
│   ├── io_utils.py
│   └── split_logic.py
├── scripts/
│   ├── 01_inspect.py … 09_threshold_select.py
├── eval_external/
│   ├── feature_map.py            # CIC-2018/2017 column rename maps
│   ├── run_eval_cic2018.py       # Cross-dataset evaluation
│   └── diagnose_domain_shift.py  # Feature distribution comparison
├── config.py
├── run_all.py                    # Full training pipeline
├── INFERENCE_GUIDE.md            # Step-by-step inference guide
└── requirements.txt
```

## Quick inference

```python
import catboost
import json, joblib
import numpy as np

# 1. Load artifacts
model = catboost.CatBoostClassifier()
model.load_model("model/model.cbm")

with open("artifacts/preprocessing_params.json") as f:
    params = json.load(f)
scaler = joblib.load("artifacts/scaler.joblib")

# 2. Build feature vector (71 values in order from feature_contract.json)
X_raw = ...  # dict or DataFrame with CICFlowMeter column names

# 3. Preprocess
X = X_raw.replace([np.inf, -np.inf], np.nan)
X = X.fillna(params["fill_values"])
X_scaled = scaler.transform(X)

# 4. Predict
proba = model.predict_proba(X_scaled)[:, 1]
label = np.where(proba >= 0.85, "anomaly",
        np.where(proba >= 0.70, "warning", "normal"))
```

See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for the full feature table, CICFlowMeter setup, and dataset adaptation instructions.

## Training data

Dataset: **CIC IoT Dataset 2024** — 47 attack types, ~8 M flows  
Not included in this repo (3.2 GB). Download from [UNB CIC](https://www.unb.ca/cic/datasets/iotdataset-2024.html) and place in `data/`.

To reproduce training:
```bash
pip install -r requirements.txt
python run_all.py
```

## Requirements

```
catboost>=1.2.3
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
joblib>=1.3.0
```

## License

MIT — model weights and code are free to use for research and educational purposes.
