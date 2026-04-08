"""
Шаг 4: Обучение CatBoost MultiClass на CIC IoT 2023.

Вывод:
  models/catboost/model_mc.cbm
  models/catboost/class_mapping.json
  models/catboost/feature_contract.json  (копия)
  models/catboost/metrics_mc.json
  models/catboost/feature_importance.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODELS_DIR, STAGE_ROOT, CLASS_NAMES, NUM_CLASSES, CATBOOST_PARAMS, SEED

ARTIFACTS_DIR = STAGE_ROOT / "artifacts"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

with open(ARTIFACTS_DIR / "feature_contract.json") as f:
    FEATURES = json.load(f)

print("Загрузка данных...")
X_train = pd.read_parquet(DATA_DIR / "train_X.parquet").values.astype("float32")
y_train = pd.read_parquet(DATA_DIR / "train_y.parquet")["target_mc"].values.astype("int8")
X_val   = pd.read_parquet(DATA_DIR / "val_X.parquet").values.astype("float32")
y_val   = pd.read_parquet(DATA_DIR / "val_y.parquet")["target_mc"].values.astype("int8")

print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
print(f"  Val:   {X_val.shape[0]:,} x {X_val.shape[1]}")

from sklearn.utils import class_weight as cw_util
from catboost import CatBoostClassifier, Pool

train_pool = Pool(X_train, y_train, feature_names=FEATURES)
val_pool   = Pool(X_val,   y_val,   feature_names=FEATURES)

print(f"\nОбучение CatBoost ({NUM_CLASSES} классов)...")
model = CatBoostClassifier(**CATBOOST_PARAMS)
t0 = time.time()
model.fit(train_pool, eval_set=val_pool, use_best_model=True)
elapsed = time.time() - t0
print(f"Обучение завершено: {elapsed:.1f}s, best_iter={model.best_iteration_}")

# Сохраняем модель
model.save_model(str(MODELS_DIR / "model_mc.cbm"))

# Class mapping
mapping = {str(k): v for k, v in CLASS_NAMES.items()}
with open(MODELS_DIR / "class_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

# Feature contract (копируем в папку модели для удобства)
import shutil
shutil.copy(ARTIFACTS_DIR / "feature_contract.json", MODELS_DIR / "feature_contract.json")
shutil.copy(ARTIFACTS_DIR / "preprocessing_params.json", MODELS_DIR / "preprocessing_params.json")

# Feature importance
fi = dict(zip(FEATURES, model.get_feature_importance().tolist()))
fi_top = dict(sorted(fi.items(), key=lambda x: -x[1])[:30])
with open(MODELS_DIR / "feature_importance.json", "w") as f:
    json.dump(fi_top, f, indent=2)

# Метрики на val
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_recall_fscore_support,
)
val_pred = model.predict(X_val).flatten().astype(int)
class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
print("\n-- Метрики на VAL ---")
print(classification_report(y_val, val_pred, target_names=class_labels, zero_division=0))

acc = accuracy_score(y_val, val_pred)
macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)

prec, rec, f1, sup = precision_recall_fscore_support(
    y_val, val_pred, labels=list(range(NUM_CLASSES)), zero_division=0
)
per_class = {
    CLASS_NAMES[i]: {
        "precision": round(float(prec[i]), 4),
        "recall":    round(float(rec[i]), 4),
        "f1":        round(float(f1[i]), 4),
        "support":   int(sup[i]),
    }
    for i in range(NUM_CLASSES)
}

metrics = {
    "model": "CatBoost MultiClass (CIC IoT 2023)",
    "feature_space": "cic2023_47",
    "num_features": len(FEATURES),
    "best_iteration": model.best_iteration_,
    "train_time_sec": round(elapsed, 1),
    "num_classes": NUM_CLASSES,
    "val": {
        "accuracy":    round(acc, 6),
        "macro_f1":    round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class":   per_class,
    }
}
with open(MODELS_DIR / "metrics_mc.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"\nAccuracy:    {acc:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"\nМодель >> {MODELS_DIR / 'model_mc.cbm'}")
print("\n[OK] Шаг 4 завершён")
