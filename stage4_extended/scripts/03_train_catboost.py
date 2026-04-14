"""
Шаг 3: Обучить CatBoost MultiClass на augmented train.

Использует:
  - stage4_extended/data/train_augmented_X.parquet
  - stage3_cic2023/data/val_X.parquet + val_y.parquet  (валидация без изменений)

Сохраняет:
  - stage4_extended/models/catboost/model_mc.cbm
  - stage4_extended/models/catboost/class_mapping.json
  - stage4_extended/models/catboost/metrics_mc.json
  - stage4_extended/models/catboost/preprocessing_params.json  (для scaler)
  - stage4_extended/models/catboost/feature_contract.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    STAGE3_ROOT, DATA_DIR, MODELS_DIR,
    FEATURES_2023, CLASS_NAMES, NUM_CLASSES,
    CATBOOST_PARAMS, SEED,
)

print("=" * 60)
print("Stage4 / Шаг 3: Обучение CatBoost MultiClass")
print("=" * 60)

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score

# ── Загрузка данных ───────────────────────────────────────────
print("\nЧитаем train_augmented...")
X_train = pd.read_parquet(DATA_DIR / "train_augmented_X.parquet")
y_train = pd.read_parquet(DATA_DIR / "train_augmented_y.parquet")["target"]

print(f"Train: {len(X_train):,} строк, {X_train.shape[1]} признаков")
print("Train классы:", y_train.value_counts().to_dict())

print("\nЧитаем val (Stage3)...")
X_val = pd.read_parquet(STAGE3_ROOT / "data" / "val_X.parquet")
y_val_df = pd.read_parquet(STAGE3_ROOT / "data" / "val_y.parquet")
y_val = y_val_df.iloc[:, 0]

print(f"Val:   {len(X_val):,} строк")

# Гарантируем порядок колонок
X_train = X_train[FEATURES_2023]
X_val   = X_val[FEATURES_2023]

# ── Обучение ──────────────────────────────────────────────────
print("\nЗапускаем CatBoost...")
model = CatBoostClassifier(**CATBOOST_PARAMS)

t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
)
elapsed = time.time() - t0
print(f"\nОбучение заняло: {elapsed/60:.1f} мин")
print(f"Лучшая итерация: {model.best_iteration_}")

# ── Сохранение модели ─────────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODELS_DIR / "model_mc.cbm"
model.save_model(str(model_path))
print(f"Модель: {model_path}")

# class_mapping.json (нужен приложению)
class_mapping = {str(k): v for k, v in CLASS_NAMES.items()}
with open(MODELS_DIR / "class_mapping.json", "w") as f:
    json.dump(class_mapping, f, ensure_ascii=False, indent=2)

# feature_contract.json
with open(MODELS_DIR / "feature_contract.json", "w") as f:
    json.dump(FEATURES_2023, f, ensure_ascii=False, indent=2)

# Копируем preprocessing_params из Stage3 (одинаковые признаки)
import shutil
stage3_pp = STAGE3_ROOT / "models" / "catboost" / "preprocessing_params.json"
if stage3_pp.exists():
    shutil.copy(stage3_pp, MODELS_DIR / "preprocessing_params.json")
    print(f"preprocessing_params скопирован из Stage3")

# ── Быстрая валидация ─────────────────────────────────────────
print("\nВалидация на val set...")
y_pred = model.predict(X_val).flatten().astype(int)
macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
print(f"Macro F1 (val): {macro_f1:.4f}")
print("\nPer-class:")
report = classification_report(
    y_val, y_pred,
    target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
    zero_division=0
)
print(report)

# ── Сохранение метрик ─────────────────────────────────────────
from sklearn.metrics import precision_recall_fscore_support
p, r, f, s = precision_recall_fscore_support(y_val, y_pred, average=None,
                                               labels=list(range(NUM_CLASSES)),
                                               zero_division=0)
metrics = {
    "macro_f1":    float(macro_f1),
    "best_iter":   int(model.best_iteration_),
    "train_time_min": round(elapsed / 60, 1),
    "per_class": {
        CLASS_NAMES[i]: {
            "precision": float(p[i]),
            "recall":    float(r[i]),
            "f1":        float(f[i]),
            "support":   int(s[i]),
        }
        for i in range(NUM_CLASSES)
    }
}
with open(MODELS_DIR / "metrics_mc.json", "w") as f_:
    json.dump(metrics, f_, ensure_ascii=False, indent=2)
print(f"\nМетрики: {MODELS_DIR / 'metrics_mc.json'}")
print(f"\n[OK] Шаг 3 завершён. Macro F1 = {macro_f1:.4f}")
