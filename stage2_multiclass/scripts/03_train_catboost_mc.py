"""
Шаг 3: Обучение многоклассовой CatBoost-модели.

Вход:
  data/train_augmented.parquet   (из шага 2, если есть)
  или data/train_mc.parquet      (только IoT, если шаг 2 не запускался)
  data/val_mc.parquet

Выход:
  models/catboost/model_mc.cbm
  models/catboost/class_mapping.json
  models/catboost/metrics_mc.json
  models/catboost/feature_importance.json
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
    DATA_DIR, MODELS_DIR, FEATURE_CONTRACT_PATH,
    CLASS_NAMES, NUM_CLASSES, CATBOOST_MC_PARAMS, SEED,
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Признаки ──────────────────────────────────────────────────
with open(FEATURE_CONTRACT_PATH) as f:
    FEATURES: list[str] = json.load(f)


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    # Предпочитаем augmented train, иначе обычный
    if name == "train":
        aug = DATA_DIR / "train_augmented.parquet"
        path = aug if aug.exists() else DATA_DIR / "train_mc.parquet"
    else:
        path = DATA_DIR / f"{name}_mc.parquet"

    print(f"  Читаем {path.name} ...")
    df = pd.read_parquet(path)
    feats = [c for c in FEATURES if c in df.columns]
    X = df[feats].values.astype("float32")
    y = df["target_mc"].values.astype("int8")
    print(f"    {X.shape[0]:,} строк x {X.shape[1]} признаков")
    dist = {CLASS_NAMES[i]: int(v) for i, v in zip(*np.unique(y, return_counts=True))}
    print(f"    Классы: {dist}")
    return X, y


print("Загружаем данные...")
X_train, y_train = load_split("train")
X_val,   y_val   = load_split("val")

# ── Обучение ──────────────────────────────────────────────────
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    print("catboost не установлен: pip install catboost")
    sys.exit(1)

train_pool = Pool(X_train, y_train, feature_names=FEATURES[:X_train.shape[1]])
val_pool   = Pool(X_val,   y_val,   feature_names=FEATURES[:X_val.shape[1]])

print(f"\nНачинаем обучение CatBoost MultiClass ({NUM_CLASSES} классов)...")
print(f"Параметры: {CATBOOST_MC_PARAMS}")

model = CatBoostClassifier(**CATBOOST_MC_PARAMS)
t0 = time.time()
model.fit(train_pool, eval_set=val_pool, use_best_model=True)
elapsed = time.time() - t0
print(f"\nОбучение завершено за {elapsed:.1f} сек, лучшая итерация: {model.best_iteration_}")

# ── Сохраняем модель ──────────────────────────────────────────
model_path = MODELS_DIR / "model_mc.cbm"
model.save_model(str(model_path))
print(f"Модель >> {model_path}")

# ── Class mapping ─────────────────────────────────────────────
class_mapping = {str(k): v for k, v in CLASS_NAMES.items()}
mapping_path = MODELS_DIR / "class_mapping.json"
with open(mapping_path, "w") as f:
    json.dump(class_mapping, f, indent=2, ensure_ascii=False)
print(f"Маппинг >> {mapping_path}")

# ── Feature importance ────────────────────────────────────────
fi = dict(zip(FEATURES, model.get_feature_importance().tolist()))
fi_sorted = dict(sorted(fi.items(), key=lambda x: -x[1])[:30])
fi_path = MODELS_DIR / "feature_importance.json"
with open(fi_path, "w") as f:
    json.dump(fi_sorted, f, indent=2)
print(f"Feature importance (top-30) >> {fi_path}")

# ── Быстрые метрики на val ────────────────────────────────────
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)

val_pred = model.predict(X_val).flatten().astype(int)
class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

print("\n-- Метрики на val --------------------------------------")
print(classification_report(y_val, val_pred, target_names=class_labels, zero_division=0))

acc = accuracy_score(y_val, val_pred)
macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)
print(f"Accuracy:    {acc:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

# Сохраняем метрики
from sklearn.metrics import precision_recall_fscore_support
prec, rec, f1, sup = precision_recall_fscore_support(
    y_val, val_pred, zero_division=0
)
per_class = {}
for i in range(NUM_CLASSES):
    if i < len(prec):
        per_class[CLASS_NAMES[i]] = {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]), 4),
            "f1":        round(float(f1[i]), 4),
            "support":   int(sup[i]),
        }

metrics = {
    "model": "CatBoost MultiClass",
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
metrics_path = MODELS_DIR / "metrics_mc.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"\nМетрики >> {metrics_path}")
print("\n[OK] Шаг 3 завершён")
