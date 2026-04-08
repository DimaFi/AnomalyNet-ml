"""
Шаг 3: Предобработка — замена inf/NaN медианами из train.
Сохраняет preprocessing_params.json и готовые X/y parquet.

Вывод:
  data/train_X.parquet, train_y.parquet
  data/val_X.parquet,   val_y.parquet
  data/test_X.parquet,  test_y.parquet
  artifacts/preprocessing_params.json
  artifacts/feature_contract.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, STAGE_ROOT, FEATURES_2023

ARTIFACTS_DIR = STAGE_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Загрузка train для подбора fill values ────────────────────
print("Загружаем train для вычисления медиан...")
df_train = pd.read_parquet(DATA_DIR / "train.parquet")
feats = [c for c in FEATURES_2023 if c in df_train.columns]
print(f"  Признаков: {len(feats)}")

X_train = df_train[feats].replace([np.inf, -np.inf], np.nan).astype("float32")

# Медианные fill values по каждому признаку
fill_values = {}
for col in feats:
    median = float(X_train[col].median())
    fill_values[col] = 0.0 if np.isnan(median) else median

params = {"fill_values": fill_values, "features": feats}
params_path = ARTIFACTS_DIR / "preprocessing_params.json"
with open(params_path, "w") as f:
    json.dump(params, f, indent=2, ensure_ascii=False)
print(f"Preprocessing params >> {params_path}")

# Feature contract (список признаков в порядке обучения)
fc_path = ARTIFACTS_DIR / "feature_contract.json"
with open(fc_path, "w") as f:
    json.dump(feats, f, indent=2, ensure_ascii=False)
print(f"Feature contract >> {fc_path}")


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feats_present = [c for c in feats if c in df.columns]
    X = df[feats_present].replace([np.inf, -np.inf], np.nan).astype("float32")
    for col in feats_present:
        X[col] = X[col].fillna(fill_values.get(col, 0.0))
    y = df["target_mc"].astype("int8")
    return X, y


# ── Обрабатываем все 3 сплита ─────────────────────────────────
for split in ["train", "val", "test"]:
    df = pd.read_parquet(DATA_DIR / f"{split}.parquet")
    X, y = preprocess(df)
    X.to_parquet(DATA_DIR / f"{split}_X.parquet", index=False, compression="zstd")
    y.to_frame("target_mc").to_parquet(DATA_DIR / f"{split}_y.parquet", index=False)
    print(f"{split}: {X.shape[0]:,} x {X.shape[1]}  |  классы: {y.value_counts().to_dict()}")

print("\n[OK] Шаг 3 завершён")
