"""
Шаг 2: Объединить Stage3 train + CIC-IDS-2018, применить per-class cap.

Вход:
  - stage3_cic2023/data/train_X.parquet + train_y.parquet (863K строк)
  - stage4_extended/data/cic2018_mapped.parquet (из шага 1)

Выход: stage4_extended/data/train_augmented.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    STAGE3_ROOT, DATA_DIR, FEATURES_2023,
    CLASS_NAMES, CLASS_SAMPLE_CAP, SEED,
)

print("=" * 60)
print("Stage4 / Шаг 2: Merge Stage3 + CIC-IDS-2018")
print("=" * 60)

rng = np.random.default_rng(SEED)

# ── Загрузка Stage3 train ─────────────────────────────────────
stage3_X = pd.read_parquet(STAGE3_ROOT / "data" / "train_X.parquet")
stage3_y = pd.read_parquet(STAGE3_ROOT / "data" / "train_y.parquet")
target_col = stage3_y.columns[0]
stage3_X["target"] = stage3_y[target_col].values

print(f"\nStage3 train: {len(stage3_X):,} строк")
print("  Классы:", stage3_y[target_col].value_counts().to_dict())

# ── Загрузка CIC-IDS-2018 ─────────────────────────────────────
cic2018_path = DATA_DIR / "cic2018_mapped.parquet"
if not cic2018_path.exists():
    print(f"Файл {cic2018_path} не найден. Запустите сначала шаг 1.")
    sys.exit(1)

df_2018 = pd.read_parquet(cic2018_path)
print(f"\nCIC-IDS-2018: {len(df_2018):,} строк")
print("  Классы:", df_2018["target"].value_counts().to_dict())

# ── Объединение ────────────────────────────────────────────────
df_all = pd.concat([stage3_X, df_2018], ignore_index=True)
print(f"\nДо капирования: {len(df_all):,} строк")
print("  Классы:", df_all["target"].value_counts().to_dict())

# ── Per-class cap ─────────────────────────────────────────────
parts = []
print("\nПрименяем per-class cap:")
for cls_id, cap in CLASS_SAMPLE_CAP.items():
    subset = df_all[df_all["target"] == cls_id]
    orig_n = len(subset)
    if orig_n == 0:
        print(f"  {cls_id} {CLASS_NAMES[cls_id]:12s}: 0 строк (пропуск)")
        continue
    if orig_n > cap:
        subset = subset.sample(n=cap, random_state=SEED)
    parts.append(subset)
    print(f"  {cls_id} {CLASS_NAMES[cls_id]:12s}: {orig_n:>10,} => {len(subset):>10,}")

df_final = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED)
print(f"\nФинальный размер: {len(df_final):,} строк")

# ── Разделяем X и y ───────────────────────────────────────────
y_out = df_final["target"].reset_index(drop=True)
X_out = df_final[FEATURES_2023].astype("float32").reset_index(drop=True)

out_X = DATA_DIR / "train_augmented_X.parquet"
out_y = DATA_DIR / "train_augmented_y.parquet"
X_out.to_parquet(out_X, index=False, compression="zstd")
y_out.to_frame("target").to_parquet(out_y, index=False, compression="zstd")

print(f">> X сохранён: {out_X}")
print(f">> y сохранён: {out_y}")
print("\nИтоговое распределение классов:")
for cls_id, cnt in y_out.value_counts().sort_index().items():
    print(f"  {cls_id} {CLASS_NAMES[cls_id]:12s}: {cnt:>10,}")

print("[OK] Шаг 2 завершён")
