"""
Шаг 2: Читаем все 169 файлов CIC IoT 2023, маппим метки,
делаем стратифицированный split train/val/test с гарантированными
примерами редких классов (Recon, BruteForce, WebAttack) в val и test.

Вывод:
  data/train.parquet
  data/val.parquet
  data/test.parquet
  data/split_stats.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATASET_DIR, DATA_DIR, FEATURES_2023, LABEL_COL,
    LABEL_TO_CLASS, CLASS_NAMES, CLASS_SAMPLE_CAP, SPLIT_RATIOS, SEED,
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

files = sorted(DATASET_DIR.glob("*.csv"))
print(f"Читаем {len(files)} файлов...")

frames = []
t0 = time.time()
unknown_labels: dict[str, int] = {}

for i, fp in enumerate(files):
    df = pd.read_csv(fp, low_memory=False)

    # Маппинг меток
    raw = df[LABEL_COL].str.strip().str.lower()
    mc  = raw.map(LABEL_TO_CLASS)

    unknown = df[mc.isna()][LABEL_COL].value_counts()
    for k, v in unknown.items():
        unknown_labels[str(k)] = unknown_labels.get(str(k), 0) + int(v)

    df = df[mc.notna()].copy()
    mc = mc[mc.notna()]

    # Выбираем только нужные признаки
    feat_cols = [c for c in FEATURES_2023 if c in df.columns]
    out = df[feat_cols].copy().astype("float32", errors="ignore")
    out["target_mc"] = mc.values.astype("int8")
    frames.append(out)

    if (i + 1) % 30 == 0:
        print(f"  [{i+1}/{len(files)}]  {time.time()-t0:.0f}s")

if unknown_labels:
    print(f"\nОтброшены неизвестные метки: {unknown_labels}")

print(f"\nОбъединяем {len(frames)} фреймов...")
df_all = pd.concat(frames, ignore_index=True)
print(f"Итого: {len(df_all):,} строк")

dist = df_all["target_mc"].value_counts().sort_index()
print("\nРаспределение по классам:")
for cls_id, cnt in dist.items():
    print(f"  {cls_id} {CLASS_NAMES[cls_id]:12s} {cnt:>10,}")

# ── Стратифицированный split ──────────────────────────────────
print("\nДелаем стратифицированный split 70/15/15...")
# Сначала отделяем test
df_trainval, df_test = train_test_split(
    df_all,
    test_size=SPLIT_RATIOS["test"],
    stratify=df_all["target_mc"],
    random_state=SEED,
)
# Потом val из оставшегося
val_ratio = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])
df_train, df_val = train_test_split(
    df_trainval,
    test_size=val_ratio,
    stratify=df_trainval["target_mc"],
    random_state=SEED,
)

# Применяем cap только к train
cap_parts = []
for cls_id, cap in CLASS_SAMPLE_CAP.items():
    subset = df_train[df_train["target_mc"] == cls_id]
    if len(subset) > cap:
        subset = subset.sample(n=cap, random_state=SEED)
    if len(subset) > 0:
        cap_parts.append(subset)
df_train = pd.concat(cap_parts).sample(frac=1, random_state=SEED)

# Сохраняем
for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    out = DATA_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    dist_split = df["target_mc"].value_counts().sort_index()
    print(f"\n  {name}: {len(df):,} строк")
    for cls_id, cnt in dist_split.items():
        name_cls = CLASS_NAMES.get(int(cls_id), "?")
        print(f"    {cls_id} {name_cls:12s} {cnt:>8,}")

# Статистика
stats = {
    split: {
        CLASS_NAMES[int(i)]: int(v)
        for i, v in df["target_mc"].value_counts().sort_index().items()
    }
    for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]
}
with open(DATA_DIR / "split_stats.json", "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Сплиты сохранены в {DATA_DIR}")
