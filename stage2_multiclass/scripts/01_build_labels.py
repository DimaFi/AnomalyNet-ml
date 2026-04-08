"""
Шаг 1: Строим многоклассовые метки из уже готовых сплитов stage1.

Читает train/val/test parquet из stage1/data/splits/.
В каждом файле есть колонка `source_class_folder` — имя папки атаки.
Добавляем `target_mc` (int8, 0-7) по маппингу из config.py.

Выходные файлы → stage2_multiclass/data/
  train_mc.parquet   — 71 признак + target_mc (без unknown-меток)
  val_mc.parquet
  test_mc.parquet
  class_distribution.json  — сколько семплов каждого класса
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    STAGE1_SPLITS_DIR, DATA_DIR, FEATURE_CONTRACT_PATH,
    LABEL_TO_CLASS, CLASS_NAMES, SEED,
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Загрузка списка признаков ─────────────────────────────────
with open(FEATURE_CONTRACT_PATH) as f:
    FEATURES: list[str] = json.load(f)

print(f"Признаков: {len(FEATURES)}")


def map_label(raw: str) -> int | None:
    """Возвращает class_id или None если метка неизвестна."""
    key = str(raw).strip().lower()
    return LABEL_TO_CLASS.get(key)


def process_split(name: str) -> pd.DataFrame:
    path = STAGE1_SPLITS_DIR / f"{name}.parquet"
    print(f"\n[{name}] Читаем {path} ...")
    df = pd.read_parquet(path)
    print(f"  Строк: {len(df):,}  Колонок: {len(df.columns)}")

    # Маппинг метки
    df["target_mc"] = df["source_class_folder"].apply(map_label)
    unknown = df["target_mc"].isna()
    if unknown.sum() > 0:
        bad = df.loc[unknown, "source_class_folder"].value_counts()
        print(f"  Отброшено (неизвестные метки): {unknown.sum():,}")
        print(f"  {bad.to_dict()}")
    df = df[~unknown].copy()
    df["target_mc"] = df["target_mc"].astype("int8")

    # Оставляем только признаки + метку
    keep = [c for c in FEATURES if c in df.columns] + ["target_mc"]
    missing_feats = [c for c in FEATURES if c not in df.columns]
    if missing_feats:
        print(f"  ВНИМАНИЕ: нет {len(missing_feats)} признаков: {missing_feats[:5]}")
    df = df[keep]

    # Статистика по классам
    dist = df["target_mc"].value_counts().sort_index()
    print("  Распределение классов:")
    for cls_id, count in dist.items():
        print(f"    {cls_id} {CLASS_NAMES[cls_id]:12s} {count:>10,}")

    return df


distributions = {}
for split in ["train", "val", "test"]:
    df = process_split(split)
    out = DATA_DIR / f"{split}_mc.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    print(f"  >> Сохранено: {out} ({len(df):,} строк)")
    distributions[split] = {
        CLASS_NAMES[i]: int(v)
        for i, v in df["target_mc"].value_counts().sort_index().items()
    }

# Сохраняем статистику
dist_path = DATA_DIR / "class_distribution.json"
with open(dist_path, "w", encoding="utf-8") as f:
    json.dump(distributions, f, ensure_ascii=False, indent=2)
print(f"\nСтатистика сохранена >> {dist_path}")
print("\n[OK] Шаг 1 завершён")
