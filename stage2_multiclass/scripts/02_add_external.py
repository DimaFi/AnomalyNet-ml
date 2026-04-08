"""
Шаг 2: Обогащение train-выборки данными из CIC-IDS2018 и CIC-IDS2017.

Для каждого внешнего датасета:
  1. Читаем файлы, извлекаем признаки + метку
  2. Переименовываем колонки через feature_map.py
  3. Маппим текстовую метку → target_mc
  4. Применяем inf→NaN→fill (те же параметры что в stage1)
  5. Добавляем к train

Финальная выборка — с per-class cap (CLASS_SAMPLE_CAP).

Вывод: stage2_multiclass/data/train_augmented.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, FEATURE_CONTRACT_PATH, PREPROCESSING_PARAMS,
    CIC2018_DIR, CIC2017_DIR,
    LABEL_TO_CLASS, CLASS_NAMES, CLASS_SAMPLE_CAP, SEED,
)

# Добавляем путь к stage1 для feature_map
STAGE1_EVAL = Path(__file__).resolve().parents[2] / "stage1_v2_cl" / "eval_external"
sys.path.insert(0, str(STAGE1_EVAL))
from feature_map import CIC2018_MAP, CIC2017_MAP, align_columns

rng = np.random.default_rng(SEED)

# ── Загрузка контракта и параметров предобработки ─────────────
with open(FEATURE_CONTRACT_PATH) as f:
    FEATURES: list[str] = json.load(f)

with open(PREPROCESSING_PARAMS) as f:
    pp = json.load(f)
FILL_VALUES: dict[str, float] = pp["fill_values"]

print(f"Признаков: {len(FEATURES)}")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """inf→NaN→fill_value (те же правила что в stage1)."""
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if col == "target_mc":
            continue
        fill = FILL_VALUES.get(col, 0.0)
        df[col] = df[col].fillna(fill).astype("float32")
    return df


def map_label(raw: str) -> int | None:
    return LABEL_TO_CLASS.get(str(raw).strip().lower())


# ── Базовая train из шага 1 ───────────────────────────────────
base_path = DATA_DIR / "train_mc.parquet"
print(f"\nЧитаем base train: {base_path}")
df_base = pd.read_parquet(base_path)
print(f"  Строк: {len(df_base):,}")
parts = [df_base]


# ── CIC-IDS2018 ───────────────────────────────────────────────
def load_cic2018() -> pd.DataFrame:
    files = sorted(CIC2018_DIR.glob("*.parquet"))
    if not files:
        files = sorted(CIC2018_DIR.glob("*.csv"))
    print(f"\n[CIC-IDS2018] Найдено файлов: {len(files)}")
    if not files:
        print("  Файлы не найдены, пропуск")
        return pd.DataFrame()

    frames = []
    for fp in files:
        print(f"  Читаем {fp.name} ...")
        if fp.suffix == ".parquet":
            df = pd.read_parquet(fp)
        else:
            df = pd.read_csv(fp, low_memory=False)

        if "Label" not in df.columns:
            print(f"    Нет колонки Label, пропуск")
            continue

        labels = df["Label"].apply(map_label)
        df = df[labels.notna()].copy()
        df["target_mc"] = labels[labels.notna()].astype("int8").values

        try:
            X = align_columns(df, CIC2018_MAP, FEATURES)
        except ValueError as e:
            print(f"    Ошибка выравнивания: {e}")
            continue

        X["target_mc"] = df["target_mc"].values
        frames.append(X)
        cls_dist = X["target_mc"].value_counts().to_dict()
        print(f"    {len(X):,} строк  |  классы: {cls_dist}")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    print(f"  Итого CIC-IDS2018: {len(out):,} строк")
    return out


# ── CIC-IDS2017 ───────────────────────────────────────────────
def load_cic2017() -> pd.DataFrame:
    files = sorted(CIC2017_DIR.glob("*.csv"))
    print(f"\n[CIC-IDS2017] Найдено файлов: {len(files)}")
    if not files:
        print("  Файлы не найдены, пропуск")
        return pd.DataFrame()

    frames = []
    for fp in files:
        print(f"  Читаем {fp.name} ...")
        try:
            df = pd.read_csv(fp, low_memory=False, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            try:
                df = pd.read_csv(fp, low_memory=False, encoding="latin-1", on_bad_lines="skip")
            except Exception as e:
                print(f"    Ошибка чтения: {e}")
                continue

        # Убираем пробелы из имён колонок (CIC-2017 известная проблема)
        df.columns = df.columns.str.strip()

        if "Label" not in df.columns:
            print(f"    Нет колонки Label, пропуск")
            continue

        labels = df["Label"].apply(map_label)
        df = df[labels.notna()].copy()
        df["target_mc"] = labels[labels.notna()].astype("int8").values

        try:
            X = align_columns(df, CIC2017_MAP, FEATURES)
        except ValueError as e:
            print(f"    Ошибка выравнивания: {e}")
            continue

        X["target_mc"] = df["target_mc"].values
        frames.append(X)
        cls_dist = X["target_mc"].value_counts().to_dict()
        print(f"    {len(X):,} строк  |  классы: {cls_dist}")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    print(f"  Итого CIC-IDS2017: {len(out):,} строк")
    return out


df_2018 = load_cic2018()
df_2017 = load_cic2017()

if len(df_2018) > 0:
    df_2018 = preprocess(df_2018)
    parts.append(df_2018)

if len(df_2017) > 0:
    df_2017 = preprocess(df_2017)
    parts.append(df_2017)

# ── Объединяем и применяем per-class cap ─────────────────────
print("\nОбъединение датасетов...")
df_all = pd.concat(parts, ignore_index=True)
print(f"  До капирования: {len(df_all):,} строк")

cap_frames = []
for cls_id, cap in CLASS_SAMPLE_CAP.items():
    subset = df_all[df_all["target_mc"] == cls_id]
    if len(subset) > cap:
        subset = subset.sample(n=cap, random_state=SEED)
    if len(subset) > 0:
        cap_frames.append(subset)
    name = CLASS_NAMES[cls_id]
    print(f"  Класс {cls_id} {name:12s}: {len(df_all[df_all['target_mc']==cls_id]):>8,} => {len(subset):>8,}")

df_final = pd.concat(cap_frames, ignore_index=True).sample(frac=1, random_state=SEED)
print(f"\nФинальная augmented train: {len(df_final):,} строк")

out_path = DATA_DIR / "train_augmented.parquet"
df_final.to_parquet(out_path, index=False, compression="zstd")
print(f">> Сохранено: {out_path}")
print("\n[OK] Шаг 2 завершён")
