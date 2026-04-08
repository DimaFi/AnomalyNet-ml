"""
Шаг 4: Предобработка — диагностика inf/nan, fit на train, apply на все splits.

Ключевые принципы:
  - Inf → NaN → median(train)
  - Fit scaler на train (сохраняем для будущих нейросетей, не применяем к деревьям)
  - Все артефакты сохраняются для inference

Выходы:
  data/preprocessed/train_X.parquet, train_y.parquet
  data/preprocessed/val_X.parquet, val_y.parquet
  data/preprocessed/test_X.parquet, test_y.parquet
  artifacts/feature_contract.json
  artifacts/preprocessing_params.json
  artifacts/scaler.joblib
  artifacts/inf_nan_report.json
  reports/04_preprocess_report.json
  reports/04_preprocess_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    SPLITS_DIR, PREPROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR,
    FEATURE_CONTRACT_PATH, NAN_STRATEGY, META_COLUMNS, MAX_RAM_GB,
)
from src.io_utils import load_feature_contract
from src.preprocessing import (
    diagnose_inf_nan, replace_inf, fit_nan_filler,
    apply_nan_filler, fit_scaler, save_preprocessing_artifacts,
)


def load_split(name: str) -> pd.DataFrame:
    """Загружает split parquet."""
    path = SPLITS_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Не найден: {path}")
    return pd.read_parquet(path)


def save_xy(df: pd.DataFrame, feature_columns: list, name: str):
    """Сохраняет X и y отдельно как parquet."""
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X = df[feature_columns].copy()
    y = df[[c for c in META_COLUMNS if c in df.columns]].copy()

    X.to_parquet(PREPROCESSED_DIR / f"{name}_X.parquet", index=False)
    y.to_parquet(PREPROCESSED_DIR / f"{name}_y.parquet", index=False)


def run_preprocess():
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_contract(FEATURE_CONTRACT_PATH)
    print(f"Feature contract: {len(feature_columns)} признаков")

    # Загружаем splits
    print("[1/6] Загрузка splits...")
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    print(f"  train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    # Проверяем RAM
    total_mem_gb = (train.memory_usage(deep=True).sum() +
                    val.memory_usage(deep=True).sum() +
                    test.memory_usage(deep=True).sum()) / (1024**3)
    print(f"  Суммарно в RAM: {total_mem_gb:.2f} GB (лимит: {MAX_RAM_GB} GB)")
    if total_mem_gb > MAX_RAM_GB * 0.8:
        print(f"  [ПРЕДУПРЕЖДЕНИЕ] Близко к лимиту RAM!")

    # Диагностика inf/nan на train (до замены)
    print("[2/6] Диагностика inf/nan на train...")
    inf_nan_report_train = diagnose_inf_nan(train, feature_columns)
    inf_nan_report_val = diagnose_inf_nan(val, feature_columns)
    inf_nan_report_test = diagnose_inf_nan(test, feature_columns)

    full_report = {
        "train": inf_nan_report_train,
        "val": inf_nan_report_val,
        "test": inf_nan_report_test,
    }

    print(f"  train — inf: {inf_nan_report_train['summary']['total_inf_cells']:,}, "
          f"nan: {inf_nan_report_train['summary']['total_nan_cells']:,}")
    print(f"  val   — inf: {inf_nan_report_val['summary']['total_inf_cells']:,}, "
          f"nan: {inf_nan_report_val['summary']['total_nan_cells']:,}")
    print(f"  test  — inf: {inf_nan_report_test['summary']['total_inf_cells']:,}, "
          f"nan: {inf_nan_report_test['summary']['total_nan_cells']:,}")

    # Inf → NaN
    print("[3/6] Замена inf → NaN...")
    train, n_inf_train = replace_inf(train, feature_columns)
    val, n_inf_val = replace_inf(val, feature_columns)
    test, n_inf_test = replace_inf(test, feature_columns)
    print(f"  Заменено inf: train={n_inf_train:,}, val={n_inf_val:,}, test={n_inf_test:,}")

    # Fit NaN filler на train
    print(f"[4/6] Fit NaN filler (strategy={NAN_STRATEGY}) на train...")
    fill_values = fit_nan_filler(train, feature_columns, strategy=NAN_STRATEGY)

    # Apply NaN filler на все
    train = apply_nan_filler(train, feature_columns, fill_values)
    val = apply_nan_filler(val, feature_columns, fill_values)
    test = apply_nan_filler(test, feature_columns, fill_values)

    # Проверка: нет ли ещё nan
    remaining_nan_train = train[feature_columns].isna().sum().sum()
    remaining_nan_val = val[feature_columns].isna().sum().sum()
    remaining_nan_test = test[feature_columns].isna().sum().sum()
    print(f"  Остаточные NaN: train={remaining_nan_train}, val={remaining_nan_val}, test={remaining_nan_test}")

    # Fit scaler (сохраняем, не применяем к данным для деревьев)
    print("[5/6] Fit StandardScaler на train (сохраняем для будущих моделей)...")
    scaler = fit_scaler(train, feature_columns)

    # Сохранение артефактов
    save_preprocessing_artifacts(
        ARTIFACTS_DIR, feature_columns, fill_values,
        NAN_STRATEGY, scaler, full_report,
    )

    # Сохранение preprocessed данных
    print("[6/6] Сохранение preprocessed X/y...")
    save_xy(train, feature_columns, "train")
    save_xy(val, feature_columns, "val")
    save_xy(test, feature_columns, "test")

    # Отчёт
    preprocess_report = {
        "nan_strategy": NAN_STRATEGY,
        "inf_replaced": {"train": n_inf_train, "val": n_inf_val, "test": n_inf_test},
        "remaining_nan": {"train": int(remaining_nan_train), "val": int(remaining_nan_val), "test": int(remaining_nan_test)},
        "fill_values_sample": {k: round(v, 6) for k, v in list(fill_values.items())[:10]},
        "total_ram_gb": round(total_mem_gb, 2),
    }

    json_path = REPORTS_DIR / "04_preprocess_report.json"
    json_path.write_text(json.dumps(preprocess_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown
    lines = ["# Preprocessing report\n"]
    lines.append(f"- Стратегия NaN: **{NAN_STRATEGY}**")
    lines.append(f"- RAM использовано: **{total_mem_gb:.2f} GB**\n")

    lines.append("## Inf замены\n")
    lines.append("| Split | Inf заменено |")
    lines.append("|---|---:|")
    for s, n in [("train", n_inf_train), ("val", n_inf_val), ("test", n_inf_test)]:
        lines.append(f"| {s} | {n:,} |")

    lines.append("\n## Колонки с inf (train)\n")
    if inf_nan_report_train["columns_with_inf"]:
        lines.append("| Колонка | Кол-во | % |")
        lines.append("|---|---:|---:|")
        for col, info in sorted(inf_nan_report_train["columns_with_inf"].items()):
            lines.append(f"| {col} | {info['count']:,} | {info['pct']:.2f}% |")
    else:
        lines.append("Нет колонок с inf.")

    lines.append("\n## Колонки с NaN (train, до заполнения)\n")
    if inf_nan_report_train["columns_with_nan"]:
        lines.append("| Колонка | Кол-во | % |")
        lines.append("|---|---:|---:|")
        for col, info in sorted(inf_nan_report_train["columns_with_nan"].items()):
            lines.append(f"| {col} | {info['count']:,} | {info['pct']:.2f}% |")
    else:
        lines.append("Нет колонок с NaN.")

    lines.append("\n## Артефакты\n")
    lines.append(f"- `artifacts/feature_contract.json` — {len(feature_columns)} признаков")
    lines.append(f"- `artifacts/preprocessing_params.json` — fill values")
    lines.append(f"- `artifacts/scaler.joblib` — StandardScaler")
    lines.append(f"- `artifacts/inf_nan_report.json` — полная диагностика")

    md_path = REPORTS_DIR / "04_preprocess_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nГотово:")
    print(f"  Preprocessed: {PREPROCESSED_DIR}")
    print(f"  Артефакты:    {ARTIFACTS_DIR}")
    print(f"  Report:       {json_path}")


if __name__ == "__main__":
    run_preprocess()
