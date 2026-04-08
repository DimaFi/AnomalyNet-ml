"""
Единая логика предобработки — используется и при обучении, и при inference.

Ключевой принцип: fit на train, apply на val/test/inference.
Артефакты (fill values, scaler) сохраняются для воспроизводимости.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib


def diagnose_inf_nan(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Dict:
    """
    Подробная диагностика inf и nan в фичах.
    Возвращает словарь с информацией по каждой колонке.
    """
    report = {
        "total_rows": len(df),
        "columns_with_inf": {},
        "columns_with_nan": {},
        "summary": {
            "total_inf_cells": 0,
            "total_nan_cells": 0,
            "columns_affected_inf": 0,
            "columns_affected_nan": 0,
        },
    }

    for col in feature_columns:
        if col not in df.columns:
            continue
        series = df[col]
        n_inf = int(np.isinf(series).sum()) if series.dtype.kind == "f" else 0
        n_nan = int(series.isna().sum())

        if n_inf > 0:
            report["columns_with_inf"][col] = {
                "count": n_inf,
                "pct": round(100.0 * n_inf / len(df), 4),
            }
            report["summary"]["total_inf_cells"] += n_inf
            report["summary"]["columns_affected_inf"] += 1

        if n_nan > 0:
            report["columns_with_nan"][col] = {
                "count": n_nan,
                "pct": round(100.0 * n_nan / len(df), 4),
            }
            report["summary"]["total_nan_cells"] += n_nan
            report["summary"]["columns_affected_nan"] += 1

    return report


def replace_inf(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, int]:
    """
    Заменяет ±inf на NaN в указанных колонках.
    Возвращает (df, кол-во заменённых ячеек).
    """
    total_replaced = 0
    for col in feature_columns:
        if col not in df.columns:
            continue
        mask = np.isinf(df[col])
        n = int(mask.sum())
        if n > 0:
            df.loc[mask, col] = np.nan
            total_replaced += n
    return df, total_replaced


def fit_nan_filler(
    df: pd.DataFrame,
    feature_columns: List[str],
    strategy: str = "median",
) -> Dict[str, float]:
    """
    Вычисляет fill values по train (median/mean/zero).
    Возвращает словарь {col: fill_value}.
    """
    fill_values = {}
    for col in feature_columns:
        if col not in df.columns:
            fill_values[col] = 0.0
            continue
        if strategy == "median":
            val = df[col].median()
        elif strategy == "mean":
            val = df[col].mean()
        elif strategy == "zero":
            val = 0.0
        else:
            raise ValueError(f"Неизвестная стратегия NaN fill: {strategy}")

        fill_values[col] = 0.0 if (val is None or np.isnan(val)) else float(val)
    return fill_values


def apply_nan_filler(
    df: pd.DataFrame,
    feature_columns: List[str],
    fill_values: Dict[str, float],
) -> pd.DataFrame:
    """Применяет сохранённые fill values к DataFrame."""
    for col in feature_columns:
        if col in df.columns and col in fill_values:
            df[col] = df[col].fillna(fill_values[col])
    return df


def fit_scaler(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> object:
    """
    Fit StandardScaler на train. Для gradient boosting не применяется,
    но сохраняется как артефакт для будущих нейросетей.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[feature_columns].values)
    return scaler


def save_preprocessing_artifacts(
    artifacts_dir: Path,
    feature_columns: List[str],
    fill_values: Dict[str, float],
    nan_strategy: str,
    scaler: object,
    inf_nan_report: Dict,
) -> None:
    """Сохраняет все артефакты предобработки для inference."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Feature contract
    contract_path = artifacts_dir / "feature_contract.json"
    contract_path.write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Preprocessing params
    params = {
        "nan_strategy": nan_strategy,
        "fill_values": fill_values,
        "feature_count": len(feature_columns),
    }
    params_path = artifacts_dir / "preprocessing_params.json"
    params_path.write_text(
        json.dumps(params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Scaler
    scaler_path = artifacts_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # Inf/NaN report
    report_path = artifacts_dir / "inf_nan_report.json"
    report_path.write_text(
        json.dumps(inf_nan_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_preprocessing_artifacts(artifacts_dir: Path) -> Dict:
    """Загружает артефакты для inference."""
    contract = json.loads(
        (artifacts_dir / "feature_contract.json").read_text(encoding="utf-8")
    )
    params = json.loads(
        (artifacts_dir / "preprocessing_params.json").read_text(encoding="utf-8")
    )
    scaler = joblib.load(artifacts_dir / "scaler.joblib")
    return {
        "feature_columns": contract,
        "fill_values": params["fill_values"],
        "nan_strategy": params["nan_strategy"],
        "scaler": scaler,
    }


def preprocess_for_inference(
    df: pd.DataFrame,
    artifacts_dir: Path,
) -> pd.DataFrame:
    """
    Полный pipeline предобработки для inference.
    Применяет те же шаги, что при обучении:
    1. Проверяет наличие колонок
    2. inf → nan
    3. nan → fill_values из train
    4. Возвращает DataFrame с нужными колонками
    """
    arts = load_preprocessing_artifacts(artifacts_dir)
    feature_columns = arts["feature_columns"]
    fill_values = arts["fill_values"]

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки для inference: {missing[:10]}")

    result = df[feature_columns].copy()
    result, _ = replace_inf(result, feature_columns)
    result = apply_nan_filler(result, feature_columns, fill_values)

    return result
