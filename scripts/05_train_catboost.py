"""
Шаг 5: Обучение CatBoost (основная модель).

Выходы:
  models/catboost/model.cbm
  models/catboost/metrics.json
  models/catboost/best_threshold.json
  models/catboost/feature_importance.json
  models/catboost/roc_curve.png
  models/catboost/confusion_matrix.png
  models/catboost/feature_importance.png
  reports/05_catboost_report.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PREPROCESSED_DIR, CATBOOST_DIR, REPORTS_DIR,
    CATBOOST_PARAMS, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP,
    SEED, MAX_RAM_GB,
)
from src.evaluation import (
    compute_binary_metrics, search_best_threshold,
    plot_roc_curve, plot_confusion_matrix, plot_feature_importance,
)


def load_xy(name: str):
    X = pd.read_parquet(PREPROCESSED_DIR / f"{name}_X.parquet")
    y_df = pd.read_parquet(PREPROCESSED_DIR / f"{name}_y.parquet")
    y = y_df["target_binary"].values
    return X, y, y_df


def drop_constant_features(X_train, X_val, X_test):
    """Убираем фичи с нулевой дисперсией на train."""
    nunique = X_train.nunique()
    constant = nunique[nunique <= 1].index.tolist()
    if constant:
        print(f"  Константные фичи ({len(constant)}): {constant}")
        X_train = X_train.drop(columns=constant)
        X_val = X_val.drop(columns=constant)
        X_test = X_test.drop(columns=constant)
    return X_train, X_val, X_test, constant


def run_train_catboost():
    CATBOOST_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Загрузка данных...")
    X_train, y_train, _ = load_xy("train")
    X_val, y_val, _ = load_xy("val")
    X_test, y_test, y_test_df = load_xy("test")
    print(f"  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # Проверка RAM
    mem_gb = (X_train.memory_usage(deep=True).sum() +
              X_val.memory_usage(deep=True).sum() +
              X_test.memory_usage(deep=True).sum()) / (1024**3)
    print(f"  RAM данных: {mem_gb:.2f} GB (лимит: {MAX_RAM_GB} GB)")

    print("[2/5] Подготовка фичей...")
    X_train, X_val, X_test, constant_features = drop_constant_features(X_train, X_val, X_test)
    feature_names = X_train.columns.tolist()
    print(f"  Активных фичей: {len(feature_names)}")

    print("[3/5] Обучение CatBoost...")
    from catboost import CatBoostClassifier, Pool

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    params = dict(CATBOOST_PARAMS)
    early_stop = params.pop("early_stopping_rounds", 200)

    model = CatBoostClassifier(**params)

    t0 = time.time()
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=early_stop,
        use_best_model=True,
    )
    train_time = time.time() - t0
    print(f"  Время обучения: {train_time:.1f} сек")
    print(f"  Best iteration: {model.get_best_iteration()}")

    # Сохранение модели
    model.save_model(str(CATBOOST_DIR / "model.cbm"))

    # Предсказания
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    print("[4/5] Подбор threshold на val...")
    best_threshold, threshold_table = search_best_threshold(
        y_val, val_proba, metric="f1",
        t_min=THRESHOLD_MIN, t_max=THRESHOLD_MAX, t_step=THRESHOLD_STEP,
    )
    print(f"  Лучший threshold: {best_threshold}")

    # Метрики
    val_metrics = compute_binary_metrics(y_val, val_proba, best_threshold)
    test_metrics = compute_binary_metrics(y_test, test_proba, best_threshold)

    print(f"  Val  — F1: {val_metrics['f1']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"  Test — F1: {test_metrics['f1']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Feature importance
    fi = pd.Series(model.get_feature_importance(), index=feature_names).sort_values(ascending=False)

    print("[5/5] Сохранение артефактов...")

    # Metrics
    metrics_out = {
        "model": "CatBoost",
        "best_iteration": model.get_best_iteration(),
        "train_time_sec": round(train_time, 1),
        "best_threshold": best_threshold,
        "features_used": len(feature_names),
        "constant_features_dropped": constant_features,
        "val": val_metrics,
        "test": test_metrics,
    }
    (CATBOOST_DIR / "metrics.json").write_text(
        json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Threshold
    (CATBOOST_DIR / "best_threshold.json").write_text(
        json.dumps({"best_threshold": best_threshold, "metric": "f1"}, indent=2),
        encoding="utf-8",
    )

    # Feature importance
    fi_dict = {k: round(float(v), 6) for k, v in fi.items()}
    (CATBOOST_DIR / "feature_importance.json").write_text(
        json.dumps(fi_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Графики
    plot_roc_curve(y_test, test_proba, CATBOOST_DIR / "roc_curve.png", label="CatBoost")
    cm = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(cm, CATBOOST_DIR / "confusion_matrix.png", title="CatBoost — Test")
    plot_feature_importance(fi, CATBOOST_DIR / "feature_importance.png", top_n=25, title="CatBoost Feature Importance")

    # Сохраняем предсказания для 07_evaluate
    np.save(CATBOOST_DIR / "val_proba.npy", val_proba)
    np.save(CATBOOST_DIR / "test_proba.npy", test_proba)

    # Markdown report
    lines = ["# CatBoost — отчёт\n"]
    lines.append(f"- Best iteration: **{model.get_best_iteration()}**")
    lines.append(f"- Время обучения: **{train_time:.1f} сек**")
    lines.append(f"- Threshold: **{best_threshold}**")
    lines.append(f"- Активных фичей: **{len(feature_names)}**\n")

    lines.append("## Метрики\n")
    lines.append("| Метрика | Val | Test |")
    lines.append("|---|---:|---:|")
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        lines.append(f"| {m} | {val_metrics[m]:.4f} | {test_metrics[m]:.4f} |")

    lines.append("\n## Top-20 фичей\n")
    lines.append("| # | Фича | Importance |")
    lines.append("|---:|---|---:|")
    for i, (feat, imp) in enumerate(fi.head(20).items(), 1):
        lines.append(f"| {i} | {feat} | {imp:.4f} |")

    md_path = REPORTS_DIR / "05_catboost_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nГотово: {CATBOOST_DIR}")


if __name__ == "__main__":
    run_train_catboost()
