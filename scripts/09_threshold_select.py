"""
Шаг 9: Выбор оптимального threshold для CatBoost.

Строит полную таблицу метрик по всем threshold,
сохраняет выбранный threshold и обновляет metrics.json.

Запуск:
  python scripts/09_threshold_select.py            # показать таблицу
  python scripts/09_threshold_select.py --set 0.7  # установить threshold 0.7

Выходы:
  reports/09_threshold_report.md
  reports/09_threshold_table.csv
  models/catboost/best_threshold.json  (обновляется)
  models/catboost/metrics.json         (обновляется)
  models/catboost/confusion_matrix.png (обновляется)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PREPROCESSED_DIR, CATBOOST_DIR, REPORTS_DIR
from src.evaluation import plot_confusion_matrix


def load_data():
    y_proba_val = np.load(CATBOOST_DIR / "val_proba.npy")
    y_proba_test = np.load(CATBOOST_DIR / "test_proba.npy")
    y_val = pd.read_parquet(PREPROCESSED_DIR / "val_y.parquet")["target_binary"].values
    y_test = pd.read_parquet(PREPROCESSED_DIR / "test_y.parquet")["target_binary"].values
    return y_proba_val, y_proba_test, y_val, y_test


def compute_row(y_true, y_proba, t):
    yp = (y_proba >= t).astype(int)
    n_benign = (y_true == 0).sum()
    fp = int(((yp == 1) & (y_true == 0)).sum())
    fn = int(((yp == 0) & (y_true == 1)).sum())
    return {
        "threshold": round(t, 2),
        "accuracy":  round(float(accuracy_score(y_true, yp)), 4),
        "precision": round(float(precision_score(y_true, yp, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, yp, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, yp, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_proba)), 4),
        "fp_count":  fp,
        "fp_pct":    round(100.0 * fp / n_benign, 1) if n_benign > 0 else 0.0,
        "fn_count":  fn,
    }


def build_table(y_true, y_proba, thresholds):
    return [compute_row(y_true, y_proba, t) for t in thresholds]


def print_table(rows, title=""):
    if title:
        print(f"\n{title}")
    print(f"{'thr':>5} | {'precision':>9} | {'recall':>6} | {'f1':>6} | {'FP на benign':>14} | {'FN пропущено':>14}")
    print("-" * 75)
    for r in rows:
        print(
            f"{r['threshold']:>5.2f} | "
            f"{r['precision']:>9.4f} | "
            f"{r['recall']:>6.4f} | "
            f"{r['f1']:>6.4f} | "
            f"{r['fp_count']:>7,} ({r['fp_pct']:>4.1f}%) | "
            f"{r['fn_count']:>7,}"
        )


def set_threshold(threshold: float, y_val, y_proba_val, y_test, y_proba_test):
    """Обновляет best_threshold.json и metrics.json для выбранного threshold."""

    def full_metrics(y_true, y_proba, t):
        yp = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, yp)
        n_benign = int((y_true == 0).sum())
        fp = int(cm[0, 1])
        return {
            "threshold": round(t, 4),
            "accuracy":  round(float(accuracy_score(y_true, yp)), 6),
            "precision": round(float(precision_score(y_true, yp, zero_division=0)), 6),
            "recall":    round(float(recall_score(y_true, yp, zero_division=0)), 6),
            "f1":        round(float(f1_score(y_true, yp, zero_division=0)), 6),
            "roc_auc":   round(float(roc_auc_score(y_true, y_proba)), 6),
            "pr_auc":    round(float(average_precision_score(y_true, y_proba)), 6),
            "confusion_matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "fp_pct_benign": round(100.0 * fp / n_benign, 2) if n_benign > 0 else 0,
        }

    val_m = full_metrics(y_val, y_proba_val, threshold)
    test_m = full_metrics(y_test, y_proba_test, threshold)

    # Обновляем best_threshold.json
    thresh_path = CATBOOST_DIR / "best_threshold.json"
    thresh_path.write_text(
        json.dumps({"best_threshold": threshold, "metric": "manual_fp_optimized"}, indent=2),
        encoding="utf-8",
    )

    # Обновляем metrics.json
    metrics_path = CATBOOST_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {}
    metrics["best_threshold"] = threshold
    metrics["val"] = val_m
    metrics["test"] = test_m
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Обновляем confusion matrix plot
    cm_arr = np.array(test_m["confusion_matrix"])
    plot_confusion_matrix(
        cm_arr, CATBOOST_DIR / "confusion_matrix.png",
        title=f"CatBoost — Test (threshold={threshold})"
    )

    print(f"\n[OK] Threshold обновлён: {threshold}")
    print(f"\n=== Val (thr={threshold}) ===")
    print(f"  precision: {val_m['precision']:.4f}  recall: {val_m['recall']:.4f}  "
          f"f1: {val_m['f1']:.4f}  FP benign: {val_m['fp']:,} ({val_m['fp_pct_benign']:.1f}%)")
    print(f"\n=== Test (thr={threshold}) ===")
    print(f"  precision: {test_m['precision']:.4f}  recall: {test_m['recall']:.4f}  "
          f"f1: {test_m['f1']:.4f}  FP benign: {test_m['fp']:,} ({test_m['fp_pct_benign']:.1f}%)")
    print(f"\n  Confusion matrix (test):")
    print(f"    TN={test_m['tn']:,}  FP={test_m['fp']:,}")
    print(f"    FN={test_m['fn']:,}  TP={test_m['tp']:,}")

    return val_m, test_m


def save_report(val_rows, test_rows, chosen_threshold=None):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV таблица (test)
    df = pd.DataFrame(test_rows)
    df.to_csv(REPORTS_DIR / "09_threshold_table.csv", index=False, encoding="utf-8-sig")

    # Markdown
    lines = ["# Анализ threshold — CatBoost\n"]
    if chosen_threshold:
        lines.append(f"**Выбранный threshold: {chosen_threshold}**\n")

    lines.append("## Test set — полная таблица\n")
    lines.append("| thr | precision | recall | f1 | FP на benign | FN пропущено |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in test_rows:
        marker = " ←" if chosen_threshold and abs(r["threshold"] - chosen_threshold) < 0.001 else ""
        lines.append(
            f"| {r['threshold']:.2f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | {r['fp_count']:,} ({r['fp_pct']:.1f}%){marker} | {r['fn_count']:,} |"
        )

    lines.append("\n## Пояснение\n")
    lines.append("- **FP на benign** — сколько нормальных потоков ошибочно помечено как атака.")
    lines.append("- **FN пропущено** — сколько реальных атак не обнаружено.")
    lines.append("- Чем выше threshold → меньше FP, но больше FN (пропущенных атак).")
    lines.append("- Для IDS обычно важнее низкий FP (меньше ложных тревог) → выбирай threshold 0.6–0.8.")

    (REPORTS_DIR / "09_threshold_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(set_thr=None):
    print("Загрузка данных...")
    y_proba_val, y_proba_test, y_val, y_test = load_data()

    thresholds = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]

    val_rows  = build_table(y_val,  y_proba_val,  thresholds)
    test_rows = build_table(y_test, y_proba_test, thresholds)

    print_table(test_rows, title="Test set — метрики по threshold:")

    # Если задан --set, обновляем
    if set_thr is not None:
        val_m, test_m = set_threshold(set_thr, y_val, y_proba_val, y_test, y_proba_test)

    save_report(val_rows, test_rows, chosen_threshold=set_thr)
    print(f"\nОтчёт: {REPORTS_DIR / '09_threshold_report.md'}")
    print(f"CSV:    {REPORTS_DIR / '09_threshold_table.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set", type=float, default=None,
        help="Установить этот threshold (напр. --set 0.7)"
    )
    args = parser.parse_args()
    run(set_thr=args.set)
