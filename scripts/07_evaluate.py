"""
Шаг 7: Сравнительная оценка моделей CatBoost vs LightGBM.

Выходы:
  reports/07_comparison_report.md
  reports/07_roc_comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PREPROCESSED_DIR, CATBOOST_DIR, LIGHTGBM_DIR, REPORTS_DIR,
    META_COLUMNS,
)
from src.evaluation import (
    compute_binary_metrics, build_comparison_table,
    plot_roc_curves_comparison,
)


def load_model_results(model_dir: Path):
    """Загружает метрики и предсказания модели."""
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    val_proba = np.load(model_dir / "val_proba.npy")
    test_proba = np.load(model_dir / "test_proba.npy")
    return metrics, val_proba, test_proba


def per_class_analysis(y_true, y_pred_proba, threshold, class_labels):
    """Метрики по каждому attack family в test."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    results = {}
    for cls in sorted(class_labels.unique()):
        mask = class_labels == cls
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        n = int(mask.sum())
        # Для binary: accuracy в этой подгруппе
        correct = int((yt == yp).sum())
        results[cls] = {
            "count": n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n > 0 else 0,
        }
        # Если класс полностью attack (target_binary=1), считаем recall
        if yt.sum() == n:
            tp = int(yp.sum())
            results[cls]["recall"] = round(tp / n, 4) if n > 0 else 0
        # Если класс benign (target_binary=0), считаем specificity
        elif yt.sum() == 0:
            tn = int((yp == 0).sum())
            results[cls]["specificity"] = round(tn / n, 4) if n > 0 else 0
    return results


def run_evaluate():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Загрузка результатов моделей...")

    models = {}
    for name, model_dir in [("CatBoost", CATBOOST_DIR), ("LightGBM", LIGHTGBM_DIR)]:
        if not (model_dir / "metrics.json").exists():
            print(f"  [ПРОПУСК] {name}: metrics.json не найден в {model_dir}")
            continue
        metrics, val_proba, test_proba = load_model_results(model_dir)
        models[name] = {
            "metrics": metrics,
            "val_proba": val_proba,
            "test_proba": test_proba,
        }
        print(f"  {name}: загружен (test F1={metrics['test']['f1']:.4f})")

    if not models:
        print("[ОШИБКА] Нет ни одной обученной модели!")
        sys.exit(1)

    # Загружаем y_test и мета для per-class analysis
    y_test_df = pd.read_parquet(PREPROCESSED_DIR / "test_y.parquet")
    y_test = y_test_df["target_binary"].values
    test_classes = y_test_df["source_class_folder"] if "source_class_folder" in y_test_df.columns else None

    y_val_df = pd.read_parquet(PREPROCESSED_DIR / "val_y.parquet")
    y_val = y_val_df["target_binary"].values

    print("[2/3] Построение сравнительной таблицы...")

    # Test metrics comparison
    test_metrics_dict = {}
    for name, data in models.items():
        test_metrics_dict[name] = data["metrics"]["test"]

    comparison_table = build_comparison_table(test_metrics_dict)

    # Per-class analysis
    per_class = {}
    if test_classes is not None:
        for name, data in models.items():
            threshold = data["metrics"]["best_threshold"]
            per_class[name] = per_class_analysis(
                y_test, data["test_proba"], threshold, test_classes
            )

    print("[3/3] Сохранение отчёта...")

    # ROC comparison plot
    if len(models) >= 2:
        roc_data = []
        for name, data in models.items():
            roc_data.append((name, y_test, data["test_proba"]))
        plot_roc_curves_comparison(roc_data, REPORTS_DIR / "07_roc_comparison.png")

    # Markdown report
    lines = ["# Сравнение моделей — Stage 1 Binary Classification\n"]

    lines.append("## Сводная таблица (Test set)\n")
    lines.append(comparison_table)

    # Training info
    lines.append("\n## Информация об обучении\n")
    lines.append("| Параметр | " + " | ".join(models.keys()) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(models)) + "|")
    for param in ["best_iteration", "train_time_sec", "features_used"]:
        row = f"| {param} |"
        for name, data in models.items():
            val = data["metrics"].get(param, "—")
            row += f" {val} |"
        lines.append(row)

    # Val metrics
    lines.append("\n## Метрики на Validation\n")
    val_metrics_dict = {name: data["metrics"]["val"] for name, data in models.items()}
    lines.append(build_comparison_table(val_metrics_dict))

    # Per-class analysis
    if per_class:
        lines.append("\n## Per-class analysis (Test set)\n")
        for model_name, cls_data in per_class.items():
            lines.append(f"\n### {model_name}\n")
            lines.append("| Класс | Строк | Accuracy | Recall/Specificity |")
            lines.append("|---|---:|---:|---:|")
            for cls, info in sorted(cls_data.items()):
                extra = ""
                if "recall" in info:
                    extra = f"{info['recall']:.4f} (recall)"
                elif "specificity" in info:
                    extra = f"{info['specificity']:.4f} (spec.)"
                lines.append(f"| {cls} | {info['count']:,} | {info['accuracy']:.4f} | {extra} |")

    # Вывод
    if len(models) >= 2:
        # Определяем лучшую модель
        best_name = max(test_metrics_dict.keys(), key=lambda k: test_metrics_dict[k]["f1"])
        lines.append(f"\n## Вывод\n")
        lines.append(f"Лучшая модель по F1 на test: **{best_name}** "
                      f"(F1={test_metrics_dict[best_name]['f1']:.4f})")

    md_path = REPORTS_DIR / "07_comparison_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nГотово: {md_path}")
    for name, data in models.items():
        m = data["metrics"]["test"]
        print(f"  {name}: F1={m['f1']:.4f}, ROC-AUC={m['roc_auc']:.4f}")


if __name__ == "__main__":
    run_evaluate()
