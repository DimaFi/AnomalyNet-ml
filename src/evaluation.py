"""
Метрики, графики, сравнение моделей.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
) -> Dict:
    """Рассчитывает все бинарные метрики для заданного threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": round(threshold, 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_pred_proba)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_pred_proba)), 6),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def search_best_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
    t_min: float = 0.05,
    t_max: float = 0.95,
    t_step: float = 0.01,
) -> Tuple[float, List[Dict]]:
    """
    Grid search по threshold, оптимизация по metric (f1/recall/precision).
    Возвращает (лучший threshold, полная таблица поиска).
    """
    thresholds = np.arange(t_min, t_max + t_step / 2, t_step)
    results = []

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        row = {
            "threshold": round(float(t), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        }
        results.append(row)

    best = max(results, key=lambda r: r[metric])
    return best["threshold"], results


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Path,
    label: str = "Model",
) -> None:
    """Сохраняет ROC-кривую в PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curves_comparison(
    models_data: List[Tuple[str, np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    """ROC-кривые нескольких моделей на одном графике."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_true, y_proba in models_data:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str = "") -> None:
    """Сохраняет confusion matrix в PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Benign", "Attack"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or "Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    importances: pd.Series,
    save_path: Path,
    top_n: int = 25,
    title: str = "Feature Importance",
) -> None:
    """Сохраняет bar chart важности фичей в PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importances.nlargest(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    ax.barh(top.index, top.values)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def build_comparison_table(models_metrics: Dict[str, Dict]) -> str:
    """Markdown-таблица сравнения моделей."""
    lines = [
        "| Метрика | " + " | ".join(models_metrics.keys()) + " |",
        "|---|" + "|".join(["---:"] * len(models_metrics)) + "|",
    ]
    metric_names = ["threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    for m in metric_names:
        row = f"| {m} |"
        for model_name, metrics in models_metrics.items():
            val = metrics.get(m, "—")
            if isinstance(val, float):
                val = f"{val:.4f}"
            row += f" {val} |"
        lines.append(row)
    return "\n".join(lines)
