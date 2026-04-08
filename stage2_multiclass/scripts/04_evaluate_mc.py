"""
Шаг 4: Финальная оценка на test-выборке.

Строит:
  reports/confusion_matrix.png      — тепловая карта 8×8
  reports/per_class_metrics.png     — bar chart F1 по классам
  models/catboost/metrics_mc.json   — дополняем test-метриками
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR, MODELS_DIR, REPORTS_DIR, FEATURE_CONTRACT_PATH,
    CLASS_NAMES, NUM_CLASSES,
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

with open(FEATURE_CONTRACT_PATH) as f:
    FEATURES: list[str] = json.load(f)

# ── Загрузка модели и теста ───────────────────────────────────
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model(str(MODELS_DIR / "model_mc.cbm"))
print(f"Модель загружена, признаков: {len(model.feature_names_)}")

print("Читаем test_mc.parquet ...")
df_test = pd.read_parquet(DATA_DIR / "test_mc.parquet")
feats = [c for c in FEATURES if c in df_test.columns]
X_test = df_test[feats].values.astype("float32")
y_test = df_test["target_mc"].values.astype(int)
print(f"  {X_test.shape[0]:,} строк")

# ── Предсказания ──────────────────────────────────────────────
print("Инференс...")
proba = model.predict_proba(X_test)          # (N, 8)
y_pred = proba.argmax(axis=1).astype(int)

# ── Метрики ───────────────────────────────────────────────────
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_recall_fscore_support,
)

class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))

print("\n-- Метрики на TEST ---")
print(classification_report(
    y_test, y_pred,
    labels=present_classes,
    target_names=[CLASS_NAMES[i] for i in present_classes],
    zero_division=0
))

acc        = accuracy_score(y_test, y_pred)
macro_f1   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
print(f"Accuracy:    {acc:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

# ── Обновляем metrics_mc.json ─────────────────────────────────
metrics_path = MODELS_DIR / "metrics_mc.json"
with open(metrics_path) as f:
    metrics = json.load(f)

prec, rec, f1, sup = precision_recall_fscore_support(
    y_test, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
)
per_class_test = {}
for i in range(NUM_CLASSES):
    per_class_test[CLASS_NAMES[i]] = {
        "precision": round(float(prec[i]), 4),
        "recall":    round(float(rec[i]), 4),
        "f1":        round(float(f1[i]), 4),
        "support":   int(sup[i]),
    }

metrics["test"] = {
    "accuracy":    round(acc, 6),
    "macro_f1":    round(macro_f1, 6),
    "weighted_f1": round(weighted_f1, 6),
    "per_class":   per_class_test,
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"\nМетрики обновлены >> {metrics_path}")

# ── Confusion matrix heatmap ──────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    # Нормализуем по строкам (recall-oriented)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Абсолютные значения
    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix (absolute)", fontsize=13)
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_yticks(range(NUM_CLASSES))
    axes[0].set_xticklabels(class_labels, rotation=35, ha="right")
    axes[0].set_yticklabels(class_labels)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm[i, j]
            if v > 0:
                color = "white" if v > cm.max() * 0.5 else "black"
                axes[0].text(j, i, f"{v:,}", ha="center", va="center",
                             fontsize=7, color=color)
    fig.colorbar(im0, ax=axes[0])

    # Нормализованные (recall)
    im1 = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (normalized by row = recall)", fontsize=13)
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_yticks(range(NUM_CLASSES))
    axes[1].set_xticklabels(class_labels, rotation=35, ha="right")
    axes[1].set_yticklabels(class_labels)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            if v > 0.005:
                color = "white" if v > 0.5 else "black"
                axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=8, color=color)
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix >> {cm_path}")

    # Per-class F1 bar chart
    f1_vals = [per_class_test[CLASS_NAMES[i]]["f1"] for i in range(NUM_CLASSES)]
    colors = ["#5cc97a" if v >= 0.9 else "#f0bc5e" if v >= 0.75 else "#e86070"
              for v in f1_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(class_labels, f1_vals, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-class F1 Score (test set)", fontsize=13)
    ax.axvline(0.9, color="gray", linestyle="--", linewidth=0.8, label="0.90")
    for bar, val in zip(bars, f1_vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.legend()
    plt.tight_layout()
    f1_path = REPORTS_DIR / "per_class_f1.png"
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class F1 >> {f1_path}")

except ImportError:
    print("matplotlib не найден — графики не построены (pip install matplotlib)")

print("\n[OK] Шаг 4 завершён")
