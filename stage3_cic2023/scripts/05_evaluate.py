"""
Шаг 5: Финальная оценка на test-выборке.

Вывод:
  reports/confusion_matrix.png
  reports/per_class_f1.png
  models/catboost/metrics_mc.json  (дополнение test-секцией)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, MODELS_DIR, REPORTS_DIR, CLASS_NAMES, NUM_CLASSES

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from catboost import CatBoostClassifier
model = CatBoostClassifier()
model.load_model(str(MODELS_DIR / "model_mc.cbm"))

print("Загрузка test...")
X_test = pd.read_parquet(DATA_DIR / "test_X.parquet").values.astype("float32")
y_test = pd.read_parquet(DATA_DIR / "test_y.parquet")["target_mc"].values.astype(int)
print(f"  {X_test.shape[0]:,} строк")

print("Инференс...")
proba  = model.predict_proba(X_test)
y_pred = proba.argmax(axis=1).astype(int)

from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
)
class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
present = sorted(np.unique(np.concatenate([y_test, y_pred])))

print("\n-- Метрики на TEST ---")
print(classification_report(
    y_test, y_pred,
    labels=present,
    target_names=[CLASS_NAMES[i] for i in present],
    zero_division=0,
))

acc        = accuracy_score(y_test, y_pred)
macro_f1   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
print(f"Accuracy:    {acc:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

prec, rec, f1, sup = precision_recall_fscore_support(
    y_test, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
)
per_class_test = {
    CLASS_NAMES[i]: {
        "precision": round(float(prec[i]), 4),
        "recall":    round(float(rec[i]), 4),
        "f1":        round(float(f1[i]), 4),
        "support":   int(sup[i]),
    }
    for i in range(NUM_CLASSES)
}

with open(MODELS_DIR / "metrics_mc.json") as f:
    metrics = json.load(f)
metrics["test"] = {
    "accuracy":    round(acc, 6),
    "macro_f1":    round(macro_f1, 6),
    "weighted_f1": round(weighted_f1, 6),
    "per_class":   per_class_test,
}
with open(MODELS_DIR / "metrics_mc.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# ── Confusion matrix ──────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, title, fmt in [
        (axes[0], cm,      "Confusion Matrix (absolute)",           "{:,}"),
        (axes[1], cm_norm, "Confusion Matrix (recall, row-normalized)", "{:.2f}"),
    ]:
        im = ax.imshow(data, cmap="Blues",
                       vmin=0, vmax=(1 if "norm" in title.lower() else None))
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(class_labels, rotation=35, ha="right")
        ax.set_yticklabels(class_labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        threshold = data.max() * 0.5
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v = data[i, j]
                if v > (0.005 if "norm" in title.lower() else 0):
                    color = "white" if v > threshold else "black"
                    ax.text(j, i, fmt.format(v),
                            ha="center", va="center", fontsize=7.5, color=color)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    out = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix >> {out}")

    # Per-class F1 bar chart
    f1_vals = [per_class_test[CLASS_NAMES[i]]["f1"] for i in range(NUM_CLASSES)]
    colors  = ["#5cc97a" if v >= 0.90 else "#f0bc5e" if v >= 0.70 else "#e86070"
               for v in f1_vals]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(class_labels, f1_vals, color=colors)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-class F1 Score — test set (CIC IoT 2023 model)", fontsize=12)
    ax.axvline(0.90, color="gray", linestyle="--", linewidth=0.8, label="0.90")
    for bar, val in zip(bars, f1_vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.legend()
    plt.tight_layout()
    out2 = REPORTS_DIR / "per_class_f1.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class F1 >> {out2}")

except ImportError:
    print("matplotlib не найден")

print("\n[OK] Шаг 5 завершён")
