"""
Шаг 4: Полная оценка Stage4 на test set + сравнение со Stage3.

Оценивает на:
  - stage3_cic2023/data/test_X.parquet (IoT 2023 тест — проверяем не деградировали ли)
  - stage4_extended/data/cic2018_mapped.parquet (CIC-IDS-2018 — новые данные)

Сохраняет:
  - stage4_extended/reports/comparison_stage3_vs_stage4.json
  - stage4_extended/reports/confusion_matrix.png
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    STAGE3_ROOT, DATA_DIR, MODELS_DIR, REPORTS_DIR,
    FEATURES_2023, CLASS_NAMES, NUM_CLASSES,
)

print("=" * 60)
print("Stage4 / Шаг 4: Оценка и сравнение")
print("=" * 60)

from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report, f1_score,
    confusion_matrix, precision_recall_fscore_support
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Загружаем Stage4
print("\nЗагружаем Stage4 модель...")
stage4 = CatBoostClassifier()
stage4.load_model(str(MODELS_DIR / "model_mc.cbm"))

# Загружаем Stage3 для сравнения
print("Загружаем Stage3 модель...")
stage3 = CatBoostClassifier()
stage3.load_model(str(STAGE3_ROOT / "models" / "catboost" / "model_mc.cbm"))


def evaluate(model, X, y, name: str) -> dict:
    y_pred = model.predict(X).flatten().astype(int)
    present_labels = sorted(set(y.tolist()) | set(y_pred.tolist()))
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0, labels=present_labels)
    p, r, f, s = precision_recall_fscore_support(
        y, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    print(f"\n[{name}] Macro F1 = {macro_f1:.4f}")
    report = classification_report(
        y, y_pred,
        labels=present_labels,
        target_names=[CLASS_NAMES[i] for i in present_labels],
        zero_division=0
    )
    print(report)
    return {
        "macro_f1": float(macro_f1),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])
            } for i in range(NUM_CLASSES)
        },
        "y_pred": y_pred.tolist(),
    }


# ── Тест 1: IoT 2023 test set (не должны деградировать) ───────
print("\n" + "=" * 40)
print("Тест A: CIC IoT 2023 test set")
X_test_iot = pd.read_parquet(STAGE3_ROOT / "data" / "test_X.parquet")[FEATURES_2023]
y_test_iot = pd.read_parquet(STAGE3_ROOT / "data" / "test_y.parquet").iloc[:, 0]

res_stage3_iot = evaluate(stage3, X_test_iot, y_test_iot, "Stage3 на IoT test")
res_stage4_iot = evaluate(stage4, X_test_iot, y_test_iot, "Stage4 на IoT test")

# ── Тест 2: CIC-IDS-2018 (новые данные) ───────────────────────
print("\n" + "=" * 40)
print("Тест B: CIC-IDS-2018 (новые данные, не видел при обучении)")
df_2018 = pd.read_parquet(DATA_DIR / "cic2018_mapped.parquet")
# Используем только тестовую часть (20%) для честной оценки
df_2018_test = df_2018.sample(frac=0.20, random_state=42)
X_test_2018 = df_2018_test[FEATURES_2023]
y_test_2018 = df_2018_test["target"].astype(int)

res_stage3_2018 = evaluate(stage3, X_test_2018, y_test_2018, "Stage3 на CIC-IDS-2018")
res_stage4_2018 = evaluate(stage4, X_test_2018, y_test_2018, "Stage4 на CIC-IDS-2018")

# ── Сравнение ─────────────────────────────────────────────────
print("\n" + "=" * 40)
print("СРАВНЕНИЕ Stage3 vs Stage4")
print(f"{'':20s}  {'Stage3':>10}  {'Stage4':>10}  {'Δ':>8}")
print("-" * 52)

def show_comparison(s3_res, s4_res, dataset_name: str):
    print(f"\n  {dataset_name}")
    s3_f1 = s3_res["macro_f1"]
    s4_f1 = s4_res["macro_f1"]
    delta  = s4_f1 - s3_f1
    sign   = "+" if delta >= 0 else ""
    print(f"  {'Macro F1':20s}: {s3_f1:10.4f}  {s4_f1:10.4f}  {sign}{delta:8.4f}")
    for cls_name in CLASS_NAMES.values():
        s3_cls = s3_res["per_class"].get(cls_name, {})
        s4_cls = s4_res["per_class"].get(cls_name, {})
        if s3_cls.get("support", 0) == 0 and s4_cls.get("support", 0) == 0:
            continue
        f3 = s3_cls.get("f1", 0.0)
        f4 = s4_cls.get("f1", 0.0)
        d  = f4 - f3
        sign = "+" if d >= 0 else ""
        print(f"  {cls_name:20s}: {f3:10.4f}  {f4:10.4f}  {sign}{d:8.4f}")

show_comparison(res_stage3_iot,  res_stage4_iot,  "IoT 2023 test")
show_comparison(res_stage3_2018, res_stage4_2018, "CIC-IDS-2018 test")

# ── Confusion matrix Stage4 на IoT test ───────────────────────
y_pred_cm = np.array(res_stage4_iot["y_pred"])
cm = confusion_matrix(y_test_iot, y_pred_cm, labels=list(range(NUM_CLASSES)))
names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=names, yticklabels=names,
            cmap="Blues", ax=ax)
ax.set_xlabel("Предсказано")
ax.set_ylabel("Истинно")
ax.set_title(f"Stage4 — Confusion Matrix (IoT 2023 test)\nMacro F1 = {res_stage4_iot['macro_f1']:.4f}")
plt.tight_layout()
cm_path = REPORTS_DIR / "confusion_matrix.png"
fig.savefig(str(cm_path), dpi=150)
print(f"\nConfusion matrix: {cm_path}")

# ── Сохранение результатов ─────────────────────────────────────
comparison = {
    "iot2023_test": {
        "stage3_macro_f1": res_stage3_iot["macro_f1"],
        "stage4_macro_f1": res_stage4_iot["macro_f1"],
        "delta": res_stage4_iot["macro_f1"] - res_stage3_iot["macro_f1"],
    },
    "cic2018_test": {
        "stage3_macro_f1": res_stage3_2018["macro_f1"],
        "stage4_macro_f1": res_stage4_2018["macro_f1"],
        "delta": res_stage4_2018["macro_f1"] - res_stage3_2018["macro_f1"],
    },
    "stage3_per_class_iot":   res_stage3_iot["per_class"],
    "stage4_per_class_iot":   res_stage4_iot["per_class"],
    "stage3_per_class_2018":  res_stage3_2018["per_class"],
    "stage4_per_class_2018":  res_stage4_2018["per_class"],
}
out_path = REPORTS_DIR / "comparison_stage3_vs_stage4.json"
with open(out_path, "w") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)
print(f"Сравнение: {out_path}")

print("\n[OK] Шаг 4 завершён")
