"""
Оценка модели (CIC IoT-DIAD 2024) на датасете CIC-IDS 2018.

Запуск из корня stage1_v2_cl:
    python eval_external/run_eval_cic2018.py

Датасет: G:\Диплом\CSE-CIC-IDS2018\  (10 Parquet-файлов, ~696 MB)
Модель:  models/catboost/model.cbm
Порог:   models/catboost/best_threshold.json  (0.70)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Пути
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess_for_inference
from eval_external.feature_map import CIC2018_MAP, CIC2018_LABEL_COL, CIC2018_BENIGN_VALUE, align_columns

ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH    = ROOT / "models" / "catboost" / "model.cbm"
THRESH_PATH   = ROOT / "models" / "catboost" / "best_threshold.json"
DATA_DIR      = Path(r"G:\Диплом\CSE-CIC-IDS2018")
RESULTS_DIR   = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Загрузка feature contract
# --------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = json.loads(
    (ARTIFACTS_DIR / "feature_contract.json").read_text(encoding="utf-8")
)


def load_threshold() -> float:
    data = json.loads(THRESH_PATH.read_text(encoding="utf-8"))
    # формат: {"threshold": 0.70, ...} или просто число
    if isinstance(data, dict):
        return float(data.get("threshold", data.get("best_threshold", 0.5)))
    return float(data)


def load_cic2018() -> tuple[pd.DataFrame, pd.Series]:
    """
    Загружает все 10 Parquet-файлов, выравнивает колонки, возвращает (X, y).
    y: 0 = Benign, 1 = Attack
    Также возвращает оригинальные метки для breakdown.
    """
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Parquet-файлы не найдены в {DATA_DIR}")

    print(f"Найдено файлов: {len(parquet_files)}")
    chunks = []

    for fp in parquet_files:
        print(f"  Загружаю {fp.name} ...", end=" ", flush=True)
        t0 = time.time()
        df = pd.read_parquet(fp)
        print(f"{len(df):,} строк, {time.time()-t0:.1f}s")
        chunks.append(df)

    full = pd.concat(chunks, ignore_index=True)
    print(f"\nВсего строк: {len(full):,}")

    # Сохраняем оригинальные метки до выравнивания
    raw_labels: pd.Series = full[CIC2018_LABEL_COL].copy()

    # Выравниваем признаки
    X = align_columns(full, CIC2018_MAP, FEATURE_COLUMNS)

    # Бинарная метка
    y = (raw_labels != CIC2018_BENIGN_VALUE).astype(int)
    y.index = X.index

    return X, y, raw_labels


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, proba)
    cm   = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "="*60)
    print("ИТОГОВЫЕ МЕТРИКИ (CIC-IDS 2018, бинарная классификация)")
    print("="*60)
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Precision      : {prec:.4f}  (из предсказанных атак — реальные)")
    print(f"  Recall (TPR)   : {rec:.4f}  (из реальных атак — обнаружено)")
    print(f"  Specificity    : {specificity:.4f}  (из реального трафика — верно=Benign)")
    print(f"  F1-score       : {f1:.4f}")
    print(f"  ROC-AUC        : {auc:.4f}")
    print(f"  FP-rate        : {fpr:.4f}  ({fp:,} ложных тревог из {tn+fp:,} benign)")
    print(f"\n  Матрица ошибок:")
    print(f"              Предсказано Benign | Предсказано Attack")
    print(f"  Реально Benign  {tn:>12,}   |   {fp:>12,}")
    print(f"  Реально Attack  {fn:>12,}   |   {tp:>12,}")

    return dict(
        accuracy=round(acc, 4), precision=round(prec, 4), recall=round(rec, 4),
        specificity=round(specificity, 4), f1=round(f1, 4), auc=round(auc, 4),
        fpr=round(fpr, 4), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    )


def print_per_class(raw_labels: pd.Series, y_pred: np.ndarray) -> list[dict]:
    """Breakdown recall по каждому типу атаки из оригинальных меток."""
    labels_arr = raw_labels.values
    pred_arr   = np.array(y_pred)

    classes = sorted(set(labels_arr))
    rows = []

    print("\n" + "="*60)
    print("ДЕТЕКЦИЯ ПО ТИПАМ ТРАФИКА")
    print(f"{'Класс':<30} {'Записей':>10} {'Обнаружено':>12} {'Пропущено':>10} {'Recall/Spec':>12}")
    print("-"*60)

    for cls in classes:
        mask = labels_arr == cls
        total = int(mask.sum())
        if total == 0:
            continue

        if cls == CIC2018_BENIGN_VALUE:
            # Для Benign — specificity (сколько верно определили как Benign)
            correct = int(((mask) & (pred_arr == 0)).sum())
            missed  = total - correct
            metric_val = correct / total
            label = "Specificity"
        else:
            # Для атак — recall (сколько обнаружено)
            correct = int(((mask) & (pred_arr == 1)).sum())
            missed  = total - correct
            metric_val = correct / total
            label = "Recall"

        symbol = "⚠" if metric_val < 0.7 else " "
        print(f"  {symbol} {cls:<28} {total:>10,} {correct:>12,} {missed:>10,} {metric_val:>11.3f}")
        rows.append(dict(
            class_name=cls, total=total, detected=correct,
            missed=missed, metric=round(metric_val, 4), metric_type=label,
        ))

    return rows


def main():
    print("=" * 60)
    print("Оценка модели IoT-IDS на CIC-IDS 2018")
    print("=" * 60)

    # 1. Загрузка модели
    if not MODEL_PATH.exists():
        print(f"[ОШИБКА] Модель не найдена: {MODEL_PATH}")
        sys.exit(1)

    print(f"\nМодель: {MODEL_PATH}")
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    threshold = load_threshold()
    print(f"Порог:  {threshold}")

    # 2. Загрузка данных
    print(f"\nДанные: {DATA_DIR}")
    t_load = time.time()
    X, y_true, raw_labels = load_cic2018()
    print(f"Загрузка: {time.time()-t_load:.1f}s")
    print(f"Benign: {(y_true==0).sum():,} | Attack: {(y_true==1).sum():,}")

    # 3. Предобработка
    print("\nПредобработка (inf→NaN→fill медианами)...")
    t_prep = time.time()
    X_clean = preprocess_for_inference(X, ARTIFACTS_DIR)
    print(f"  Готово за {time.time()-t_prep:.1f}s")

    # 4. Предсказание
    print("\nПредсказание...")
    t_pred = time.time()
    proba = model.predict_proba(X_clean.values)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    print(f"  Готово за {time.time()-t_pred:.1f}s")

    # 5. Метрики
    metrics = print_metrics(y_true.values, y_pred, proba)
    per_class = print_per_class(raw_labels, y_pred)

    # 6. Сохранение
    result = {
        "dataset": "CIC-IDS 2018",
        "model": "CatBoost (IoT-DIAD 2024)",
        "threshold": threshold,
        "total_samples": len(y_true),
        "metrics": metrics,
        "per_class": per_class,
    }
    out_path = RESULTS_DIR / "cic2018_eval.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультат сохранён: {out_path}")


if __name__ == "__main__":
    main()
