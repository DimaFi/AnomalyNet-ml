"""
Шаг 8: Финальная валидация всего pipeline.

Проверки:
  1. Feature contract: нет дупликатов, нет запрещённых колонок
  2. Все фичи присутствуют в preprocessed данных
  3. Мета-колонки присутствуют в splits
  4. Нет пересечения файлов между splits
  5. Одинаковая схема preprocessed X
  6. Нет NaN/Inf в preprocessed X
  7. Train balance проверка
  8. Модели существуют и метрики адекватны
  9. Артефакты preprocessing полные
  10. BruteForce/Recon отсутствуют в val/test (ожидаемо)

Выходы:
  reports/08_audit_report.json
  reports/08_audit_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    SPLITS_DIR, PREPROCESSED_DIR, ARTIFACTS_DIR, CATBOOST_DIR,
    LIGHTGBM_DIR, REPORTS_DIR, META_COLUMNS,
)
from src.io_utils import load_feature_contract


def check(name: str, condition: bool, detail: str = "") -> dict:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": detail}


def run_audit():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print("=" * 60)
    print("ФИНАЛЬНЫЙ АУДИТ PIPELINE")
    print("=" * 60)

    # 1. Feature contract
    print("\n[1] Feature contract...")
    contract_path = ARTIFACTS_DIR / "feature_contract.json"
    contract_exists = contract_path.exists()
    results.append(check("Contract exists", contract_exists))

    feature_columns = []
    if contract_exists:
        feature_columns = load_feature_contract(contract_path)
        results.append(check("Contract count", len(feature_columns) > 0,
                             f"{len(feature_columns)} фичей"))
        results.append(check("No duplicates in contract",
                             len(feature_columns) == len(set(feature_columns))))
        forbidden = {"source_dataset", "source_class_folder", "source_relative_file",
                     "source_file_name", "target_binary", "raw_stage2_label"}
        overlap = set(feature_columns) & forbidden
        results.append(check("No forbidden columns in contract", len(overlap) == 0,
                             str(overlap) if overlap else ""))

    # 2. Preprocessed data
    print("\n[2] Preprocessed data...")
    for split in ("train", "val", "test"):
        x_path = PREPROCESSED_DIR / f"{split}_X.parquet"
        y_path = PREPROCESSED_DIR / f"{split}_y.parquet"
        results.append(check(f"{split}_X exists", x_path.exists()))
        results.append(check(f"{split}_y exists", y_path.exists()))

    # 3. Features in preprocessed
    print("\n[3] Features presence...")
    for split in ("train", "val", "test"):
        x_path = PREPROCESSED_DIR / f"{split}_X.parquet"
        if x_path.exists() and feature_columns:
            df = pd.read_parquet(x_path, columns=feature_columns[:1])  # быстрая проверка
            X_cols = set(pd.read_parquet(x_path, columns=None).columns)
            missing = [c for c in feature_columns if c not in X_cols]
            results.append(check(f"All features in {split}_X",
                                 len(missing) == 0, f"missing: {missing[:5]}" if missing else ""))

    # 4. No file overlap between splits
    print("\n[4] File overlap check...")
    split_files = {}
    for split in ("train", "val", "test"):
        y_path = SPLITS_DIR / f"{split}.parquet"
        if y_path.exists():
            df = pd.read_parquet(y_path, columns=["source_relative_file"])
            split_files[split] = set(df["source_relative_file"].unique())

    if len(split_files) == 3:
        tv_overlap = split_files["train"] & split_files["val"]
        tt_overlap = split_files["train"] & split_files["test"]
        vt_overlap = split_files["val"] & split_files["test"]
        results.append(check("No train-val file overlap", len(tv_overlap) == 0,
                             str(tv_overlap) if tv_overlap else ""))
        results.append(check("No train-test file overlap", len(tt_overlap) == 0,
                             str(tt_overlap) if tt_overlap else ""))
        results.append(check("No val-test file overlap", len(vt_overlap) == 0,
                             str(vt_overlap) if vt_overlap else ""))

    # 5. Schema consistency
    print("\n[5] Schema consistency...")
    schemas = {}
    for split in ("train", "val", "test"):
        x_path = PREPROCESSED_DIR / f"{split}_X.parquet"
        if x_path.exists():
            schemas[split] = sorted(pd.read_parquet(x_path, columns=None).columns.tolist())
    if len(schemas) == 3:
        results.append(check("Same schema train=val", schemas["train"] == schemas["val"]))
        results.append(check("Same schema train=test", schemas["train"] == schemas["test"]))

    # 6. No NaN/Inf in preprocessed
    print("\n[6] NaN/Inf check in preprocessed...")
    for split in ("train", "val", "test"):
        x_path = PREPROCESSED_DIR / f"{split}_X.parquet"
        if x_path.exists():
            X = pd.read_parquet(x_path)
            n_nan = int(X.isna().sum().sum())
            n_inf = int(np.isinf(X.select_dtypes(include=["number"])).sum().sum())
            results.append(check(f"No NaN in {split}_X", n_nan == 0, f"NaN count: {n_nan}"))
            results.append(check(f"No Inf in {split}_X", n_inf == 0, f"Inf count: {n_inf}"))

    # 7. Train balance
    print("\n[7] Train balance...")
    train_y_path = PREPROCESSED_DIR / "train_y.parquet"
    if train_y_path.exists():
        y_train = pd.read_parquet(train_y_path)
        n_benign = int((y_train["target_binary"] == 0).sum())
        n_attack = int((y_train["target_binary"] == 1).sum())
        ratio = n_attack / n_benign if n_benign > 0 else 0
        results.append(check("Train balance ~1:1", 0.8 <= ratio <= 1.2,
                             f"benign={n_benign:,}, attack={n_attack:,}, ratio={ratio:.3f}"))

    # 8. Models exist
    print("\n[8] Models...")
    for name, model_dir, model_file in [
        ("CatBoost", CATBOOST_DIR, "model.cbm"),
        ("LightGBM", LIGHTGBM_DIR, "model.lgb"),
    ]:
        model_path = model_dir / model_file
        results.append(check(f"{name} model exists", model_path.exists()))
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            test_f1 = m.get("test", {}).get("f1", 0)
            results.append(check(f"{name} test F1 > 0.5", test_f1 > 0.5,
                                 f"F1={test_f1:.4f}"))

    # 9. Artifacts
    print("\n[9] Preprocessing artifacts...")
    for fname in ["feature_contract.json", "preprocessing_params.json", "scaler.joblib"]:
        results.append(check(f"Artifact: {fname}", (ARTIFACTS_DIR / fname).exists()))

    # 10. Expected class coverage
    print("\n[10] Class coverage (val/test)...")
    for split in ("val", "test"):
        y_path = SPLITS_DIR / f"{split}.parquet"
        if y_path.exists():
            df = pd.read_parquet(y_path, columns=["source_class_folder"])
            classes = set(df["source_class_folder"].unique())
            # BruteForce и Recon должны отсутствовать
            for expected_absent in ["BruteForce", "Recon"]:
                results.append(check(
                    f"{expected_absent} absent from {split} (expected)",
                    expected_absent not in classes,
                    f"classes in {split}: {sorted(classes)}",
                ))

    # Summary
    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    all_passed = failed == 0

    print(f"ИТОГО: {passed}/{total} PASS, {failed} FAIL")
    print(f"СТАТУС: {'ALL PASSED' if all_passed else 'HAS FAILURES'}")
    print("=" * 60)

    # Save reports
    report = {
        "all_passed": all_passed,
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "checks": results,
    }

    json_path = REPORTS_DIR / "08_audit_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Финальный аудит pipeline\n"]
    lines.append(f"- Статус: **{'ALL PASSED' if all_passed else 'HAS FAILURES'}**")
    lines.append(f"- Проверок: **{total}**, пройдено: **{passed}**, провалено: **{failed}**\n")

    lines.append("## Результаты проверок\n")
    lines.append("| # | Проверка | Статус | Детали |")
    lines.append("|---:|---|---|---|")
    for i, r in enumerate(results, 1):
        icon = "PASS" if r["status"] == "PASS" else "FAIL"
        lines.append(f"| {i} | {r['name']} | {icon} | {r['detail']} |")

    md_path = REPORTS_DIR / "08_audit_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nОтчёт: {json_path}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_audit()
