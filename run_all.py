"""
Запуск полного pipeline одной командой.
Выполняет шаги 01–08 последовательно.
При ошибке на любом шаге — останавливается и сообщает, какой шаг упал.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

# Добавляем корень pipeline в путь
sys.path.insert(0, str(Path(__file__).resolve().parent))

STEPS = [
    ("01_inspect",       "scripts.01_inspect",       "run_inspect"),
    ("02_build_compact", "scripts.02_build_compact", "run_build_compact"),
    ("03_build_splits",  "scripts.03_build_splits",  "run_build_splits"),
    ("04_preprocess",    "scripts.04_preprocess",    "run_preprocess"),
    ("05_train_catboost","scripts.05_train_catboost","run_train_catboost"),
    ("06_train_lightgbm","scripts.06_train_lightgbm","run_train_lightgbm"),
    ("07_evaluate",      "scripts.07_evaluate",      "run_evaluate"),
    ("08_audit",         "scripts.08_audit",         "run_audit"),
]


def main():
    print("=" * 60)
    print("STAGE 1 v2 — FULL PIPELINE")
    print("=" * 60)

    total_start = time.time()

    for step_name, module_path, func_name in STEPS:
        print(f"\n{'=' * 60}")
        print(f">>> {step_name}")
        print(f"{'=' * 60}\n")

        step_start = time.time()
        try:
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            func()
        except SystemExit as e:
            if e.code and e.code != 0:
                print(f"\n[ОШИБКА] Шаг {step_name} завершился с кодом {e.code}")
                sys.exit(e.code)
        except Exception as e:
            print(f"\n[ОШИБКА] Шаг {step_name} упал: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        elapsed = time.time() - step_start
        print(f"\n[OK] {step_name} — {elapsed:.1f} сек")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE ЗАВЕРШЁН УСПЕШНО")
    print(f"Общее время: {total_elapsed:.1f} сек ({total_elapsed / 60:.1f} мин)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
