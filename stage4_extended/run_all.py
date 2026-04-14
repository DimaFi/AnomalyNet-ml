"""
Stage4 Extended — запуск всего пайплайна.

Использование:
  python run_all.py          # все шаги
  python run_all.py --from 2 # с шага 2
  python run_all.py --only 3 # только шаг 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    (1, "scripts/01_prepare_cic2018.py",  "CIC-IDS-2018 → CIC2023 features"),
    (2, "scripts/02_merge_augment.py",    "Merge Stage3 + CIC-IDS-2018"),
    (3, "scripts/03_train_catboost.py",   "Обучение CatBoost (GPU)"),
    (4, "scripts/04_evaluate.py",         "Оценка Stage3 vs Stage4"),
]

HERE = Path(__file__).parent

def run_step(num: int, script: str, desc: str):
    path = HERE / script
    print(f"\n{'='*60}")
    print(f"ШАГ {num}: {desc}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, str(path)], cwd=str(HERE))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ОШИБКА] Шаг {num} завершился с кодом {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] Шаг {num} завершён за {elapsed/60:.1f} мин")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_step", type=int, default=1)
    parser.add_argument("--only", dest="only_step", type=int, default=None)
    args = parser.parse_args()

    steps = SCRIPTS
    if args.only_step:
        steps = [(n, s, d) for n, s, d in SCRIPTS if n == args.only_step]
    else:
        steps = [(n, s, d) for n, s, d in SCRIPTS if n >= args.from_step]

    total_start = time.time()
    for num, script, desc in steps:
        run_step(num, script, desc)

    print(f"\n{'='*60}")
    print(f"Всё завершено за {(time.time()-total_start)/60:.1f} мин")
    print(f"Модель: G:/Диплом/IoT/stage4_extended/models/catboost/model_mc.cbm")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
