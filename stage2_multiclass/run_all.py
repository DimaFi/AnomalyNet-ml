"""
stage2_multiclass — запуск всего пайплайна.

Использование:
    python run_all.py            # все шаги
    python run_all.py 1 2 3      # только указанные шаги

Шаги:
    1 — build_labels    (читает stage1 splits, добавляет target_mc)
    2 — add_external    (CIC-IDS2018 + CIC-IDS2017 → train_augmented)
    3 — train           (обучение CatBoost MultiClass)
    4 — evaluate        (метрики + confusion matrix)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = {
    1: "scripts/01_build_labels.py",
    2: "scripts/02_add_external.py",
    3: "scripts/03_train_catboost_mc.py",
    4: "scripts/04_evaluate_mc.py",
}

ROOT = Path(__file__).parent
steps = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else list(SCRIPTS)

print("=" * 55)
print("  stage2_multiclass pipeline")
print(f"  Запускаем шаги: {steps}")
print("=" * 55)

for step in steps:
    script = ROOT / SCRIPTS[step]
    print(f"\n{'=' * 55}")
    print(f"  ШАГ {step}: {script.name}")
    print("=" * 55)
    result = subprocess.run([sys.executable, str(script)], check=False)
    if result.returncode != 0:
        print(f"\n✗ Шаг {step} завершился с ошибкой (код {result.returncode})")
        sys.exit(result.returncode)

print("\n" + "=" * 55)
print("  Все шаги завершены успешно ✓")
print("=" * 55)
