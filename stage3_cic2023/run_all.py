"""
stage3_cic2023 — запуск пайплайна.

    python run_all.py            # все шаги 1-5
    python run_all.py 2 3 4 5    # начиная с шага 2
    python run_all.py 4 5        # только обучение + оценка
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

SCRIPTS = {
    1: "scripts/01_explore.py",
    2: "scripts/02_build_splits.py",
    3: "scripts/03_preprocess.py",
    4: "scripts/04_train_catboost.py",
    5: "scripts/05_evaluate.py",
}
ROOT  = Path(__file__).parent
steps = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else list(SCRIPTS)
print("=" * 50)
print(f"  stage3_cic2023  |  шаги: {steps}")
print("=" * 50)
for step in steps:
    script = ROOT / SCRIPTS[step]
    print(f"\n{'='*50}\n  ШАГ {step}: {script.name}\n{'='*50}")
    r = subprocess.run([sys.executable, str(script)], check=False)
    if r.returncode != 0:
        print(f"\n[ERROR] Шаг {step} завершился с кодом {r.returncode}")
        sys.exit(r.returncode)
print("\n\n  Все шаги завершены [OK]")
