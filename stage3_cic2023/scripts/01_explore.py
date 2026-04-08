"""
Шаг 1: Разведочный анализ CIC IoT Dataset 2023.
Считает распределение классов, статистику признаков,
сохраняет отчёт.

Вывод:
  data/class_counts_raw.json   -- кол-во записей каждого исходного класса
  data/feature_stats.json      -- min/max/mean/std по каждому признаку
  reports/class_distribution.png
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_DIR, DATA_DIR, REPORTS_DIR, LABEL_COL

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(DATASET_DIR.glob("*.csv"))
print(f"Файлов: {len(files)}")
print(f"Директория: {DATASET_DIR}")

label_counts: dict[str, int] = {}
feature_buffers: dict[str, list] = {}
t0 = time.time()

for i, fp in enumerate(files):
    df = pd.read_csv(fp, low_memory=False)
    # Метки
    for lbl, cnt in df[LABEL_COL].value_counts().items():
        label_counts[str(lbl)] = label_counts.get(str(lbl), 0) + int(cnt)
    # Признаки (каждые 5 файлов пишем буфер)
    feat_cols = [c for c in df.columns if c != LABEL_COL]
    for col in feat_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if col not in feature_buffers:
            feature_buffers[col] = []
        # Семплируем максимум 1000 значений на файл для экономии памяти
        if len(vals) > 1000:
            vals = vals[np.random.choice(len(vals), 1000, replace=False)]
        feature_buffers[col].extend(vals.tolist())

    if (i + 1) % 20 == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(files)}] {elapsed:.0f}s  Меток: {len(label_counts)}")

print(f"\nВсего {len(files)} файлов обработано за {time.time()-t0:.0f}s")

# Итог по классам
total = sum(label_counts.values())
print(f"\nИтого записей: {total:,}")
print("\nРаспределение классов (полное):")
for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
    pct = cnt / total * 100
    print(f"  {cnt:>10,}  {pct:5.2f}%  {lbl}")

with open(DATA_DIR / "class_counts_raw.json", "w") as f:
    json.dump(label_counts, f, indent=2, ensure_ascii=False)
print(f"\nСохранено: {DATA_DIR / 'class_counts_raw.json'}")

# Статистика признаков
feat_stats = {}
for col, vals in feature_buffers.items():
    arr = np.array(vals, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        continue
    feat_stats[col] = {
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "n":    len(arr),
    }

with open(DATA_DIR / "feature_stats.json", "w") as f:
    json.dump(feat_stats, f, indent=2, ensure_ascii=False)
print(f"Сохранено: {DATA_DIR / 'feature_stats.json'}")

# График распределения классов
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 7))
    lbls = [k for k, v in sorted(label_counts.items(), key=lambda x: -x[1])]
    vals = [label_counts[k] for k in lbls]
    colors = ["#5cc97a" if "Benign" in k else
              "#e86070" if "DDoS" in k else
              "#f0bc5e" if "DoS" in k else
              "#7db8f7" for k in lbls]
    bars = ax.barh(lbls[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("Количество записей")
    ax.set_title("CIC IoT Dataset 2023 — распределение классов (полный датасет)", fontsize=12)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(val + total*0.003, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=8)
    ax.set_xscale("log")
    plt.tight_layout()
    out = REPORTS_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График >> {out}")
except ImportError:
    pass

print("\n[OK] Шаг 1 завершён")
