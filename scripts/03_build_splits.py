"""
Шаг 3: File-aware split на train/val/test с балансировкой атак.

Выходы:
  data/splits/train.parquet
  data/splits/val.parquet
  data/splits/test.parquet
  reports/03_split_report.json
  reports/03_split_report.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    COMPACT_DIR, SPLITS_DIR, REPORTS_DIR, FEATURE_CONTRACT_PATH,
    ATTACK_SPLIT_TARGETS, TRAIN_ATTACK_TO_BENIGN_RATIO,
    TRAIN_ATTACK_POLICY, BATCH_SIZE, PARQUET_COMPRESSION, SEED,
    META_COLUMNS, MAX_RAM_GB,
)
from src.io_utils import load_feature_contract
from src.split_logic import (
    build_file_stats, build_split_manifest,
    build_train_attack_quotas, ExactPerFileSampler,
)


def write_split_parquets(
    input_dir: Path,
    output_dir: Path,
    feature_columns: list,
    manifest: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> dict:
    """Streaming write: читаем compact, распределяем по splits, train attack — exact sampling."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(str(input_dir), format="parquet")
    keep_columns = feature_columns + META_COLUMNS
    file_to_split = manifest.set_index("source_relative_file")["split"].to_dict()
    sampler = ExactPerFileSampler(manifest, seed=seed)

    writers = {"train": None, "val": None, "test": None}
    output_paths = {s: output_dir / f"{s}.parquet" for s in writers}
    stats = {s: {"rows": 0, "benign": 0, "attack": 0, "class_rows": defaultdict(int)} for s in writers}

    for batch in dataset.to_batches(columns=keep_columns, batch_size=batch_size, use_threads=False):
        df = batch.to_pandas()
        df["__split"] = df["source_relative_file"].map(file_to_split)
        df = df[df["__split"].notna()].copy()
        if df.empty:
            continue

        for split_name in ("train", "val", "test"):
            part = df[df["__split"] == split_name].copy()
            if part.empty:
                continue

            # Train attack: exact sampling
            if split_name == "train":
                benign = part[part["target_binary"] == 0]
                attack = part[part["target_binary"] == 1].reset_index(drop=True)

                if not attack.empty:
                    keep_mask = np.zeros(len(attack), dtype=bool)
                    for fk, idx in attack.groupby("source_relative_file", sort=False).indices.items():
                        idx = np.asarray(idx, dtype=np.int64)
                        local_mask = sampler.sample_mask(fk, len(idx))
                        keep_mask[idx] = local_mask
                    attack = attack.loc[keep_mask]

                part = pd.concat([benign, attack], ignore_index=True)

            part = part[keep_columns].copy()
            if part.empty:
                continue

            # Статистика
            stats[split_name]["rows"] += len(part)
            stats[split_name]["benign"] += int((part["target_binary"] == 0).sum())
            stats[split_name]["attack"] += int((part["target_binary"] == 1).sum())
            for cls, cnt in part["source_class_folder"].value_counts().items():
                stats[split_name]["class_rows"][str(cls)] += int(cnt)

            # Запись
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writers[split_name] is None:
                writers[split_name] = pq.ParquetWriter(
                    str(output_paths[split_name]), table.schema,
                    compression=PARQUET_COMPRESSION,
                )
            writers[split_name].write_table(table)

    for w in writers.values():
        if w is not None:
            w.close()

    sampler.validate()

    # Конвертируем defaultdict
    for s in stats:
        stats[s]["class_rows"] = dict(sorted(stats[s]["class_rows"].items()))

    return {"output_paths": {k: str(v) for k, v in output_paths.items()}, "stats": stats}


def run_build_splits():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_contract(FEATURE_CONTRACT_PATH)
    print(f"Feature contract: {len(feature_columns)} признаков")

    print("[1/4] Сбор file-level статистики...")
    file_stats = build_file_stats(COMPACT_DIR, BATCH_SIZE)
    print(f"  Файлов: {len(file_stats)}, строк: {file_stats['rows'].sum():,}")

    print("[2/4] File-aware split...")
    manifest = build_split_manifest(file_stats, ATTACK_SPLIT_TARGETS)
    for s in ("train", "val", "test"):
        n_files = (manifest["split"] == s).sum()
        n_rows = manifest.loc[manifest["split"] == s, "rows"].sum()
        print(f"  {s}: {n_files} файлов, {n_rows:,} строк (до sampling)")

    print("[3/4] Расчёт train attack quotas...")
    manifest = build_train_attack_quotas(manifest, TRAIN_ATTACK_TO_BENIGN_RATIO, TRAIN_ATTACK_POLICY)
    total_quota = manifest["train_attack_quota"].sum()
    print(f"  Train attack quota: {total_quota:,}")

    print("[4/4] Запись split parquets...")
    result = write_split_parquets(
        COMPACT_DIR, SPLITS_DIR, feature_columns, manifest, BATCH_SIZE, SEED,
    )

    # Отчёты
    report = {
        "config": {
            "feature_count": len(feature_columns),
            "attack_split_targets": ATTACK_SPLIT_TARGETS,
            "train_ratio": TRAIN_ATTACK_TO_BENIGN_RATIO,
            "policy": TRAIN_ATTACK_POLICY,
            "seed": SEED,
        },
        "file_stats_total": {
            "files": len(file_stats),
            "rows": int(file_stats["rows"].sum()),
        },
        "written": result["stats"],
        "manifest": manifest.to_dict(orient="records"),
    }

    json_path = REPORTS_DIR / "03_split_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Markdown
    ws = result["stats"]
    lines = ["# Split report\n"]
    lines.append(f"- Файлов в compact: **{len(file_stats)}**")
    lines.append(f"- Строк в compact: **{file_stats['rows'].sum():,}**")
    lines.append(f"- Policy: **{TRAIN_ATTACK_POLICY}**, ratio: **{TRAIN_ATTACK_TO_BENIGN_RATIO}**\n")

    lines.append("## Written splits\n")
    lines.append("| Split | Строк | Benign | Attack |")
    lines.append("|---|---:|---:|---:|")
    for s in ("train", "val", "test"):
        lines.append(f"| {s} | {ws[s]['rows']:,} | {ws[s]['benign']:,} | {ws[s]['attack']:,} |")

    for s in ("train", "val", "test"):
        lines.append(f"\n### {s} — по классам\n")
        lines.append("| Класс | Строк |")
        lines.append("|---|---:|")
        for cls, cnt in sorted(ws[s]["class_rows"].items()):
            lines.append(f"| {cls} | {cnt:,} |")

    # Классы только в train
    train_classes = set(ws["train"]["class_rows"].keys())
    val_classes = set(ws["val"]["class_rows"].keys())
    test_classes = set(ws["test"]["class_rows"].keys())
    train_only = train_classes - val_classes - test_classes
    if train_only:
        lines.append("\n## Классы только в train (нет в val/test)\n")
        for c in sorted(train_only):
            lines.append(f"- {c}")

    md_path = REPORTS_DIR / "03_split_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nГотово:")
    for s in ("train", "val", "test"):
        print(f"  {s}: {ws[s]['rows']:,} строк")
    print(f"  Report: {json_path}")


if __name__ == "__main__":
    run_build_splits()
