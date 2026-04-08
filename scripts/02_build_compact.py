"""
Шаг 2: Конвертация сырых CSV → compact Parquet.

Отличия от v1:
  - Нет cap на DoS — все строки сохраняются
  - Inf/NaN НЕ заменяются (диагностика на шаге 04)
  - Улучшенная отчётность

Выходы:
  data/compact/*.parquet
  reports/02_compact_manifest.csv
  reports/02_compact_report.json
  reports/02_compact_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    RAW_DATASET_ROOT, COMPACT_DIR, REPORTS_DIR,
    KNOWN_HEADERLESS_RELATIVE, DROP_COLUMNS, META_COLUMNS,
    CHUNKSIZE, MAX_RAM_GB,
)
from src.io_utils import (
    list_csv_files, relative_str, top_class_folder,
    is_headerless_file, discover_canonical_columns,
    normalize_column_names, safe_numeric_convert, sanitize_name,
    make_progress,
)


def process_file(
    csv_path: Path,
    out_dir: Path,
    canonical_columns: List[str],
    chunksize: int,
    counters: Dict[str, int],
    manifest_rows: List[Dict],
) -> int:
    """Обрабатывает один CSV → parquet-части. Возвращает кол-во записанных строк."""
    rel = relative_str(csv_path, RAW_DATASET_ROOT)
    class_folder = top_class_folder(csv_path, RAW_DATASET_ROOT)
    target_binary = 0 if class_folder == "Benign" else 1
    headerless = is_headerless_file(csv_path, RAW_DATASET_ROOT, KNOWN_HEADERLESS_RELATIVE)

    read_kwargs = {"chunksize": chunksize, "dtype": str, "low_memory": True}
    if headerless:
        read_kwargs["header"] = None
        read_kwargs["names"] = canonical_columns
    else:
        read_kwargs["header"] = 0

    written = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, **read_kwargs)):
        chunk.columns = normalize_column_names(chunk.columns.tolist())

        # Выравнивание схемы
        for col in canonical_columns:
            if col not in chunk.columns:
                chunk[col] = pd.NA
        chunk = chunk[canonical_columns].copy()

        # Мета-колонки
        chunk["source_dataset"] = "CIC-IoT-DIAD2024"
        chunk["source_class_folder"] = class_folder
        chunk["source_relative_file"] = rel
        chunk["source_file_name"] = csv_path.name
        chunk["target_binary"] = target_binary
        chunk["raw_stage2_label"] = class_folder

        # Удаляем ненужные
        drop = [c for c in DROP_COLUMNS if c in chunk.columns]
        chunk = chunk.drop(columns=drop)

        # Numeric conversion (inf/nan остаются!)
        chunk = safe_numeric_convert(chunk, exclude_cols=META_COLUMNS)

        # Компактные типы мета
        chunk["target_binary"] = chunk["target_binary"].astype("int8")
        for mc in ["source_dataset", "source_class_folder", "source_relative_file",
                    "source_file_name", "raw_stage2_label"]:
            chunk[mc] = chunk[mc].astype("string")

        # Запись parquet
        file_key = sanitize_name(rel)
        counters[file_key] = counters.get(file_key, 0) + 1
        out_file = out_dir / f"{file_key}.part_{counters[file_key]:05d}.parquet"
        chunk.to_parquet(out_file, index=False)

        rows_written = len(chunk)
        written += rows_written

        manifest_rows.append({
            "source_relative_file": rel,
            "source_class_folder": class_folder,
            "headerless_fixed": headerless,
            "chunk_index": chunk_idx,
            "rows_written": rows_written,
            "output_parquet": str(out_file.name),
        })

    return written


def run_build_compact():
    COMPACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATASET_ROOT.exists():
        print(f"[ОШИБКА] Не найден датасет: {RAW_DATASET_ROOT}")
        sys.exit(1)

    files = list_csv_files(RAW_DATASET_ROOT)
    print(f"Найдено CSV: {len(files)}")

    canonical = discover_canonical_columns(files, RAW_DATASET_ROOT, KNOWN_HEADERLESS_RELATIVE)
    print(f"Каноническая схема: {len(canonical)} колонок")

    manifest_rows: List[Dict] = []
    counters: Dict[str, int] = {}
    class_stats: Dict[str, Dict] = {}
    total_written = 0

    for csv_path in make_progress(files, desc="CSV → Parquet"):
        cls = top_class_folder(csv_path, RAW_DATASET_ROOT)
        if cls not in class_stats:
            class_stats[cls] = {"files": 0, "rows": 0}
        class_stats[cls]["files"] += 1

        written = process_file(
            csv_path, COMPACT_DIR, canonical, CHUNKSIZE, counters, manifest_rows,
        )
        class_stats[cls]["rows"] += written
        total_written += written

    # Manifest CSV
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = REPORTS_DIR / "02_compact_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    # JSON report
    report = {
        "dataset_root": str(RAW_DATASET_ROOT),
        "output_dir": str(COMPACT_DIR),
        "chunksize": CHUNKSIZE,
        "files_found": len(files),
        "files_processed": len(files),
        "canonical_columns": len(canonical),
        "rows_written_total": total_written,
        "class_stats": {k: v for k, v in sorted(class_stats.items())},
        "headerless_fixed": sorted(KNOWN_HEADERLESS_RELATIVE),
        "drop_columns": DROP_COLUMNS,
        "notes": [
            "Inf/NaN НЕ заменены — диагностика на шаге 04.",
            "Нет ограничения по количеству строк (нет cap на DoS).",
            "Binary target: Benign=0, остальные=1.",
        ],
    }

    json_path = REPORTS_DIR / "02_compact_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report
    lines = ["# Compact Parquet — отчёт\n"]
    lines.append(f"- Файлов обработано: **{len(files)}**")
    lines.append(f"- Строк записано: **{total_written:,}**")
    lines.append(f"- Каноническая схема: **{len(canonical)}** колонок")
    lines.append(f"- Chunksize: **{CHUNKSIZE:,}**\n")

    lines.append("## Статистика по классам\n")
    lines.append("| Класс | Файлов | Строк |")
    lines.append("|---|---:|---:|")
    for cls, st in sorted(class_stats.items()):
        lines.append(f"| {cls} | {st['files']} | {st['rows']:,} |")

    lines.append("\n## Примечания\n")
    for n in report["notes"]:
        lines.append(f"- {n}")

    md_path = REPORTS_DIR / "02_compact_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nГотово:")
    print(f"  Compact: {COMPACT_DIR}")
    print(f"  Строк:   {total_written:,}")
    print(f"  Manifest: {manifest_csv}")
    print(f"  Report:   {json_path}")


if __name__ == "__main__":
    run_build_compact()
