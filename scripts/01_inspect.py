"""
Шаг 1: Аудит сырых CSV датасета CIC IoT-DIAD 2024.
Только чтение — ничего не меняет.

Выходы:
  reports/01_inspect_report.json
  reports/01_inspect_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAW_DATASET_ROOT, REPORTS_DIR, KNOWN_HEADERLESS_RELATIVE
from src.io_utils import list_csv_files, make_progress


def detect_header_and_columns(path: Path, sample_bytes: int = 65536) -> dict:
    """Читает начало файла, определяет наличие заголовка и кол-во колонок."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
    except Exception as e:
        return {"error": str(e), "has_header": None, "columns": 0}

    if not first_line:
        return {"has_header": None, "columns": 0, "first_line_preview": ""}

    parts = first_line.split(",")
    has_header = first_line.startswith("Flow ID,")

    return {
        "has_header": has_header,
        "columns": len(parts),
        "first_line_preview": first_line[:200],
    }


def run_inspect():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATASET_ROOT.exists():
        print(f"[ОШИБКА] Папка датасета не найдена: {RAW_DATASET_ROOT}")
        sys.exit(1)

    files = list_csv_files(RAW_DATASET_ROOT)
    print(f"Найдено CSV: {len(files)}")

    inventory = []
    class_stats = {}

    for path in make_progress(files, desc="Инспекция CSV"):
        rel = str(path.relative_to(RAW_DATASET_ROOT))
        class_folder = path.relative_to(RAW_DATASET_ROOT).parts[0]
        size_mb = round(path.stat().st_size / (1024 * 1024), 2)

        info = detect_header_and_columns(path)
        is_known_headerless = rel in KNOWN_HEADERLESS_RELATIVE

        entry = {
            "relative_path": rel,
            "class_folder": class_folder,
            "size_mb": size_mb,
            "has_header": info.get("has_header"),
            "columns": info.get("columns", 0),
            "known_headerless": is_known_headerless,
            "suspicious": not info.get("has_header", True) and not is_known_headerless,
        }
        inventory.append(entry)

        if class_folder not in class_stats:
            class_stats[class_folder] = {"files": 0, "total_mb": 0.0}
        class_stats[class_folder]["files"] += 1
        class_stats[class_folder]["total_mb"] = round(
            class_stats[class_folder]["total_mb"] + size_mb, 2
        )

    suspicious = [e for e in inventory if e["suspicious"]]

    report = {
        "dataset_root": str(RAW_DATASET_ROOT),
        "total_files": len(files),
        "total_size_mb": round(sum(e["size_mb"] for e in inventory), 2),
        "class_stats": dict(sorted(class_stats.items())),
        "known_headerless_count": sum(1 for e in inventory if e["known_headerless"]),
        "suspicious_count": len(suspicious),
        "suspicious_files": [e["relative_path"] for e in suspicious],
        "inventory": inventory,
    }

    json_path = REPORTS_DIR / "01_inspect_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown
    lines = ["# Аудит сырого датасета CIC IoT-DIAD 2024\n"]
    lines.append(f"- Корень: `{RAW_DATASET_ROOT}`")
    lines.append(f"- Файлов: **{len(files)}**")
    lines.append(f"- Общий размер: **{report['total_size_mb']:.1f} MB**")
    lines.append(f"- Известные файлы без заголовка: **{report['known_headerless_count']}**")
    lines.append(f"- Подозрительные: **{len(suspicious)}**\n")

    lines.append("## Статистика по классам\n")
    lines.append("| Класс | Файлов | Размер (MB) |")
    lines.append("|---|---:|---:|")
    for cls, st in sorted(class_stats.items()):
        lines.append(f"| {cls} | {st['files']} | {st['total_mb']:.1f} |")

    if suspicious:
        lines.append("\n## Подозрительные файлы\n")
        for e in suspicious:
            lines.append(f"- `{e['relative_path']}`")

    md_path = REPORTS_DIR / "01_inspect_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nГотово:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print(f"  Подозрительных файлов: {len(suspicious)}")


if __name__ == "__main__":
    run_inspect()
