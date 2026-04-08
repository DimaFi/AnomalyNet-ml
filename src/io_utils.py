"""
Утилиты ввода-вывода: чтение CSV/Parquet, chunked-операции, работа со схемой.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Generator, List, Optional

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def make_progress(iterable, desc: str = "", total: Optional[int] = None):
    """Обёртка для progress bar (tqdm, если установлен)."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def list_csv_files(root: Path) -> List[Path]:
    """Рекурсивно находит все CSV в директории, сортирует."""
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def relative_str(path: Path, root: Path) -> str:
    """Относительный путь как строка."""
    return str(path.relative_to(root))


def top_class_folder(path: Path, root: Path) -> str:
    """Верхняя папка класса (Benign, DDoS, DoS, ...)."""
    return path.relative_to(root).parts[0]


def is_headerless_file(path: Path, root: Path, known_set: set) -> bool:
    """Проверяет, есть ли файл в списке известных без заголовка."""
    return relative_str(path, root) in known_set


def discover_canonical_columns(files: List[Path], root: Path, known_headerless: set) -> List[str]:
    """
    Определяет каноническую схему колонок из первого файла с заголовком.
    Ожидаем 84 колонки.
    """
    for path in files:
        if is_headerless_file(path, root, known_headerless):
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
        if first_line.startswith("Flow ID,"):
            return [c.strip() for c in first_line.split(",")]
    raise RuntimeError("Не найден CSV с нормальным заголовком для канонической схемы.")


def normalize_column_names(cols: List[str]) -> List[str]:
    """Убирает пробелы в названиях колонок."""
    return [str(c).strip() for c in cols]


def safe_numeric_convert(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    """
    Переводит все колонки (кроме exclude) в numeric → float32.
    НЕ заменяет inf/nan — это задача preprocessing.
    """
    work_cols = [c for c in df.columns if c not in exclude_cols]
    for col in work_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_cols = df[work_cols].select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].astype("float32")
    return df


def load_feature_contract(path: Path) -> List[str]:
    """Загружает JSON-контракт признаков."""
    if not path.exists():
        raise FileNotFoundError(f"Feature contract не найден: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"Некорректный JSON feature contract: {path}")
    return data


def sanitize_name(s: str) -> str:
    """Безопасное имя для parquet-файлов."""
    return s.replace("\\", "_").replace("/", "_").replace(" ", "_").replace(":", "_")


def stable_int_from_string(text: str) -> int:
    """Стабильный int-хеш для seed из строки."""
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % (2**32 - 1)
    return value
