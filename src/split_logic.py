"""
File-aware split: раскладка файлов по train/val/test,
балансировка атак, exact one-pass sampling.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .io_utils import stable_int_from_string


# ============================================================
# FILE-LEVEL STATS
# ============================================================

MIN_SCAN_COLUMNS = [
    "source_class_folder",
    "source_relative_file",
    "source_file_name",
    "target_binary",
]


def build_file_stats(input_dir, batch_size: int) -> pd.DataFrame:
    """
    Стриминговый подсчёт строк по файлам из compact parquet.
    Возвращает DataFrame: source_relative_file, source_file_name,
    source_class_folder, target_binary, rows.
    """
    dataset = ds.dataset(str(input_dir), format="parquet")
    for col in MIN_SCAN_COLUMNS:
        if col not in dataset.schema.names:
            raise ValueError(f"Отсутствует колонка: {col}")

    acc = defaultdict(lambda: {
        "source_relative_file": None,
        "source_file_name": None,
        "source_class_folder": None,
        "target_binary": None,
        "rows": 0,
    })

    for batch in dataset.to_batches(
        columns=MIN_SCAN_COLUMNS, batch_size=batch_size, use_threads=False,
    ):
        df = batch.to_pandas()
        grp = (
            df.groupby(MIN_SCAN_COLUMNS, dropna=False)
            .size()
            .reset_index(name="rows")
        )
        for row in grp.itertuples(index=False):
            key = str(row.source_relative_file)
            acc[key]["source_relative_file"] = key
            acc[key]["source_file_name"] = str(row.source_file_name)
            acc[key]["source_class_folder"] = str(row.source_class_folder)
            acc[key]["target_binary"] = int(row.target_binary)
            acc[key]["rows"] += int(row.rows)

    stats = pd.DataFrame(acc.values())
    if stats.empty:
        raise ValueError(f"Пустой dataset: {input_dir}")

    return stats.sort_values(
        by=["target_binary", "source_class_folder", "rows", "source_relative_file"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


# ============================================================
# SPLIT ASSIGNMENT
# ============================================================

def _choose_split_lowest_fill(current_rows: dict, target_rows: dict) -> str:
    """Greedy: выбираем split с наименьшим заполнением."""
    priority = {"train": 0, "val": 1, "test": 2}
    best, best_score = None, None
    for split in ("train", "val", "test"):
        target = target_rows.get(split, 0)
        score = current_rows.get(split, 0) / target if target > 0 else float("inf")
        if best_score is None or score < best_score or (score == best_score and priority[split] < priority.get(best, 3)):
            best_score = score
            best = split
    return best


def assign_benign_files(benign_df: pd.DataFrame) -> Dict[str, str]:
    """
    Benign файлы: минимум 2 в train, 1 в val, 1 в test.
    При >4 файлах — greedy по заполненности.
    """
    benign_df = benign_df.sort_values(by="rows", ascending=False).reset_index(drop=True)
    n = len(benign_df)

    if n < 3:
        raise ValueError(f"Слишком мало benign-файлов ({n}), нужно минимум 3.")

    # Гарантируем минимум: 1 в val, 1 в test
    # Первый файл (самый большой) → train, второй → train, третий → val, четвёртый → test
    assignments = {}
    current_rows = {"train": 0, "val": 0, "test": 0}

    # Seed: по одному в каждый split, начиная с train
    if n >= 4:
        seed_order = ["train", "train", "val", "test"]
    else:
        seed_order = ["train", "val", "test"]

    for i in range(min(len(seed_order), n)):
        key = benign_df.loc[i, "source_relative_file"]
        split = seed_order[i]
        assignments[key] = split
        current_rows[split] += int(benign_df.loc[i, "rows"])

    # Остальные — greedy
    total_rows = int(benign_df["rows"].sum())
    target_rows = {"train": total_rows * 0.5, "val": total_rows * 0.25, "test": total_rows * 0.25}

    for i in range(len(seed_order), n):
        key = benign_df.loc[i, "source_relative_file"]
        split = _choose_split_lowest_fill(current_rows, target_rows)
        assignments[key] = split
        current_rows[split] += int(benign_df.loc[i, "rows"])

    return assignments


def assign_attack_files(attack_df: pd.DataFrame, split_targets: Dict[str, float]) -> Dict[str, str]:
    """
    Раскладка attack-файлов по семействам:
    - 1 файл → train only
    - 2 файла → train + test
    - ≥3 → гарантируем train/val/test, остальные greedy
    """
    assignments = {}

    for family, fam_df in attack_df.groupby("source_class_folder", sort=True):
        fam_df = fam_df.sort_values(by="rows", ascending=False).reset_index(drop=True)
        n = len(fam_df)

        if n == 1:
            assignments[fam_df.loc[0, "source_relative_file"]] = "train"
            continue

        if n == 2:
            assignments[fam_df.loc[0, "source_relative_file"]] = "train"
            assignments[fam_df.loc[1, "source_relative_file"]] = "test"
            continue

        total_rows = int(fam_df["rows"].sum())
        target_rows = {s: total_rows * split_targets[s] for s in ("train", "val", "test")}
        current_rows = {"train": 0, "val": 0, "test": 0}

        for i, split in enumerate(["train", "val", "test"]):
            key = fam_df.loc[i, "source_relative_file"]
            assignments[key] = split
            current_rows[split] += int(fam_df.loc[i, "rows"])

        for i in range(3, n):
            key = fam_df.loc[i, "source_relative_file"]
            split = _choose_split_lowest_fill(current_rows, target_rows)
            assignments[key] = split
            current_rows[split] += int(fam_df.loc[i, "rows"])

    return assignments


def build_split_manifest(
    file_stats: pd.DataFrame,
    attack_split_targets: Dict[str, float],
) -> pd.DataFrame:
    """Объединяет benign + attack assignments в единый manifest."""
    benign_df = file_stats[file_stats["target_binary"] == 0].copy()
    attack_df = file_stats[file_stats["target_binary"] == 1].copy()

    benign_assign = assign_benign_files(benign_df)
    attack_assign = assign_attack_files(attack_df, attack_split_targets)

    all_assign = {**benign_assign, **attack_assign}
    manifest = file_stats.copy()
    manifest["split"] = manifest["source_relative_file"].map(all_assign)

    if manifest["split"].isna().any():
        missing = manifest.loc[manifest["split"].isna(), "source_relative_file"].tolist()
        raise ValueError(f"Файлы без split: {missing[:10]}")

    return manifest


# ============================================================
# TRAIN ATTACK QUOTAS
# ============================================================

def _allocate_integer_quotas(capacities: pd.Series, total_budget: int) -> pd.Series:
    """Целочисленное пропорциональное распределение бюджета."""
    capacities = capacities.astype("int64")
    total_budget = int(min(total_budget, capacities.sum()))
    if total_budget <= 0:
        return pd.Series(0, index=capacities.index, dtype="int64")
    if total_budget >= capacities.sum():
        return capacities.copy()

    raw = total_budget * capacities / capacities.sum()
    floor = np.floor(raw).astype(np.int64)
    floor = np.minimum(floor, capacities.values)
    quotas = pd.Series(floor, index=capacities.index, dtype="int64")
    leftover = int(total_budget - quotas.sum())

    if leftover > 0:
        remainders = (raw - quotas).sort_values(ascending=False)
        for idx in remainders.index:
            if leftover <= 0:
                break
            if quotas.loc[idx] < capacities.loc[idx]:
                quotas.loc[idx] += 1
                leftover -= 1
    return quotas


def allocate_family_balanced_quotas(
    train_attack_df: pd.DataFrame,
    target_attack_rows: int,
) -> pd.DataFrame:
    """
    Family-balanced: равная базовая квота по семействам,
    маленькие семейства берут всё, остаток — пропорционально.
    """
    family_rows = (
        train_attack_df.groupby("source_class_folder")["rows"]
        .sum().astype("int64").sort_index()
    )
    target = int(min(target_attack_rows, family_rows.sum()))
    if target <= 0:
        out = train_attack_df.copy()
        out["train_attack_quota"] = 0
        return out

    n_families = len(family_rows)
    base = target // n_families
    family_quota = pd.Series(
        np.minimum(base, family_rows.values),
        index=family_rows.index, dtype="int64",
    )

    leftover = int(target - family_quota.sum())
    if leftover > 0:
        residual = family_rows - family_quota
        extra = _allocate_integer_quotas(residual, leftover)
        family_quota += extra

    df = train_attack_df.copy()
    df["train_attack_quota"] = 0
    for family, fam_quota in family_quota.items():
        mask = df["source_class_folder"] == family
        caps = df.loc[mask].set_index("source_relative_file")["rows"]
        file_q = _allocate_integer_quotas(caps, int(fam_quota))
        for fk, q in file_q.items():
            df.loc[df["source_relative_file"] == fk, "train_attack_quota"] = int(q)
    return df


def build_train_attack_quotas(
    manifest: pd.DataFrame,
    ratio: float,
    policy: str,
) -> pd.DataFrame:
    """Рассчитывает квоты attack-строк для train."""
    train_benign_rows = int(
        manifest.loc[
            (manifest["split"] == "train") & (manifest["target_binary"] == 0), "rows"
        ].sum()
    )
    train_attack_df = manifest.loc[
        (manifest["split"] == "train") & (manifest["target_binary"] == 1)
    ].copy()

    target = int(min(round(train_benign_rows * ratio), train_attack_df["rows"].sum()))

    if policy == "family_balanced":
        quotas_df = allocate_family_balanced_quotas(train_attack_df, target)
    else:
        raise ValueError(f"Неизвестная policy: {policy}")

    result = manifest.copy()
    result["train_attack_quota"] = 0
    qmap = quotas_df.set_index("source_relative_file")["train_attack_quota"].to_dict()
    result["train_attack_quota"] = (
        result["source_relative_file"].map(qmap).fillna(0).astype("int64")
    )
    return result


# ============================================================
# EXACT ONE-PASS SAMPLER
# ============================================================

class ExactPerFileSampler:
    """One-pass exact sampling без replacement для train attack файлов."""

    def __init__(self, manifest: pd.DataFrame, seed: int):
        self.state = {}
        attack_train = manifest[
            (manifest["split"] == "train") & (manifest["target_binary"] == 1)
        ]
        for row in attack_train.itertuples(index=False):
            fk = row.source_relative_file
            self.state[fk] = {
                "total_rows": int(row.rows),
                "quota": int(row.train_attack_quota),
                "seen": 0,
                "taken": 0,
                "rng": np.random.default_rng(seed + stable_int_from_string(fk)),
            }

    def sample_mask(self, file_key: str, n_rows: int) -> np.ndarray:
        st = self.state[file_key]
        mask = np.zeros(n_rows, dtype=bool)
        for i in range(n_rows):
            remaining = st["total_rows"] - st["seen"]
            to_take = st["quota"] - st["taken"]
            if to_take <= 0:
                st["seen"] += 1
                continue
            take = to_take >= remaining or st["rng"].random() < (to_take / remaining)
            if take:
                mask[i] = True
                st["taken"] += 1
            st["seen"] += 1
        return mask

    def validate(self):
        for fk, st in self.state.items():
            if st["seen"] != st["total_rows"]:
                raise ValueError(f"Sampler: seen mismatch для {fk}")
            if st["taken"] != st["quota"]:
                raise ValueError(f"Sampler: taken mismatch для {fk}")
