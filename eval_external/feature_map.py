"""
Маппинг колонок внешних датасетов → имена признаков модели.

Модель обучена на CIC IoT-DIAD 2024 (71 признак).
Другие датасеты CICFlowMeter содержат те же признаки, но с другими именами.
"""

from __future__ import annotations
from typing import Dict, List
import pandas as pd


# ---------------------------------------------------------------------------
# CIC-IDS 2018 (CSE-CIC-IDS2018)
# 10 Parquet-файлов, 78 колонок, Label в последней колонке
# ---------------------------------------------------------------------------
CIC2018_MAP: Dict[str, str] = {
    "Total Fwd Packets":       "Total Fwd Packet",
    "Total Backward Packets":  "Total Bwd packets",
    "Fwd Packets Length Total":"Total Length of Fwd Packet",
    "Bwd Packets Length Total":"Total Length of Bwd Packet",
    "CWE Flag Count":          "CWR Flag Count",
    "Avg Packet Size":         "Average Packet Size",
    "Avg Fwd Segment Size":    "Fwd Segment Size Avg",
    "Avg Bwd Segment Size":    "Bwd Segment Size Avg",
    "Init Fwd Win Bytes":      "FWD Init Win Bytes",
    "Init Bwd Win Bytes":      "Bwd Init Win Bytes",
    "Fwd Act Data Packets":    "Fwd Act Data Pkts",
}

# Метка в датасете 2018
CIC2018_LABEL_COL = "Label"
CIC2018_BENIGN_VALUE = "Benign"

# ---------------------------------------------------------------------------
# CIC-IDS 2017 (GeneratedLabelledFlows)
# 8 CSV-файлов, 85 колонок, Label в последней колонке
# Примечание: в датасете есть дублирующаяся колонка Fwd Header Length (pos 62)
# ---------------------------------------------------------------------------
CIC2017_MAP: Dict[str, str] = {
    "Total Fwd Packets":       "Total Fwd Packet",
    "Total Backward Packets":  "Total Bwd packets",
    "Total Length of Fwd Packets": "Total Length of Fwd Packet",
    "Total Length of Bwd Packets": "Total Length of Bwd Packet",
    "Min Packet Length":       "Packet Length Min",
    "Max Packet Length":       "Packet Length Max",
    "CWE Flag Count":          "CWR Flag Count",
    "Average Packet Size":     "Average Packet Size",   # уже совпадает
    "Avg Fwd Segment Size":    "Fwd Segment Size Avg",
    "Avg Bwd Segment Size":    "Bwd Segment Size Avg",
    "Init_Win_bytes_forward":  "FWD Init Win Bytes",
    "Init_Win_bytes_backward": "Bwd Init Win Bytes",
    "act_data_pkt_fwd":        "Fwd Act Data Pkts",
    "min_seg_size_forward":    "Fwd Seg Size Min",
}

CIC2017_LABEL_COL = "Label"
CIC2017_BENIGN_VALUE = "BENIGN"


# ---------------------------------------------------------------------------
# Общая функция выравнивания
# ---------------------------------------------------------------------------

def align_columns(
    df: pd.DataFrame,
    rename_map: Dict[str, str],
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Приводит датафрейм внешнего датасета к формату, который ждёт модель.

    1. Переименовывает колонки по rename_map
    2. Выбирает только нужные feature_columns
    3. Приводит типы к float32

    Если каких-то колонок нет — поднимает ValueError с конкретным списком.
    """
    # Убираем дубли колонок (известная проблема CIC-IDS 2017)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    df = df.rename(columns=rename_map)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"После маппинга отсутствуют {len(missing)} колонок: {missing[:10]}"
        )

    result = df[feature_columns].copy()
    result = result.astype("float32", errors="ignore")
    return result
