"""
Центральная конфигурация pipeline stage1_v2_cl.
Все пути, гиперпараметры, настройки — в одном месте.
Переключение local / Colab через переменную IS_COLAB.
"""

from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# ОПРЕДЕЛЕНИЕ СРЕДЫ
# ============================================================

IS_COLAB = "COLAB_RELEASE_TAG" in os.environ

if IS_COLAB:
    PROJECT_ROOT = Path("/content/drive/MyDrive/IoT")
else:
    PROJECT_ROOT = Path(r"G:\Диплом\IoT")

RAW_DATASET_ROOT = PROJECT_ROOT / "CIC IoT-DIAD 2024 Dataset" / "CIC-IoT-DIAD2024"

STAGE_ROOT = PROJECT_ROOT / "stage1_v2_cl"

# ============================================================
# ПУТИ ВНУТРИ PIPELINE
# ============================================================

# Данные
DATA_DIR = STAGE_ROOT / "data"
COMPACT_DIR = DATA_DIR / "compact"
SPLITS_DIR = DATA_DIR / "splits"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

# Артефакты для inference
ARTIFACTS_DIR = STAGE_ROOT / "artifacts"

# Модели
MODELS_DIR = STAGE_ROOT / "models"
CATBOOST_DIR = MODELS_DIR / "catboost"
LIGHTGBM_DIR = MODELS_DIR / "lightgbm"

# Отчёты
REPORTS_DIR = STAGE_ROOT / "reports"

# Контракт признаков (исходный)
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "contracts" / "production_safe_features_no_ports.json"

# ============================================================
# COMPACT BUILD (02_build_compact)
# ============================================================

# ============================================================
# ОГРАНИЧЕНИЯ РЕСУРСОВ
# ============================================================

MAX_RAM_GB = 24                       # верхняя граница RAM для pipeline
MAX_WORKERS = 4 if not IS_COLAB else 2  # потоки для параллельных операций

CHUNKSIZE = 100_000 if not IS_COLAB else 50_000

# 5 известных CSV без заголовка в датасете 2024
KNOWN_HEADERLESS_RELATIVE = {
    r"DoS\DoS-TCP_Flood\DoS-TCP_Flood.pcap_Flow.csv",
    r"DoS\DoS-TCP_Flood\DoS-TCP_Flood1.pcap_Flow.csv",
    r"DoS\DoS-TCP_Flood\DoS-TCP_Flood10.pcap_Flow.csv",
    r"DoS\DoS-TCP_Flood\DoS-TCP_Flood2.pcap_Flow.csv",
    r"DoS\DoS-TCP_Flood\DoS-TCP_Flood3.pcap_Flow.csv",
}

# Колонки, которые удаляем при конвертации (не нужны для обучения)
DROP_COLUMNS = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]

# Мета-колонки, добавляемые при compact build
META_COLUMNS = [
    "source_dataset",
    "source_class_folder",
    "source_relative_file",
    "source_file_name",
    "target_binary",
    "raw_stage2_label",
]

# ============================================================
# FILE-AWARE SPLIT (03_build_splits)
# ============================================================

ATTACK_SPLIT_TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}
TRAIN_ATTACK_TO_BENIGN_RATIO = 1.0
TRAIN_ATTACK_POLICY = "family_balanced"
SEED = 42

BATCH_SIZE = 200_000
PARQUET_COMPRESSION = "zstd"

# ============================================================
# PREPROCESSING (04_preprocess)
# ============================================================

INF_REPLACEMENT = "nan"       # inf -> nan, потом fill
NAN_STRATEGY = "median"       # "median" | "zero" | "mean"

# ============================================================
# ГИПЕРПАРАМЕТРЫ МОДЕЛЕЙ
# ============================================================

CATBOOST_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "early_stopping_rounds": 200,
    "task_type": "GPU",
    "random_seed": SEED,
    "verbose": 200,
    # Ограничение GPU-памяти: не занимаем всю видеокарту
    "gpu_ram_part": 0.7,
}

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": 8,
    "min_child_samples": 50,
    "n_estimators": 3000,
    "device": "gpu" if not IS_COLAB else "cpu",
    "random_state": SEED,
    "verbose": -1,
    # LightGBM GPU: ограничиваем использование
    "gpu_use_dp": False,
}

LIGHTGBM_FIT_PARAMS = {
    "callbacks": None,  # заполняется в скрипте обучения
}

# ============================================================
# THRESHOLD SEARCH
# ============================================================

THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
