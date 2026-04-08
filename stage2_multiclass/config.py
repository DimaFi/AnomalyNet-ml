"""
stage2_multiclass — конфигурация многоклассовой модели IDS.
Переиспользует артефакты и сплиты из stage1_v2_cl,
не трогает stage1 вообще.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Среда ──────────────────────────────────────────────────────
IS_COLAB = "COLAB_RELEASE_TAG" in os.environ

if IS_COLAB:
    PROJECT_ROOT = Path("/content/drive/MyDrive/IoT")
else:
    PROJECT_ROOT = Path(r"G:\Диплом\IoT")

STAGE1_ROOT = PROJECT_ROOT / "stage1_v2_cl"
STAGE_ROOT  = PROJECT_ROOT / "stage2_multiclass"

# ── Входные данные (из stage1 — только читаем) ────────────────
STAGE1_SPLITS_DIR  = STAGE1_ROOT / "data" / "splits"
STAGE1_ARTIFACTS   = STAGE1_ROOT / "artifacts"

FEATURE_CONTRACT_PATH = STAGE1_ARTIFACTS / "feature_contract.json"
PREPROCESSING_PARAMS  = STAGE1_ARTIFACTS / "preprocessing_params.json"

# Внешние датасеты
CIC2018_DIR = PROJECT_ROOT.parent / "CSE-CIC-IDS2018"
CIC2017_DIR = PROJECT_ROOT.parent / "GeneratedLabelledFlows" / "TrafficLabelling"

# ── Выходные пути stage2 ──────────────────────────────────────
DATA_DIR         = STAGE_ROOT / "data"
MODELS_DIR       = STAGE_ROOT / "models" / "catboost"
ARTIFACTS_DIR    = STAGE_ROOT / "artifacts"
REPORTS_DIR      = STAGE_ROOT / "reports"

SEED = 42

# ── Таксономия 8 классов ──────────────────────────────────────
# Каждый класс — список строк, которые встречаются в label-колонке
# исходных датасетов (после lower().strip())

CLASS_NAMES = {
    0: "Benign",
    1: "DoS",
    2: "DDoS",
    3: "Recon",
    4: "BruteForce",
    5: "WebAttack",
    6: "Bot",
    7: "Spoofing",
}
NUM_CLASSES = len(CLASS_NAMES)

# Маппинг: нижний регистр строки метки → class_id
LABEL_TO_CLASS: dict[str, int] = {
    # Benign
    "benign": 0,
    "normal": 0,
    # DoS
    "dos": 1,
    "dos attacks-hulk": 1,
    "dos attacks-goldeneye": 1,
    "dos attacks-slowloris": 1,
    "dos attacks-slowhttptest": 1,
    "dos hulk": 1,
    "dos goldeneye": 1,
    "dos slowloris": 1,
    "dos slowhttptest": 1,
    # DDoS
    "ddos": 2,
    "ddos attacks-loic-http": 2,
    "ddos attack-hoic": 2,
    "ddos attack-loic-udp": 2,
    # Recon
    "recon": 3,
    "portscan": 3,
    "port scan": 3,
    "reconnaissance": 3,
    # BruteForce
    "bruteforce": 4,
    "brute force": 4,
    "ssh-bruteforce": 4,
    "ftp-bruteforce": 4,
    "ftp-patator": 4,
    "ssh-patator": 4,
    # WebAttack
    "web-based": 5,
    "webattack": 5,
    "web attack": 5,
    "brute force -web": 5,
    "brute force -xss": 5,
    "sql injection": 5,
    "xss": 5,
    # Bot / Mirai
    "mirai": 6,
    "bot": 6,
    "botnet": 6,
    # Spoofing (только IoT-2024)
    "spoofing": 7,
}

# ── Лимиты семплирования (на класс, для train augmented) ──────
# Предотвращает утечку памяти при большом DoS/DDoS
CLASS_SAMPLE_CAP = {
    0: 300_000,   # Benign
    1: 300_000,   # DoS
    2: 200_000,   # DDoS
    3: 80_000,    # Recon
    4: 80_000,    # BruteForce
    5: 50_000,    # WebAttack
    6: 80_000,    # Bot
    7: 80_000,    # Spoofing
}

# ── Гиперпараметры CatBoost (multiclass) ─────────────────────
CATBOOST_MC_PARAMS = {
    "iterations":            3000,
    "learning_rate":         0.05,
    "depth":                 8,
    "l2_leaf_reg":           5,
    "loss_function":         "MultiClass",
    "eval_metric":           "TotalF1",
    "classes_count":         NUM_CLASSES,
    "early_stopping_rounds": 200,
    "task_type":             "GPU",
    "gpu_ram_part":          0.7,
    "random_seed":           SEED,
    "verbose":               200,
    # Веса классов: повышаем редкие
    "class_weights": [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 2.0, 2.0],
}
