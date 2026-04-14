"""
stage4_extended — конфигурация.

Stage3 (CIC IoT 2023) + CIC-IDS-2018 augmentation.
Те же 46 признаков, та же таксономия 8 классов.
Цель: улучшить обобщение на стандартные VPS-атаки (domain shift fix).
"""
from __future__ import annotations
import os
from pathlib import Path

IS_COLAB = "COLAB_RELEASE_TAG" in os.environ

if IS_COLAB:
    PROJECT_ROOT = Path("/content/drive/MyDrive/IoT")
else:
    PROJECT_ROOT = Path(r"G:\Диплом\IoT")

STAGE3_ROOT    = PROJECT_ROOT / "stage3_cic2023"
STAGE4_ROOT    = PROJECT_ROOT / "stage4_extended"
CIC2018_DIR    = Path(r"G:\Диплом\CSE-CIC-IDS2018")

DATA_DIR    = STAGE4_ROOT / "data"
MODELS_DIR  = STAGE4_ROOT / "models" / "catboost"
REPORTS_DIR = STAGE4_ROOT / "reports"

SEED = 42

# ── 46 признаков (идентичны Stage3) ───────────────────────────
FEATURES_2023 = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration",
    "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]

# ── Таксономия 8 классов ───────────────────────────────────────
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

# Маппинг меток CIC-IDS-2018 -> class_id
CIC2018_LABEL_MAP: dict[str, int] = {
    # Benign
    "benign":                         0,
    # DoS
    "dos attacks-hulk":               1,
    "dos attacks-goldeneye":          1,
    "dos attacks-slowloris":          1,
    "dos attacks-slowhttptest":       1,
    # DDoS
    "ddos attack-loic-udp":           2,
    "ddos attack-hoic":               2,
    "ddos attacks-loic-http":         2,
    "ddos attack-loic-http":          2,
    # Recon
    "infilteration":                  3,  # reconnaissance
    # BruteForce
    "ssh-bruteforce":                 4,
    "ftp-bruteforce":                 4,
    "brute force -web":               4,
    "brute force -xss":               4,
    # WebAttack
    "sql injection":                  5,
    # Bot
    "bot":                            6,
}

# Per-class cap для augmented train
# Базовый Stage3 уже имеет: Benign=200K, DoS=200K, DDoS=200K, Bot=150K, Recon=50K
# CIC-IDS-2018 добавляет в основном DoS/DDoS/BruteForce
CLASS_SAMPLE_CAP = {
    0: 300_000,   # Benign (Stage3 200K + до 100K из 2018)
    1: 300_000,   # DoS    (Stage3 200K + до 100K из 2018)
    2: 300_000,   # DDoS   (Stage3 200K + до 100K из 2018)
    3: 80_000,    # Recon  (Stage3 50K + до 30K из 2018)
    4: 50_000,    # BruteForce (Stage3 ~5K + до 45K из 2018!)
    5: 20_000,    # WebAttack
    6: 200_000,   # Bot
    7: 50_000,    # Spoofing
}

# ── CatBoost параметры ─────────────────────────────────────────
CATBOOST_PARAMS = {
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
    # Поднимаем вес BruteForce (раньше недопредставлен) и WebAttack
    "class_weights": [1.0, 1.0, 1.0, 2.0, 4.0, 4.0, 1.5, 2.0],
}
