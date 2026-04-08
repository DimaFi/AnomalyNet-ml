"""
stage3_cic2023 — конфигурация.
Модель на CIC IoT Dataset 2023 (47 признаков, 34 класса -> 8 унифицированных).
Отдельный feature space, не совместим с stage1/stage2 напрямую.
"""
from __future__ import annotations
import os
from pathlib import Path

IS_COLAB = "COLAB_RELEASE_TAG" in os.environ

if IS_COLAB:
    PROJECT_ROOT = Path("/content/drive/MyDrive/IoT")
else:
    PROJECT_ROOT = Path(r"G:\Диплом\IoT")

DATASET_DIR = PROJECT_ROOT / "CIC IoT Dataset 2023"
STAGE_ROOT  = PROJECT_ROOT / "stage3_cic2023"

DATA_DIR    = STAGE_ROOT / "data"
MODELS_DIR  = STAGE_ROOT / "models" / "catboost"
REPORTS_DIR = STAGE_ROOT / "reports"

LABEL_COL = "label"
SEED = 42

# ── 47 признаков датасета 2023 ─────────────────────────────────
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

# ── Унифицированная таксономия 8 классов ──────────────────────
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

# Маппинг исходных меток -> class_id
LABEL_TO_CLASS: dict[str, int] = {
    # Benign
    "benigntraffic": 0,
    # DoS
    "dos-udp_flood": 1,
    "dos-tcp_flood": 1,
    "dos-syn_flood": 1,
    "dos-http_flood": 1,
    # DDoS
    "ddos-icmp_flood": 2,
    "ddos-udp_flood": 2,
    "ddos-tcp_flood": 2,
    "ddos-pshack_flood": 2,
    "ddos-syn_flood": 2,
    "ddos-rstfinflood": 2,
    "ddos-synonymousip_flood": 2,
    "ddos-icmp_fragmentation": 2,
    "ddos-udp_fragmentation": 2,
    "ddos-ack_fragmentation": 2,
    "ddos-http_flood": 2,
    "ddos-slowloris": 2,
    # Recon
    "recon-hostdiscovery": 3,
    "recon-osscan": 3,
    "recon-portscan": 3,
    "recon-pingsweep": 3,
    "vulnerabilityscan": 3,
    # BruteForce
    "dictionarybruteforce": 4,
    # WebAttack
    "sqlinjection": 5,
    "xss": 5,
    "commandinjection": 5,
    "browserhijacking": 5,
    "uploading_attack": 5,
    # Bot / Mirai
    "mirai-greeth_flood": 6,
    "mirai-udpplain": 6,
    "mirai-greip_flood": 6,
    "backdoor_malware": 6,
    # Spoofing / MITM
    "mitm-arpspoofing": 7,
    "dns_spoofing": 7,
}

# ── Разбивка train/val/test ────────────────────────────────────
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Лимиты на класс в train (предотвращаем перекос DoS/DDoS)
CLASS_SAMPLE_CAP = {
    0: 200_000,   # Benign
    1: 200_000,   # DoS
    2: 200_000,   # DDoS
    3: 50_000,    # Recon
    4: 5_000,     # BruteForce (всего ~11K, берём 70% в train)
    5: 8_000,     # WebAttack  (всего ~19K)
    6: 150_000,   # Bot/Mirai
    7: 50_000,    # Spoofing
}

# ── CatBoost MultiClass ────────────────────────────────────────
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
    # Поднимаем веса редких классов
    "class_weights": [1.0, 1.0, 1.0, 3.0, 6.0, 6.0, 1.5, 2.0],
}
