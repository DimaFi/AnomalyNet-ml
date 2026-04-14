"""
Шаг 1: Загрузить CIC-IDS-2018, сконвертировать в 46 CIC2023 признаков.

CIC-IDS-2018 — 10 parquet-файлов, 78 CICFlowMeter колонок.
Нет колонок Destination Port / Source Port → протокольные флаги (HTTP/SSH/DNS...)
выставляются в 0 для всех строк. Флаги TCP/UDP/ICMP вычисляются из Protocol.

Выход: stage4_extended/data/cic2018_mapped.parquet
"""
from __future__ import annotations

import sys
import io
import math
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CIC2018_DIR, DATA_DIR, FEATURES_2023, CIC2018_LABEL_MAP, CLASS_NAMES, SEED

print("=" * 60)
print("Stage4 / Шаг 1: CIC-IDS-2018 → CIC2023 features")
print("=" * 60)


def map_label(raw: str) -> int | None:
    return CIC2018_LABEL_MAP.get(str(raw).strip().lower())


def convert_to_cic2023(df: pd.DataFrame) -> pd.DataFrame:
    """
    Конвертирует строки CIC-IDS-2018 в 46 CIC2023 признаков.
    Возвращает датафрейм с колонками FEATURES_2023.
    """
    eps = 1e-9  # защита от деления на ноль

    # -- Базовые значения --
    fwd_pkts   = df["Total Fwd Packets"].clip(lower=0)
    bwd_pkts   = df["Total Backward Packets"].clip(lower=0)
    total_pkts = fwd_pkts + bwd_pkts

    # Flow Duration: в CIC-IDS-2018 в микросекундах -> конвертируем в секунды
    flow_dur_s = (df["Flow Duration"].clip(lower=0) / 1e6).clip(lower=eps)

    fwd_len_total = df["Fwd Packets Length Total"].clip(lower=0)
    bwd_len_total = df["Bwd Packets Length Total"].clip(lower=0)
    tot_sum = fwd_len_total + bwd_len_total

    # Header length среднее на пакет
    header_len = (df["Fwd Header Length"].clip(lower=0) /
                  fwd_pkts.clip(lower=1))

    protocol = df["Protocol"].fillna(0).astype(float)

    # -- Скорости --
    rate   = total_pkts / flow_dur_s
    srate  = fwd_pkts   / flow_dur_s
    drate  = bwd_pkts   / flow_dur_s

    # -- TCP флаги --
    fin = df["FIN Flag Count"].clip(lower=0)
    syn = df["SYN Flag Count"].clip(lower=0)
    rst = df["RST Flag Count"].clip(lower=0)
    psh = df["PSH Flag Count"].clip(lower=0)
    ack = df["ACK Flag Count"].clip(lower=0)
    urg = df["URG Flag Count"].clip(lower=0)
    ece = df["ECE Flag Count"].clip(lower=0)
    cwr = df["CWE Flag Count"].clip(lower=0)  # CIC-IDS-2018 называет CWE

    # -- Протоколы из поля Protocol --
    tcp  = (protocol == 6).astype(float)
    udp  = (protocol == 17).astype(float)
    icmp = (protocol == 1).astype(float)

    # -- Статистика размеров пакетов --
    pkt_min = df["Packet Length Min"].clip(lower=0)
    pkt_max = df["Packet Length Max"].clip(lower=0)
    pkt_avg = df["Packet Length Mean"].clip(lower=0)
    pkt_std = df["Packet Length Std"].clip(lower=0)
    pkt_var = df["Packet Length Variance"].clip(lower=0)

    # -- Производные статистики (как в CIC IoT 2023) --
    fwd_mean = df["Fwd Packet Length Mean"].fillna(0).clip(lower=0)
    bwd_mean = df["Bwd Packet Length Mean"].fillna(0).clip(lower=0)
    fwd_std  = df["Fwd Packet Length Std"].fillna(0).clip(lower=0)
    bwd_std  = df["Bwd Packet Length Std"].fillna(0).clip(lower=0)

    magnitude = np.sqrt(fwd_mean ** 2 + bwd_mean ** 2)
    radius    = np.sqrt((fwd_std ** 2 + bwd_std ** 2) / 2.0)
    # Covariance недоступна из агрегированных статистик -> 0
    covariance = pd.Series(0.0, index=df.index)
    weight = total_pkts / flow_dur_s  # пакеты/сек (идентично Rate)

    # IAT — Flow IAT Mean в микросекундах (CIC-IDS-2018 тоже в мкс)
    iat = df["Flow IAT Mean"].fillna(0).clip(lower=0)

    out = pd.DataFrame({
        "flow_duration":   flow_dur_s,
        "Header_Length":   header_len,
        "Protocol Type":   protocol,
        "Duration":        flow_dur_s,
        "Rate":            rate,
        "Srate":           srate,
        "Drate":           drate,
        "fin_flag_number": fin,
        "syn_flag_number": syn,
        "rst_flag_number": rst,
        "psh_flag_number": psh,
        "ack_flag_number": ack,
        "ece_flag_number": ece,
        "cwr_flag_number": cwr,
        "ack_count":       ack,
        "syn_count":       syn,
        "fin_count":       fin,
        "urg_count":       urg,
        "rst_count":       rst,
        # Port-based flags: нет port info в CIC-IDS-2018 → 0
        "HTTP":    0.0, "HTTPS":  0.0, "DNS":    0.0,
        "Telnet":  0.0, "SMTP":   0.0, "SSH":    0.0, "IRC": 0.0,
        "TCP":     tcp,
        "UDP":     udp,
        "DHCP":    0.0, "ARP":    0.0,
        "ICMP":    icmp,
        "IPv":     4.0, "LLC":    0.0,
        "Tot sum":     tot_sum,
        "Min":         pkt_min,
        "Max":         pkt_max,
        "AVG":         pkt_avg,
        "Std":         pkt_std,
        "Tot size":    tot_sum,
        "IAT":         iat,
        "Number":      total_pkts.astype(float),
        "Magnitue":    magnitude,   # опечатка намеренна — соответствует датасету
        "Radius":      radius,
        "Covariance":  covariance,
        "Variance":    pkt_var,
        "Weight":      weight,
    })

    # Проверяем порядок колонок
    assert list(out.columns) == FEATURES_2023, \
        f"Column mismatch: {set(out.columns) ^ set(FEATURES_2023)}"

    return out.astype("float32")


# ── Загрузка файлов ────────────────────────────────────────────
files = sorted(CIC2018_DIR.glob("*.parquet"))
print(f"Найдено файлов: {len(files)}")

frames = []
label_counts: dict[int, int] = {i: 0 for i in range(8)}

for fp in files:
    print(f"\n  Читаем {fp.name} ...")
    df = pd.read_parquet(fp)

    # Маппинг меток
    raw_labels = df["Label"].apply(map_label)
    mask = raw_labels.notna()
    df = df[mask].copy()
    labels = raw_labels[mask].astype("int8")

    if len(df) == 0:
        print(f"    Нет подходящих строк, пропуск")
        continue

    # Конвертируем признаки
    try:
        X = convert_to_cic2023(df)
    except Exception as e:
        print(f"    Ошибка конвертации: {e}")
        import traceback; traceback.print_exc()
        continue

    X["target"] = labels.values
    frames.append(X)

    cls_dist = labels.value_counts().to_dict()
    for cls_id, cnt in cls_dist.items():
        label_counts[cls_id] = label_counts.get(cls_id, 0) + cnt
    print(f"    {len(X):,} строк | классы: { {CLASS_NAMES[k]: v for k, v in cls_dist.items()} }")

if not frames:
    print("Нет данных для сохранения")
    sys.exit(1)

out_df = pd.concat(frames, ignore_index=True)
print(f"\nИтого строк: {len(out_df):,}")
print("Распределение классов:")
for cls_id, cnt in sorted(label_counts.items()):
    if cnt > 0:
        print(f"  {cls_id} {CLASS_NAMES[cls_id]:12s}: {cnt:>10,}")

# Очищаем inf/NaN
out_df = out_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

DATA_DIR.mkdir(parents=True, exist_ok=True)
out_path = DATA_DIR / "cic2018_mapped.parquet"
out_df.to_parquet(out_path, index=False, compression="zstd")
print(f"\n>> Сохранено: {out_path}")
print("[OK] Шаг 1 завершён")
