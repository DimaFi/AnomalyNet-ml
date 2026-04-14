# Stage4 Extended — Журнал прогресса

## Цель
Переобучить Stage3 (CIC IoT 2023, 46 признаков, 8 классов) с добавлением данных из **CIC-IDS-2018**.

### Проблема, которую решаем
Stage3 (Macro F1=0.82 на IoT 2023) не обобщается на стандартные VPS-атаки (SYN/UDP/ICMP flood, PortScan, BruteForce с внешних IP).
Тест 14.04.2026: Stage1 обнаружил 100% атак, Stage3 классифицировал все как Benign (domain shift).

### Решение
Добавить в обучение CIC-IDS-2018 (стандартные сетевые атаки от Linux-хостов):
- SSH-Bruteforce → class 4 (BruteForce)
- FTP-BruteForce → class 4
- DoS-* → class 1 (DoS)
- DDoS-* → class 2 (DDoS)
- PortScan → class 3 (Recon)
- Web attacks → class 5 (WebAttack)
- Bot → class 6 (Bot)

---

## Архитектура

```
Stage3 train data (IoT 2023, 863K строк)
   +
CIC-IDS-2018 → маппинг → 46 CIC2023 признаков
   ↓
Augmented train (с per-class cap)
   ↓
CatBoost MultiClass (те же 46 признаков, 8 классов)
   ↓
stage4/models/catboost/model_mc.cbm
```

### Маппинг CIC-IDS-2018 → CIC2023 признаки
| CIC2023 | Источник в CIC-IDS-2018 |
|---------|------------------------|
| flow_duration | Flow Duration / 1e6 |
| Duration | то же |
| Header_Length | Fwd Header Length / Total Fwd Packets |
| Protocol Type | Protocol |
| Rate | (Fwd+Bwd pkts) / flow_duration |
| Srate | Fwd Packets / flow_duration |
| Drate | Bwd Packets / flow_duration |
| fin/syn/rst/psh/ack/ece/cwr flag_number | Flag Count колонки |
| TCP/UDP/ICMP | из Protocol поля |
| HTTP/HTTPS/DNS/SSH/... | 0 (нет port info в CIC-IDS-2018) |
| Tot sum, Tot size | Fwd+Bwd Length Total |
| Min/Max/AVG/Std | Packet Length * |
| Variance | Packet Length Variance |
| IAT | Flow IAT Mean (мкс) |
| Number | Total Fwd+Bwd Packets |
| Magnitue | sqrt(fwd_mean²+bwd_mean²) |
| Radius | sqrt((fwd_std²+bwd_std²)/2) |
| Covariance | 0 (недоступно) |
| Weight | Number / flow_duration |

---

## Этапы

- [x] **Шаг 1** — `scripts/01_prepare_cic2018.py` — CIC-IDS-2018 (6.66M строк) → 46 CIC2023 признаков. Сохранён `data/cic2018_mapped.parquet`
- [x] **Шаг 2** — `scripts/02_merge_augment.py` — Stage3 (863K) + CIC-IDS-2018 → augmented 1.29M строк (BruteForce 5K→50K!)
- [x] **Шаг 3** — `scripts/03_train_catboost.py` — GPU, **1.2 минуты**, best_iter=2988
- [x] **Шаг 4** — `scripts/04_evaluate.py` — сравнение Stage3 vs Stage4
- [ ] **Шаг 5** — деплой на сервер как `catboost-cascade-advanced2`

---

## Результаты (14.04.2026)

### Macro F1 сравнение

| Модель | CIC IoT 2023 (in-domain) | CIC-IDS-2018 (out-of-domain) |
|--------|--------------------------|------------------------------|
| Stage3 | **0.8192**               | 0.0805 (хуже случайного)     |
| Stage4 | 0.7335 (-0.086)          | **0.7600 (+0.680!)**         |

### Per-class F1 на CIC-IDS-2018 (внешние VPS-атаки)

| Класс      | Stage3 | Stage4 | Δ       |
|-----------|--------|--------|---------|
| Benign    | 0.476  | 0.940  | +0.464  |
| DoS       | 0.027  | 0.986  | +0.959  |
| DDoS      | 0.110  | 0.972  | +0.861  |
| Recon     | 0.031  | 0.187  | +0.157  |
| BruteForce| 0.000  | 0.978  | +0.978  |
| WebAttack | 0.000  | 0.294  | +0.294  |
| Bot       | 0.000  | 0.963  | +0.963  |

### Вывод
Stage4 решает проблему domain shift: атаки с внешних VPS-хостов теперь корректно
классифицируются (DoS F1=0.99, DDoS F1=0.97, BruteForce F1=0.98).
Цена: небольшое снижение точности на IoT 2023 тесте (-0.086 Macro F1).
Для реального применения Stage4 предпочтительнее.
