# Анализ датасета CIC IoT Dataset 2023

**Источник**: Canadian Institute for Cybersecurity, University of New Brunswick  
**Путь**: `G:\Диплом\IoT\CIC IoT Dataset 2023`  
**Год**: 2023  
**Объём**: ~47 миллионов сетевых потоков

---

## Общая характеристика

CIC IoT 2023 — специализированный датасет для задач обнаружения атак в IoT-среде.
В отличие от CIC-IDS2017/2018, которые имитируют корпоративную сеть, этот датасет
собирался на реальных IoT-устройствах: умные камеры, термостаты, смарт-телевизоры,
умные замки, медицинские сенсоры и другие.

---

## Структура файлов

Датасет поставляется в виде директорий по типам атак, каждая содержит CSV-файлы:

```
CIC IoT Dataset 2023/
├── BenignTraffic.csv                    # нормальный трафик (~164K потоков)
├── DDoS/
│   ├── DDoS-ICMP_Flood.csv
│   ├── DDoS-UDP_Flood.csv
│   ├── DDoS-TCP_Flood.csv
│   ├── DDoS-PSH-ACK_Flood.csv
│   ├── DDoS-SYN_Flood.csv
│   ├── DDoS-RSTFINFlood.csv
│   ├── DDoS-SynonymousIP_Flood.csv
│   ├── DDoS-ICMP_Fragmentation.csv
│   ├── DDoS-UDP_Fragmentation.csv
│   ├── DDoS-ACK_Fragmentation.csv
│   ├── DDoS-HTTP_Flood.csv
│   └── DDoS-SlowLoris.csv              # 12 подтипов DDoS
├── DoS/
│   ├── DoS-UDP_Flood.csv
│   ├── DoS-TCP_Flood.csv
│   ├── DoS-SYN_Flood.csv
│   └── DoS-HTTP_Flood.csv              # 4 подтипа DoS
├── Recon/
│   ├── Recon-HostDiscovery.csv
│   ├── Recon-OSScan.csv
│   ├── Recon-PortScan.csv
│   └── Recon-PingSweep.csv             # + VulnerabilityScan
├── Web-Based/
│   ├── SqlInjection.csv
│   ├── XSS.csv
│   └── CommandInjection.csv
├── BruteForce/
│   └── DictionaryBruteForce.csv
├── Spoofing/
│   ├── MITM-ArpSpoofing.csv
│   └── DNS_Spoofing.csv
└── Mirai/
    ├── Mirai-greeth_flood.csv
    ├── Mirai-udpplain.csv
    └── Mirai-udp_flood.csv
```

Итого: **169 CSV-файлов** (включая подпапки, каждая атака может разбита на несколько файлов).

---

## 34 исходных класса

| # | Исходная метка | Унифицированный класс | Количество записей |
|---|---|---|---|
| 1 | BenignTraffic | Benign (0) | ~658K |
| 2 | DDoS-ICMP_Flood | DDoS (2) | ~5.8M |
| 3 | DDoS-UDP_Flood | DDoS (2) | ~5.8M |
| 4 | DDoS-TCP_Flood | DDoS (2) | ~5.8M |
| 5 | DDoS-PSH-ACK_Flood | DDoS (2) | ~5.8M |
| 6 | DDoS-SYN_Flood | DDoS (2) | ~5.8M |
| 7 | DDoS-RSTFINFlood | DDoS (2) | ~5.8M |
| 8 | DDoS-SynonymousIP_Flood | DDoS (2) | ~5.8M |
| 9 | DDoS-ICMP_Fragmentation | DDoS (2) | ~180K |
| 10 | DDoS-UDP_Fragmentation | DDoS (2) | ~180K |
| 11 | DDoS-ACK_Fragmentation | DDoS (2) | ~180K |
| 12 | DDoS-HTTP_Flood | DDoS (2) | ~180K |
| 13 | DDoS-SlowLoris | DDoS (2) | ~180K |
| 14 | DoS-UDP_Flood | DoS (1) | ~1.5M |
| 15 | DoS-TCP_Flood | DoS (1) | ~1.5M |
| 16 | DoS-SYN_Flood | DoS (1) | ~1.5M |
| 17 | DoS-HTTP_Flood | DoS (1) | ~1.5M |
| 18 | Recon-HostDiscovery | Recon (3) | ~90K |
| 19 | Recon-OSScan | Recon (3) | ~90K |
| 20 | Recon-PortScan | Recon (3) | ~90K |
| 21 | Recon-PingSweep | Recon (3) | ~65K |
| 22 | VulnerabilityScan | Recon (3) | ~30K |
| 23 | SqlInjection | WebAttack (5) | ~7K |
| 24 | XSS | WebAttack (5) | ~7K |
| 25 | CommandInjection | WebAttack (5) | ~7K |
| 26 | DictionaryBruteForce | BruteForce (4) | ~13K |
| 27 | MITM-ArpSpoofing | Spoofing (7) | ~200K |
| 28 | DNS_Spoofing | Spoofing (7) | ~90K |
| 29 | Mirai-greeth_flood | Bot (6) | ~530K |
| 30 | Mirai-udpplain | Bot (6) | ~530K |
| 31 | Mirai-udp_flood | Bot (6) | ~530K |

---

## Распределение по унифицированным классам (train/val/test)

| Класс | Total | Train | Val | Test |
|---|---|---|---|---|
| **Benign** | ~658K | 461K | 99K | 99K |
| **DoS** | ~4.85M | 800K (cap) | 1.21M | 1.21M |
| **DDoS** | ~33.9M | 800K (cap) | 5.10M | 5.10M |
| **Recon** | ~354K | 248K | 53K | 53K |
| **BruteForce** | ~13K | 9.2K | 2.0K | 2.0K |
| **WebAttack** | ~21K | 14.7K | 3.2K | 3.2K |
| **Bot** | ~1.59M | 800K (cap) | 396K | 396K |
| **Spoofing** | ~292K | 204K | 73K | 73K |

> Train-сплит применяет ограничение (cap=800K) на классы DoS/DDoS/Bot
> для балансировки обучения. Val и Test не ограничиваются.

---

## Особенности feature space

### 46 признаков (из 47 заявленных)

Датасет использует нестандартный набор признаков, разработанный специально для IoT-трафика.
Один признак (`Magnitude`) имеет опечатку в оригинальном файле → `Magnitue` (так и оставлено).

**Статистические признаки потока**:
- `flow_duration` — длительность потока
- `Rate`, `Srate`, `Drate` — общая скорость, скорость в src→dst, dst→src направлении
- `IAT` — межпакетный интервал
- `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size` — статистика размеров пакетов
- `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight` — корреляционные признаки
- `Number` — количество пакетов

**Флаги TCP**:
- `fin_flag_number`, `syn_flag_number`, `rst_flag_number`
- `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`
- `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`

**Сетевые протоколы (бинарные флаги)**:
- `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`
- `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IPv`, `LLC`

**Прочее**:
- `Header_Length` — длина заголовка
- `Protocol Type` — числовой код протокола
- `Duration` (отличается от `flow_duration`)

### Сравнение с CICFlowMeter (71 признак)

| Категория | CIC 2023 (46) | CICFlowMeter (71) |
|---|---|---|
| Длительность потока | flow_duration, Duration | Flow Duration |
| Пакеты Fwd/Bwd | Rate, Srate, Drate | Total Fwd/Bwd Packets + отдельно |
| Размеры пакетов | Tot sum, Min, Max, AVG, Std | Fwd/Bwd Length + Std/Variance |
| TCP флаги | 13 признаков | FIN/SYN/RST/PSH/ACK/URG/CWE/ECE counts |
| IAT | IAT (одно значение) | IAT Mean/Std/Max/Min Fwd+Bwd |
| Active/Idle | нет | Active/Idle Mean/Std/Max/Min |
| Протоколы | бинарные флаги | нет |
| Bulk-метрики | нет | Fwd/Bwd Avg Bytes/Bulk |

**Вывод**: Принципиально разные наборы. Нельзя просто переименовать колонки.
Для использования stage3 в live-приложении нужен новый экстрактор, вычисляющий
эти 46 признаков из перехваченного трафика.

---

## Качество данных

### Проблемы
- **Дисбаланс**: DDoS составляет ~70% датасета (33.9M из 47M записей)
- **Редкие классы**: BruteForce — всего 13K (0.028%), WebAttack — 21K (0.045%)
- **Opaque labeling**: исходные метки чувствительны к регистру и пробелам
  (использовался `.str.strip().str.lower()` при загрузке)

### Решения применённые в pipeline
1. **Стратифицированный split** `train_test_split(..., stratify=target_mc)` — гарантирует
   представленность всех 8 классов в val и test
2. **Per-class cap** в train: `CLASS_SAMPLE_CAP = {DoS: 800K, DDoS: 800K, Bot: 800K}` —
   предотвращает доминирование мажоритарных классов
3. **Повышенные веса** для редких классов: `class_weights=[1, 1, 1, 2, 2, 2, 1.5, 2]`

---

## Результаты обученной модели (Stage 3)

```
Test Accuracy:    99.49%
Test Macro F1:     0.8192  ← ключевой показатель (все классы равноправны)
Test Weighted F1: 99.51%

Лучшие классы: DoS=0.999, DDoS=0.999, Bot=0.999, Benign=0.950, Spoofing=0.864
Слабые классы: Recon=0.835, BruteForce=0.515, WebAttack=0.393
```

Слабость BruteForce и WebAttack объясняется крайне малым числом примеров (<2K в test),
что делает оценку неустойчивой. Модель видит эти классы и имеет recall ~0.70,
но precision падает из-за большого числа ложных срабатываний среди похожего трафика.

---

## Сравнение с датасетом CIC IoT 2024 (stage1/stage2)

| Параметр | CIC IoT 2023 | CIC IoT 2024 |
|---|---|---|
| Объём | ~47M | ~16M |
| Признаков | 46 (IoT-специфичные) | 71 (CICFlowMeter) |
| Классов исходных | 34 | 8+ |
| BruteForce | 13K (0.028%) | <5K |
| WebAttack | 21K (0.045%) | <8K |
| Совместимость с live | Нет (новый экстрактор) | Да (feature_computer.py) |

---

## Ссылки

- [CIC IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) — официальная страница UNB
- Статья: Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment", 2023
