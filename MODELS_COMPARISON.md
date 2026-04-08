# AnomalyNet — Сравнение моделей

Три итерации модели с разным охватом задачи, источниками данных и метриками.
В приложении реализованы два каскадных режима детекции: **Simple** и **Advanced**.

---

## Dual-mode архитектура детекции

```
Режим Simple (catboost-cascade-simple):
  Трафик → FlowAggregator
        → compute_cicflow_features() [71 признак]
              ↓
         Stage1 binary (F1=99.4%)
              ↓ если attack
         Stage2 multiclass (Macro F1=0.31) → attack_class: DoS/DDoS/...

Режим Advanced (catboost-cascade-advanced):
  Трафик → FlowAggregator
        → compute_cicflow_features() [71 признак]  → Stage1 binary
        → compute_cic2023_features() [46 признаков] ↓ если attack
                                              Stage3 IoT2023 (Macro F1=0.82) → attack_class
```

**Почему каскад, а не одна модель:**
Stage1 (бинарный) обеспечивает очень высокий recall (98.9%) на уровне "есть атака / нет".
Stage2/Stage3 добавляют классификацию типа атаки поверх.
Если Stage3 не может вычислить 46 признаков (слишком короткий поток) — результат Stage1 используется как fallback.

**Ключевое ограничение Advanced:** Stage3 обучена на CIC IoT 2023 датасете
с другим feature space. Признаки вычислены из того же FlowRecord, но по другой формуле.
Это единственный вариант без полной переработки инфраструктуры захвата трафика.

### Настройки в приложении

Simple:
```json
{
  "active_model_id": "catboost-cascade-simple",
  "detection_mode": "simple",
  "catboost_model_dir": "G:/Диплом/IoT/stage1_v2_cl/models/catboost",
  "catboost_secondary_model_dir": "G:/Диплом/IoT/stage2_multiclass/models/catboost",
  "catboost_secondary_artifacts_dir": ""
}
```

Advanced:
```json
{
  "active_model_id": "catboost-cascade-advanced",
  "detection_mode": "advanced",
  "catboost_model_dir": "G:/Диплом/IoT/stage1_v2_cl/models/catboost",
  "catboost_secondary_model_dir": "G:/Диплом/IoT/stage3_cic2023/models/catboost",
  "catboost_secondary_artifacts_dir": "G:/Диплом/IoT/stage3_cic2023/artifacts"
}
```

---

---

## Быстрое сравнение

| Параметр | Stage 1 (binary) | Stage 2 (multiclass) | Stage 3 (CIC 2023) |
|---|---|---|---|
| **Задача** | Benign / Attack | 8 классов атак | 8 классов атак |
| **Датасет** | CIC IoT 2024 | CIC IoT 2024 + IDS2018 + IDS2017 | CIC IoT 2023 |
| **Объём train** | ~4.2M строк | ~1.1M (после cap) | ~863K |
| **Признаков** | 71 (CICFlowMeter) | 71 (CICFlowMeter) | 46 (IoT-специфичные) |
| **Test Accuracy** | 98.85% | 95.19% | **99.49%** |
| **Test Macro F1** | — (бинарная) | 0.308 | **0.819** |
| **Test Weighted F1** | **99.41%** | 96.58% | 99.51% |
| **Готова к live** | Да | Да | Нет (разный feature space) |

---

## Детальные метрики на тестовой выборке

### Stage 1 — Бинарный детектор (IoT 2024)

> **Задача**: Benign vs Attack (2 класса). Признаки: 71 CICFlowMeter.

| Метрика | Значение |
|---|---|
| Accuracy | 98.85% |
| Precision (attack) | 99.97% |
| Recall (attack) | 98.87% |
| F1 (attack) | **99.41%** |
| ROC-AUC | 99.63% |
| PR-AUC | 99.99% |

**Вывод**: Очень высокая точность бинарного разделения. 1.13% атак пропускается (FN).

---

### Stage 2 — Многоклассовый детектор (IoT 2024 + внешние данные)

> **Задача**: 8 классов. Признаки: 71 CICFlowMeter. Аугментация: CIC-IDS2018 + CIC-IDS2017.

| Класс | Precision | Recall | F1 | Support (test) |
|---|---|---|---|---|
| Benign | 0.349 | 0.022 | **0.041** | 38,895 |
| DoS | 0.999 | 0.974 | **0.987** | 3,503,938 |
| DDoS | 0.989 | 0.886 | **0.935** | 394,910 |
| Recon | — | — | **0.000** | 0 |
| BruteForce | — | — | **0.000** | 0 |
| WebAttack | 0.004 | 0.030 | **0.008** | 1,348 |
| Bot | 0.285 | 0.504 | **0.364** | 26,482 |
| Spoofing | 0.075 | 0.618 | **0.133** | 9,349 |

**Macro F1: 0.308** | **Weighted F1: 0.966**

> **Проблема**: Recon и BruteForce отсутствуют в тестовой выборке — stage1 строил сплиты
> по файлам (по дням атак), и все файлы Recon/BruteForce попали в train.
> Следствие: модель их не обнаруживает в production.

---

### Stage 3 — Многоклассовый детектор (CIC IoT 2023)

> **Задача**: 8 классов. Признаки: 46 IoT-специфичных. Датасет 2023: 47M строк, 34 класса.
> Стратифицированный split 70/15/15 гарантирует все классы в test.

| Класс | Precision | Recall | F1 | Support (test) |
|---|---|---|---|---|
| Benign | 0.939 | 0.961 | **0.950** | 164,729 |
| DoS | 0.999 | 1.000 | **0.999** | 1,213,611 |
| DDoS | 1.000 | 0.999 | **0.999** | 5,097,684 |
| Recon | 0.802 | 0.872 | **0.835** | 53,185 |
| BruteForce | 0.406 | 0.703 | **0.515** | 1,959 |
| WebAttack | 0.273 | 0.700 | **0.393** | 3,242 |
| Bot | 0.999 | 0.999 | **0.999** | 395,601 |
| Spoofing | 0.945 | 0.795 | **0.864** | 72,976 |

**Test Accuracy: 99.49%** | **Macro F1: 0.819** | **Weighted F1: 0.995**

---

## Сравнение по классам (Stage 2 vs Stage 3, F1)

| Класс | Stage 2 F1 | Stage 3 F1 | Разница |
|---|---|---|---|
| Benign | 0.041 | **0.950** | +0.909 |
| DoS | 0.987 | **0.999** | +0.012 |
| DDoS | 0.935 | **0.999** | +0.065 |
| Recon | 0.000 | **0.835** | +0.835 |
| BruteForce | 0.000 | **0.515** | +0.515 |
| WebAttack | 0.008 | **0.393** | +0.385 |
| Bot | 0.364 | **0.999** | +0.635 |
| Spoofing | 0.133 | **0.864** | +0.731 |

Stage 3 превосходит Stage 2 по всем классам. Главная причина — правильный стратифицированный
сплит и более чистый датасет (2023 датасет специально собирался для IoT-атак).

---

## Наборы признаков

### 71 признак (Stage 1 + Stage 2) — CICFlowMeter

Вычисляются из сырого трафика (pcap/live) с помощью flow aggregator:

```
Flow Duration, Total Fwd/Bwd Packets, Total Length Fwd/Bwd,
Fwd/Bwd Packet Length Max/Min/Mean/Std,
Flow Bytes/s, Flow Packets/s,
Flow IAT Mean/Std/Max/Min,
Fwd/Bwd IAT Total/Mean/Std/Max/Min,
Fwd PSH/URG Flags, Bwd PSH/URG Flags,
Fwd/Bwd Header Length, Fwd/Bwd Packets/s,
Min/Max/Mean Packet Length, Packet Length Std/Variance,
FIN/SYN/RST/PSH/ACK/URG/CWE/ECE Flag Count,
Down/Up Ratio, Average Packet Size,
Avg Fwd/Bwd Segment Size, Fwd/Bwd Avg Bytes/Bulk, ...
Subflow Fwd/Bwd Packets/Bytes,
Init Win Bytes Fwd/Bwd, Act Data Pkt Fwd, Min Seg Size Fwd,
Active Mean/Std/Max/Min, Idle Mean/Std/Max/Min
```

**Совместимость с live capture**: Полностью совместимы. `feature_computer.py` в приложении
вычисляет все 71 признак из перехваченного трафика.

### 46 признаков (Stage 3) — IoT-специфичные

Вычисляются другим инструментом (не CICFlowMeter):

```
flow_duration, Header_Length, Protocol Type, Duration,
Rate, Srate, Drate,
fin_flag_number, syn_flag_number, rst_flag_number,
psh_flag_number, ack_flag_number, ece_flag_number, cwr_flag_number,
ack_count, syn_count, fin_count, urg_count, rst_count,
HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC,
TCP, UDP, DHCP, ARP, ICMP, IPv, LLC,
Tot sum, Min, Max, AVG, Std, Tot size,
IAT, Number, Magnitue, Radius, Covariance, Variance, Weight
```

**Совместимость с live capture**: Несовместимы с текущим feature_computer.py.
Для использования в приложении требуется разработка нового экстрактора признаков.

---

## Архитектура для live-детекции

### Текущая реализация в приложении

```
Трафик → FlowAggregator → feature_computer.py → [71 признаков]
                                                        ↓
                                                Stage2 (MultiClass)
                                                        ↓
                                        attack_class: DoS/DDoS/Bot/...
```

### Ограничения

Модель Stage 2 работает в приложении, но имеет слабые метрики на:
- **Recon**: не детектируется (F1=0.000) — нет примеров в test split
- **BruteForce**: не детектируется (F1=0.000) — та же причина
- **WebAttack, Bot, Spoofing**: плохой recall/precision

### Рекомендация для диплома

В дипломной работе можно честно описать ситуацию:
1. Stage 2 используется в production (совместим с feature_computer)
2. Stage 3 демонстрирует значительно лучшие метрики на аналогичных 8 классах
3. Для перехода на Stage 3 нужен новый экстрактор признаков (46 IoT-признаков)
4. Stage 3 является целевым ориентиром (Macro F1=0.819 vs 0.308)

---

## Источники данных

| Датасет | Год | Трафик | Записей | Классов | Используется в |
|---|---|---|---|---|---|
| CIC IoT 2024 | 2024 | IoT устройства | ~16M | 8+ | Stage 1, Stage 2 |
| CSE-CIC-IDS2018 | 2018 | Корпоративная сеть | ~6.6M | 14 | Stage 2 (аугментация) |
| CIC-IDS2017 | 2017 | Корпоративная сеть | ~2.6M | 14 | Stage 2 (аугментация) |
| CIC IoT 2023 | 2023 | IoT устройства | ~47M | 34 | Stage 3 |

---

## Параметры обучения CatBoost

### Stage 1 (бинарный)
```python
iterations=5000, learning_rate=0.05, depth=8,
loss_function="Logloss", eval_metric="F1",
task_type="GPU", early_stopping_rounds=200
```

### Stage 2 / Stage 3 (многоклассовый)
```python
iterations=3000, learning_rate=0.05, depth=8,
loss_function="MultiClass", eval_metric="TotalF1",
task_type="GPU", early_stopping_rounds=150,
class_weights=[1, 1, 1, 2, 2, 2, 1.5, 2]  # выше веса для редких классов
```

Stage 3: best_iteration=2996, train_time=58с на RTX 4070 Ti Super.

---

## Файловая структура артефактов

```
AnomalyNet-ml/
├── model/catboost/          # Stage 1: бинарный
│   ├── model.cbm
│   ├── feature_contract.json
│   └── metrics.json
├── stage2_multiclass/       # Stage 2: 8-классовый (IoT2024 + внешние)
│   └── models/catboost/
│       ├── model_mc.cbm
│       ├── class_mapping.json
│       └── metrics_mc.json
└── stage3_cic2023/          # Stage 3: 8-классовый (IoT2023, лучшие метрики)
    └── models/catboost/
        ├── model_mc.cbm
        ├── class_mapping.json
        └── metrics_mc.json
```
