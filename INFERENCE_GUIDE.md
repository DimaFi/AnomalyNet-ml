# Руководство по инференсу: IoT IDS (CatBoost, бинарная классификация)

Модель определяет, является ли сетевой поток **атакой** или **нормальным трафиком**.

---

## 1. Быстрый старт (минимальный пример)

```python
import json
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from src.preprocessing import preprocess_for_inference

ROOT = Path("G:/Диплом/IoT/stage1_v2_cl")
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ROOT / "models/catboost/model.cbm"
THRESHOLD  = 0.70   # оптимальный порог

# 1. Загрузить модель (один раз)
model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

# 2. Подготовить данные (DataFrame с нужными 71 признаком)
df = pd.read_csv("your_traffic.csv")   # или pd.DataFrame(rows)

# 3. Предобработка (inf→NaN→заполнение медианами из обучения)
X = preprocess_for_inference(df, ARTIFACTS)

# 4. Предсказание
proba = model.predict_proba(X.values)[:, 1]   # вероятность атаки
labels = (proba >= THRESHOLD).astype(int)      # 0 = Benign, 1 = Attack

print(labels)   # массив 0/1 для каждого потока
```

---

## 2. Как получить признаки из реального трафика

Модель работает с **признаками сетевых потоков**, которые вычисляет инструмент **CICFlowMeter**.

### 2.1 Установка CICFlowMeter

1. Скачать: [github.com/ahlashkari/CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter) (releases → .jar)
2. Требуется Java 8+: `java -version`
3. Запуск: `java -jar CICFlowMeter.jar`

### 2.2 Захват трафика в реальном времени

```
CICFlowMeter → вкладка "NetWork" → выбрать сетевой интерфейс → Start
```

Результат: CSV-файл с потоками, обновляется каждые ~60 секунд.

### 2.3 Захват из готового pcap-файла

```bash
java -jar CICFlowMeter.jar -f traffic.pcap -c output_folder/
```

Или через GUI: вкладка "Offline" → выбрать .pcap файл.

### 2.4 Формат выходного файла

CICFlowMeter создаёт CSV с заголовком. Нужные колонки совпадают с именами модели.  
Лишние колонки (Flow ID, Source IP, Destination IP, Timestamp, Source Port, Destination Port) — можно оставить, `preprocess_for_inference` их проигнорирует.

---

## 3. Входные данные модели: 71 признак

Каждая строка = один двунаправленный сетевой поток (5-tuple: src_ip, dst_ip, src_port, dst_port, protocol).

| # | Признак | Тип | Описание |
|---|---------|-----|----------|
| 1 | Protocol | int | Протокол (6=TCP, 17=UDP, 0=...) |
| 2 | Flow Duration | float | Длительность потока (мкс) |
| 3 | Total Fwd Packet | int | Число пакетов src→dst |
| 4 | Total Bwd packets | int | Число пакетов dst→src |
| 5 | Total Length of Fwd Packet | float | Суммарный размер пакетов src→dst (байт) |
| 6 | Total Length of Bwd Packet | float | Суммарный размер пакетов dst→src (байт) |
| 7 | Fwd Packet Length Max | float | Макс. размер пакета src→dst |
| 8 | Fwd Packet Length Min | float | Мин. размер пакета src→dst |
| 9 | Fwd Packet Length Mean | float | Средний размер пакета src→dst |
| 10 | Fwd Packet Length Std | float | Ст.откл. размера пакета src→dst |
| 11 | Bwd Packet Length Max | float | Макс. размер пакета dst→src |
| 12 | Bwd Packet Length Min | float | Мин. размер пакета dst→src |
| 13 | Bwd Packet Length Mean | float | Средний размер пакета dst→src |
| 14 | Bwd Packet Length Std | float | Ст.откл. размера пакета dst→src |
| 15 | Flow Bytes/s | float | Байт/с для всего потока |
| 16 | Flow Packets/s | float | Пакетов/с для всего потока |
| 17 | Flow IAT Mean | float | Средний IAT между пакетами потока (мкс) |
| 18 | Flow IAT Std | float | Ст.откл. IAT потока |
| 19 | Flow IAT Max | float | Макс. IAT потока |
| 20 | Flow IAT Min | float | Мин. IAT потока |
| 21 | Fwd IAT Total | float | Суммарный IAT src→dst |
| 22 | Fwd IAT Mean | float | Средний IAT src→dst |
| 23 | Fwd IAT Std | float | Ст.откл. IAT src→dst |
| 24 | Fwd IAT Max | float | Макс. IAT src→dst |
| 25 | Fwd IAT Min | float | Мин. IAT src→dst |
| 26 | Bwd IAT Total | float | Суммарный IAT dst→src |
| 27 | Bwd IAT Mean | float | Средний IAT dst→src |
| 28 | Bwd IAT Std | float | Ст.откл. IAT dst→src |
| 29 | Bwd IAT Max | float | Макс. IAT dst→src |
| 30 | Bwd IAT Min | float | Мин. IAT dst→src |
| 31 | Fwd PSH Flags | int | Число PSH-флагов src→dst |
| 32 | Bwd PSH Flags | int | Число PSH-флагов dst→src |
| 33 | Fwd URG Flags | int | Число URG-флагов src→dst |
| 34 | Bwd URG Flags | int | Число URG-флагов dst→src |
| 35 | Fwd Header Length | int | Суммарная длина заголовков src→dst (байт) |
| 36 | Bwd Header Length | int | Суммарная длина заголовков dst→src (байт) |
| 37 | Fwd Packets/s | float | Пакетов/с src→dst |
| 38 | Bwd Packets/s | float | Пакетов/с dst→src |
| 39 | Packet Length Min | float | Мин. размер пакета в потоке |
| 40 | Packet Length Max | float | Макс. размер пакета в потоке |
| 41 | Packet Length Mean | float | Средний размер пакета |
| 42 | Packet Length Std | float | Ст.откл. размера пакета |
| 43 | Packet Length Variance | float | Дисперсия размера пакета |
| 44 | FIN Flag Count | int | Число пакетов с флагом FIN |
| 45 | SYN Flag Count | int | Число пакетов с флагом SYN |
| 46 | RST Flag Count | int | Число пакетов с флагом RST |
| 47 | PSH Flag Count | int | Число пакетов с флагом PSH |
| 48 | ACK Flag Count | int | Число пакетов с флагом ACK |
| 49 | URG Flag Count | int | Число пакетов с флагом URG |
| 50 | CWR Flag Count | int | Число пакетов с флагом CWR |
| 51 | ECE Flag Count | int | Число пакетов с флагом ECE |
| 52 | Down/Up Ratio | float | Отношение dst→src / src→dst пакетов |
| 53 | Average Packet Size | float | Средний размер пакета (байт) |
| 54 | Fwd Segment Size Avg | float | Средний размер сегмента src→dst |
| 55 | Bwd Segment Size Avg | float | Средний размер сегмента dst→src |
| 56 | Subflow Fwd Packets | int | Пакетов в субпотоке src→dst |
| 57 | Subflow Fwd Bytes | int | Байт в субпотоке src→dst |
| 58 | Subflow Bwd Packets | int | Пакетов в субпотоке dst→src |
| 59 | Subflow Bwd Bytes | int | Байт в субпотоке dst→src |
| 60 | FWD Init Win Bytes | int | Начальный размер окна TCP src→dst |
| 61 | Bwd Init Win Bytes | int | Начальный размер окна TCP dst→src |
| 62 | Fwd Act Data Pkts | int | Число пакетов с данными src→dst |
| 63 | Fwd Seg Size Min | int | Мин. размер сегмента src→dst |
| 64 | Active Mean | float | Средняя длительность активной фазы (мкс) |
| 65 | Active Std | float | Ст.откл. длительности активной фазы |
| 66 | Active Max | float | Макс. длительность активной фазы |
| 67 | Active Min | float | Мин. длительность активной фазы |
| 68 | Idle Mean | float | Средняя длительность простоя (мкс) |
| 69 | Idle Std | float | Ст.откл. длительности простоя |
| 70 | Idle Max | float | Макс. длительность простоя |
| 71 | Idle Min | float | Мин. длительность простоя |

> IAT = Inter-Arrival Time (время между пакетами), мкс = микросекунды

---

## 4. Шаги предобработки (детально)

`preprocess_for_inference()` выполняет строго те же шаги, что применялись к обучающим данным:

```
Входной DataFrame (71 колонка)
    │
    ▼  1. Проверка наличия всех 71 колонок (ValueError если что-то отсутствует)
    │
    ▼  2. inf → NaN  (бесконечности появляются при делении 0/0 в CICFlowMeter)
    │
    ▼  3. NaN → медиана (значения из training set, сохранены в artifacts/preprocessing_params.json)
    │
    ▼  Чистый DataFrame (0 NaN, 0 inf) — готов для модели
```

Ключевой момент: заполнение NaN происходит **медианами из обучающей выборки**, а не из текущих данных. Это гарантирует, что поведение при inference идентично обучению.

---

## 5. Полный пример для реального трафика

```python
import json
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from src.preprocessing import preprocess_for_inference

# --- Конфигурация ---
ROOT      = Path("G:/Диплом/IoT/stage1_v2_cl")
ARTIFACTS = ROOT / "artifacts"
MODEL     = ROOT / "models/catboost/model.cbm"
THRESHOLD = 0.70

# --- Загрузка модели (выполняется один раз при старте) ---
classifier = CatBoostClassifier()
classifier.load_model(str(MODEL))


def detect_traffic(csv_path: str) -> pd.DataFrame:
    """
    Принимает CSV от CICFlowMeter, возвращает DataFrame с колонками:
      - prediction: 0 (Benign) или 1 (Attack)
      - attack_proba: вероятность атаки (0.0–1.0)
    """
    df = pd.read_csv(csv_path)

    # Убрать пробелы в начале/конце имён колонок (баг некоторых версий CICFlowMeter)
    df.columns = df.columns.str.strip()

    # Предобработка
    X = preprocess_for_inference(df, ARTIFACTS)

    # Предсказание
    proba = classifier.predict_proba(X.values)[:, 1]
    pred  = (proba >= THRESHOLD).astype(int)

    result = df[["Flow Duration", "Protocol"]].copy()   # добавьте нужные мета-колонки
    result["attack_proba"] = np.round(proba, 4)
    result["prediction"]   = pred
    result["verdict"]      = result["prediction"].map({0: "BENIGN", 1: "ATTACK"})
    return result


# --- Запуск ---
results = detect_traffic("captured_traffic.csv")
print(results[results["prediction"] == 1])   # показать только атаки
```

---

## 6. Интерпретация результата

| prediction | attack_proba | Смысл |
|:---:|:---:|---|
| 0 | < 0.70 | Нормальный трафик (Benign) |
| 1 | ≥ 0.70 | **Атака обнаружена** |

**Почему порог 0.70, а не 0.50?**  
При пороге 0.50 ложных тревог (FP) было 34.4% от нормального трафика.  
При пороге 0.70 FP снизился до 3.3%, F1 упал лишь с 0.9964 до 0.9941.

**Что модель обнаруживает хорошо:**
- DoS-атаки (recall 99.6%)
- DDoS-атаки (recall 96.6%)

**Что обнаруживается хуже (~50% recall):**
- Mirai (IoT-ботнет)
- Spoofing
- Web-Based атаки

Для этих классов рекомендуется Stage 2: мультиклассовый классификатор только по строкам с атаками.

---

## 7. Адаптация других датасетов CICFlowMeter

Если данные получены из другого датасета или другой версии CICFlowMeter — используйте маппинг из `eval_external/feature_map.py`:

```python
from eval_external.feature_map import CIC2018_MAP, CIC2017_MAP, align_columns
import json
from pathlib import Path

FEATURE_COLUMNS = json.loads(
    (Path("artifacts/feature_contract.json")).read_text(encoding="utf-8")
)

# Для датасета CIC-IDS 2018:
X = align_columns(df_raw, CIC2018_MAP, FEATURE_COLUMNS)

# Для датасета CIC-IDS 2017:
X = align_columns(df_raw, CIC2017_MAP, FEATURE_COLUMNS)
```

### Таблица отличий имён колонок

| Признак (модель) | CIC-IDS 2018 | CIC-IDS 2017 |
|---|---|---|
| Total Fwd Packet | Total Fwd Packets | Total Fwd Packets |
| Total Bwd packets | Total Backward Packets | Total Backward Packets |
| Total Length of Fwd Packet | Fwd Packets Length Total | Total Length of Fwd Packets |
| Total Length of Bwd Packet | Bwd Packets Length Total | Total Length of Bwd Packets |
| Packet Length Min | Packet Length Min ✓ | Min Packet Length |
| Packet Length Max | Packet Length Max ✓ | Max Packet Length |
| CWR Flag Count | CWE Flag Count | CWE Flag Count |
| Average Packet Size | Avg Packet Size | Average Packet Size ✓ |
| Fwd Segment Size Avg | Avg Fwd Segment Size | Avg Fwd Segment Size |
| Bwd Segment Size Avg | Avg Bwd Segment Size | Avg Bwd Segment Size |
| FWD Init Win Bytes | Init Fwd Win Bytes | Init_Win_bytes_forward |
| Bwd Init Win Bytes | Init Bwd Win Bytes | Init_Win_bytes_backward |
| Fwd Act Data Pkts | Fwd Act Data Packets | act_data_pkt_fwd |
| Fwd Seg Size Min | Fwd Seg Size Min ✓ | min_seg_size_forward |

---

## 8. Структура артефактов

```
artifacts/
├── feature_contract.json      # Список 71 признака в нужном порядке
├── preprocessing_params.json  # Медианы для заполнения NaN (из train)
├── scaler.joblib              # StandardScaler (для будущих нейросетей)
└── inf_nan_report.json        # Диагностика inf/nan при обучении

models/catboost/
├── model.cbm                  # Обученная модель CatBoost
├── best_threshold.json        # Оптимальный порог: 0.70
├── metrics.json               # Метрики на val и test
└── feature_importance.json    # Важность признаков
```

---

## 9. Тест работоспособности

```bash
# Запустить из корня stage1_v2_cl
python eval_external/run_eval_cic2018.py
```

Ожидаемый вывод: метрики на CIC-IDS 2018, breakdown по типам атак, файл `eval_external/results/cic2018_eval.json`.
