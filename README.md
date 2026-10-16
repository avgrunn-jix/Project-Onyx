# Project Onyx: Explainable Exoplanet Detection Pipeline

Project Onyx is a modular, transparent, and explainable machine learning pipeline designed for the detection of exoplanet candidates from Kepler and TESS photometric light curves. 

Unlike black-box deep learning systems, Onyx focuses on **honest evaluation, feature explainability (SHAP), and robust anomaly detection**. It implements rigorous data leakage prevention, confidence-based abstention, and handles multi-planet systems.

## Key Features & Extensions (v2)

### 1. Robust Data Processing & Storage Efficiency
* **Local Caching & Disk Saving:** Downloading Kepler short/long cadence data generates massive amounts of FITS files. Onyx downloads each star into isolated temporary directories and automatically removes them after feature extraction.
* **Cadence-agnostic Flattening:** The detrending window length is defined in physical days (`--flatten-days=3.0`), rather than data points, ensuring transferability between Kepler (29.4 min) and TESS (2 min) cadence.
* **Universal Encoding Support:** Target list reading automatically detects and handles UTF-8, UTF-8 with BOM, and UTF-16 (LE/BE).

### 2. Multi-Signal Iterative Search
Instead of finding only the strongest transit, Onyx iteratively masks out detected transits and re-runs the Box Least Squares (BLS) periodogram up to 4 times per star. 
* Found signals are matched against the KOI/TOI catalogs. Matches are classified as `MATCHED`, `HARMONIC` (period ratio 0.5 or 2.0), or `UNMATCHED`. 
* **Note:** `UNMATCHED` signals are NOT treated as false positives. They are excluded from supervised training to prevent contamination but are evaluated in the prediction and anomaly detection phases.

### 3. Centroid Shift & False Positive Diagnostics
Onyx computes in-transit versus out-of-transit centroid shifts from light curve positional data (`centroid_col`, `centroid_row`) detrended per observation segment.
* Measures include `centroid_shift_col_sigma`, `centroid_shift_row_sigma`, `centroid_shift_total_sigma`, and physical displacement `centroid_shift_total_arcsec` (calibrated with Kepler ~3.98"/pixel and TESS ~21"/pixel scales).
* When rejecting a candidate, Onyx predicts the *likely reason* using a multi-label One-Vs-Rest XGBoost classifier trained on known FP flags (`nt`, `ss`, `co`, `ec`).

### 4. Explainability (SHAP)
Onyx employs `TreeExplainer` (Lundberg & Lee, 2017) to provide local and global interpretability.
* For every new prediction, Onyx outputs the **Top 3 contributing factors** (e.g., `secondary_depth_ppm=450 (negative)` meaning a deep secondary eclipse strongly pushed the model to reject the candidate).

### 5. Confidence and Abstention (Right to Refuse)
Onyx uses an ensemble of 20 XGBoost models trained on grouped bootstraps. It calculates `p_mean` and `p_std`. 
* If a signal falls into an ambiguous zone, the pipeline flags it as `REVIEW` instead of forcing a binary choice. Thresholds are automatically tuned on a validation set to ensure >95% precision for confident `CANDIDATE` and `FALSE_POSITIVE` calls.

### 6. Synthetic Transit Injections (Completeness Test)
To evaluate the true sensitivity of the pipeline, Onyx includes an `--inject` mode. It takes quiet stars and mathematically injects trapezoidal transits varying in radius and period.
* **Identical BLS Settings:** Synthetic signal recovery uses the exact same BLS grid and heuristics as real star extraction:
  * **Normal Grid (Default):** Period range $P \in [0.5, \min(100.0, \text{baseline}/2)]$ days; 10 duration steps linearly spaced in $[0.04, 0.45]$ days (~0.96 h to 10.8 h); frequency resolution $\Delta f = 500 \times \min(T) / \text{baseline}^2$ (`frequency_factor=500`).
  * **Quick Grid (`--quick`):** Period range $P \in [0.5, \min(100.0, \text{baseline}/2)]$ days; 4 duration steps in $[0.05, 0.40]$ days; frequency resolution $\Delta f = 5000 \times \min(T) / \text{baseline}^2$ (`frequency_factor=5000`). Automatically writes to outputs with `_quick` suffix and prints a warning.
* **Disclaimer:** Injections are idealized (no limb darkening, circular orbits, no transit starspots). Therefore, the resulting completeness heatmap represents an *upper bound* of the pipeline's true sensitivity.

### 7. Mission Transferability (Kepler → TESS)
We tested the model's ability to generalize across missions:
* **T1:** Direct transfer (All features).
* **T2:** Robust transfer (Removing `baseline_d`, `transit_count`, `gap_fraction`, which are heavily mission-biased).
* **T3:** In-domain cross-validation on TESS.
Onyx computes the Kolmogorov-Smirnov (K-S) shift for each feature (`outputs/figures/ks_shift_by_feature.png`) to identify covariate shift.

### 8. Unsupervised Anomaly Detection
Using Isolation Forest (Liu, Ting & Zhou, 2008) and Local Outlier Factor, Onyx scans all signals (including `UNMATCHED` and `HARMONIC`) for weird behavior, filtering out confident known classes. 
* **Disclaimer:** An anomaly in feature space is NOT equivalent to a physical discovery. In >90% of cases, anomalies are instrumental artifacts, extreme eclipsing binaries, or variable stars.

## Pipeline Modes

Onyx operates in several distinct modes via `main.py --mode <MODE>`:

1. `train` (Default): Extracts features, matches catalogs, trains models, outputs metrics and SHAP plots.
   * `python main.py --mode train --mission Kepler --limit 300`
2. `extract`: Runs standalone feature extraction without launching model training (useful for preparing dataset files or testing).
   * `python main.py --mode extract --mission Kepler --limit 100 --output outputs/kepler_features.csv`
   * `python main.py --mode extract --mission TESS --limit 50 --output outputs/tess_features.csv`
3. `predict`: Runs extraction on unlabelled target IDs and uses the saved model to classify them.
   * `python main.py --mode predict --targets-file my_targets.txt --output outputs/my_predictions.csv`
   * Default output: `outputs/new_candidates.csv`
4. `score-file`: Skips extraction and directly predicts on an existing CSV feature file.
   * `python main.py --mode score-file --input-file custom_features.csv --output outputs/scored_file.csv`
   * Default output: `outputs/scored_file.csv`
5. `retrain`: Retrains the model from scratch on combined data.
   * `python main.py --mode retrain --mission Kepler --limit 50`
6. `inject`: Runs the synthetic injection pipeline.
   * `python main.py --mode inject --injection-stars 100 --injections-per-star 12 --targets-file quiet_targets.txt`
   * Add `--quick` for rapid coarse-grid check (saves to `outputs/injection_results_quick.csv`).
7. `transfer`: Runs Kepler to TESS domain shift experiments (T1, T2, T3).
   * `python main.py --mode transfer --train-csv extracted_features.csv --test-csv outputs/tess_features.csv`
8. `demo`: Runs the balanced demonstration pipeline with equal training and control halves, two-way evaluation, Kepler-80 recovery, and comprehensive Russian report `summary.md`.
   * `python main.py --mode demo --marked-stars 120 --unmarked-stars 120`
   * For quick smoke test: `python main.py --mode demo --marked-stars 4 --unmarked-stars 4 --quick --smoke-test`

## Производительность (Performance)

* **Поэтапное профилирование (звезда KIC 10801249 / KOI-4611, статус False Positive, 18 файлов / 18 уникальных кварталов Q00–Q17, ~65 000 точек, 1 рабочий процесс):**
  * **Обычный режим (Default, без `--quick`, `frequency_factor=500`, 10 шагов длительности):** сумма этапов: **366,8 секунды (~6,1 минуты)** на звезду (полное время с накладными расходами ~385 с):
    * Загрузка FITS-файлов с MAST (`dl`): **107,0 с** (~29% общего времени звезды)
    * Склейка и выравнивание потока и центроидов (`stitch`): **1,4 с**
    * Итеративный BLS-поиск (`bls`, 3 итерации): **255,8 с** (~70% общего времени звезды)
      * Итерация 1: 91,3 с (сетка: 215 146 частот) → $P = 1,6750$ д, $\text{SDE} = 17,1$, $\text{SNR} = 18,1$
      * Итерация 2: 82,9 с (сетка: 215 146 частот) → $P = 95,5922$ д, $\text{SDE} = 15,9$, $\text{SNR} = 17,9$
      * Итерация 3: 81,5 с (сетка: 215 146 частот) → $P = 0,5216$ д, $\text{SDE} = 17,2$, $\text{SNR} = 15,1$
    * Извлечение 51 признака (`feats`): **1,0 с**
  * **Сравнение производительности и рекомендации по воркерам:**
    * **Локальный компьютер (1 процесс CPU):** около **385 секунд** на звезду на полном 4-летнем интервале (практически совпадает с замером 366,8 с).
    * **Google Colab (бесплатный тариф, Linux, 2 vCPU):** каждый из двух процессов тратил около **17,5 минут** на звезду, то есть при двух параллельных процессах в среднем одна звезда обрабатывается за **~8,75 минуты**. Причина более медленной работы отдельного процесса в Colab по сравнению с локальной машиной окончательно не установлена (возможная рабочая гипотеза: два виртуальных потока vCPU делят ресурсы одного физического ядра при непрерывных вычислениях BLS).
    * **Оптимизация параллелизма:** поскольку сетевая загрузка FITS занимает около 29% времени звезды (107 из 367 с) и является I/O-bound операцией, задание числа рабочих процессов (`--workers`) чуть больше числа физических процессорных ядер может ускорить общий прогон. Рекомендуется подобрать оптимальное число воркеров пробным запуском.
    * **Быстрый проверочный режим (`--quick`, `frequency_factor=5000`, 4 шага длительности):**
      * На **14 файлах / 14 уникальных кварталах** (замер на Kepler-80 / KIC 4852528, 1 процесс CPU, свежий кэш, `--no-resume`): сумма этапов: **106,3 секунды (~1,8 минуты)** на звезду (`dl` = 82,8 с, `stitch` = 1,1 с, `bls` = 21,6 с, `feats` = 0,8 с; полное время `total_time` с накладными расходами 107,6 с, процессорное время 113,1 с).
      * Ранее полученное значение 118,6 с относилось к быстрому режиму при искусственном ограничении 3 кварталами (разница обусловлена сетевыми задержками MAST при скачивании).
    * **Замер демонстрации на 60 звёздах (быстрый режим, многопроцессорный запуск):** 60 звёзд за **~16,5 минут** на 12 параллельных процессах (в среднем 177 с на звезду; из них скачивание FITS с MAST заняло в среднем 157 с на звезду).
* **Режим `--mode extract`:** Позволяет отделить длительное фотометрическое извлечение признаков от обучения моделей.
* **Отладочный флаг `--max-sectors`:** Предназначен исключительно для локальной отладки и тестирования кода. Он искусственно обрезает наблюдательный интервал (baseline), искажает признаки и астрофизические параметры сигналов. Файлы, полученные с данным флагом, **нельзя использовать для обучения основной модели** (при его указании скрипт выводит предупреждение и автоматически добавляет к имени файла суффикс `_sec{N}`).
* **Изолированный тестовый режим (`--smoke-test`):** Все результаты тестовых прогонов направляются в отдельную папку `outputs/_smoke_test/`, исключённую из Git, не затрагивая основные рабочие артефакты в `outputs/`.
* **Показательные примеры и демонстрации:** Демонстрационный режим добавляется; **Kepler-80 (KIC 4852528, KOI-500, подтверждённая многопланетная система)** выбрана показательной звездой. Звезда KIC 10801249 (KOI-4611) является затменной ложной тревогой (`FALSE POSITIVE`, `koi_fpflag_ec=1`) и для демонстрации экзопланетных открытий не используется.

## Output Artifacts

| File / Folder | Description |
|---|---|
| `extracted_features.csv` | Main feature matrix produced by extractor on real stars. Excluded from Git. |
| `test_features.csv` | Синтетический тестовый файл (случайные числа). Нужен исключительно для проверки работоспособности кода (`--smoke-test`) и никаких научных результатов не даёт. |
| `legacy/` | Архив устаревших файлов (включая `legacy_mixed_features.csv`). Исключён из Git. |
| `outputs/models/` | Serialized `pipeline_model.joblib`. |
| `outputs/figures/` | All plots: ROC, SHAP, Coverage, Transfer KS, Heatmap, Anomalies. Excluded from Git. |
| `outputs/predictions.csv` | Test set predictions with SHAP top factors. |
| `outputs/new_candidates.csv` | Default output from `predict` mode. |
| `outputs/scored_file.csv` | Default output from `score-file` mode. |
| `outputs/injection_results.csv` | Transit injection completeness table (or `_quick.csv`). |
| `outputs/folded_curves.npz` | Compressed arrays of phase-folded curves (201 bins) for all signals. Excluded from Git. |
| `outputs/demo/` | Артефакты демонстрационного режима (`--mode demo`). |

> [!NOTE]
> Метрики и графики из ранних коммитов истории репозитория (включая коммит `7f62b38`: Accuracy 0.678, ROC-AUC 0.750) относились к черновым запускам на синтетических/пробных данных и в актуальном проекте не используются. Папка `outputs/` очищена перед боевым прогоном.

## Демонстрационный режим (--mode demo)

Демонстрационный режим предназначен для воспроизводимой проверки пайплайна на сбалансированной выборке из реальных фотометрических данных миссии Kepler.

### 1. Архитектура и методология эксперимента
* **Строгое разделение 50/50 на уровне звёзд:** Выборка делится на обучающую и контрольную половины. Каждая половина содержит строго равное число отмеченных KOI-звёзд (с подтверждёнными экзопланетами и затменными ложными тревогами) и контрольных неотмеченных звёзд (без известных KOI).
* **Контроль звёздных параметров:** Неотмеченные звёзды подбираются из официального звёздного каталога Kepler (`Q1_Q17_DR25_KS`) с помощью 1-к-1 нормализованного евклидова сопоставления по яркости (`kepmag`) и числу кварталов (`st_quarters`). Это исключает тривиальное разделение групп по инструментальному шуму или длительности наблюдений.
* **Изоляция витринной системы Kepler-80:** Известная 5-планетная резонансная система Kepler-80 (KIC 4852528) принудительно направляется в контрольную половину и **строго исключается** из любого обучения, исключая утечку данных (data leakage).
* **Двунаправленная перекрёстная проверка (Two-way evaluation):**
  * **Направление A:** Обучение на Обучающей половине → Оценка на Контрольной половине.
  * **Направление B:** Обучение на Контрольной половине (Kepler-80 исключён) → Оценка на Обучающей половине.
* **Неприкосновенность основных артефактов (Immutability Check):** Перед началом и после завершения работы конвейер вычисляет и сравнивает криптографические контрольные суммы (SHA-256) всех файлов в рабочей папке `outputs/`, гарантируя отсутствие нежелательных модификаций.

### 2. Команда запуска и оценка времени
* **Полный запуск (боевой режим):**
  ```bash
  python main.py --mode demo --marked-stars 120 --unmarked-stars 120 --workers 4
  ```
* **Быстрый проверочный запуск (smoke test):**
  ```bash
  python main.py --mode demo --marked-stars 4 --unmarked-stars 4 --quick --smoke-test
  ```
* **Оценка времени расчёта:**
  * В обычном режиме (полная сетка BLS, `frequency_factor=500`): около **6,1 минуты (~367 секунд)** на звезду на одном процессе CPU.
  * Для полной выборки из 240 звёзд (120 отмеченных + 120 неотмеченных) суммарное процессорное время составляет ~1460 минут (~24,4 машино-часа).
  * При параллельной обработке:
    * на 4 физических ядрах: **~6,1 часа**;
    * на 8 физических ядрах: **~3,1 часа**;
    * на 16 физических ядрах: **~1,5–2 часа** (с учётом сетевой параллелизации скачивания FITS с MAST).
  * В быстром режиме (`--quick`): **~1,5–2 минуты** на звезду.

### 3. Выходные артефакты (`outputs/demo/`)
* `summary.md`: Комплексный аналитический отчёт на русском языке, включающий:
  * Статистику балансировки звёздных параметров по группам;
  * Диагностику фотометрической обработки и BLS-поиска по каждой звезде выборки;
  * Раздел А: матрица ошибок, Precision, Recall, ROC-AUC, доля воздержания REVIEW, 90% доверительные интервалы бутстрапом по звёздам и сравнение с эталонами (мажоритарный класс и порог SNR);
  * Раздел Б: распределение предсказанных вероятностей на неотмеченных звёздах, частота ложных тревог (FAR) при допущении отсутствия планет, скоринг аномалий (Isolation Forest, LOF) и Топ-5 кандидатов на валидацию с факторами SHAP;
  * Результаты по витринной системе Kepler-80 (таблица обнаружения с раздельным учётом точных совпадений и гармоник: формат «N точно + M по гармонике из K»);
  * Примеры классификации и SHAP-интерпретация решений модели;
  * Угрозы валидности и ограничения метода;
  * Отчёт о проверке неизменности основных файлов (SHA-256 immutability test).
* `diagnostics_stars.csv` (или `_quick.csv`): Полная таблица фотометрической диагностики по каждой звезде каждой группы выборки (число кварталов, базлайн, число сигналов, MATCHED/HARMONIC/UNMATCHED, статус, макс. SDE/SNR, пошаговые замеры времени).
* `control_signals.csv` и `control_signals.md`: Полная сводная таблица всех периодических сигналов контрольной половины с астрофизическими признаками, предсказанными вероятностями $p_{mean}$, $p_{std}$, вердиктами (CANDIDATE, FALSE_POSITIVE, REVIEW) и факторами SHAP.
* `showcase_kepler80.png`: Многопанельная графическая визуализация (исходная фотометрия за 14 уникальных кварталов / 14 FITS-файлов, сглаженная кривая блеска, периодограмма BLS в шкале нормализованной мощности SDE и фазовые профили обнаруженных планетных сигналов с маскированием других транзитов, биннингом и вердиктами модели).

## Limitations & Threats to Validity
* **Coarse Centroid Shift vs Pixel-Level PRF:** Onyx вычисляет смещение центроида во время транзита по столбцам координат центра светового пятна (`centroid_col`, `centroid_row`), выровненным по каждому кварталу/сектору отдельно и переведённым в угловые секунды (Kepler ~3.98", TESS ~21.0"). Это позволяет выявлять фоновые затменные двойные (BEB), однако не заменяет пиксельную диагностику PRF по файлам TPF. Для слабых звёзд и мелких транзитов фотометрический центроид подвержен шуму. В режиме синтетического вшивания (`inject`) центроидные признаки рассчитываются по реальному выровненному центроиду звезды в моменты вшитых транзитов (что сохраняет естественный инструментальный шум и исключает искусственное завышение полноты).
* **Несовместимость схем при обновлении признаков:** При изменении набора признаков (добавлении колонок центроида) старые файлы признаков и сохранённые модели становятся несовместимыми. В экстрактор встроен контроль схемы заголовка: при несоответствии колонок выдаётся предупреждение о необходимости запуска с `--no-resume` или указания нового файла `--output`.
* **Trapezoidal Injections:** The injection module ignores stellar limb darkening, leading to slight overestimation of completeness.
* **BLS Resolution vs Speed:** To avoid memory exhaustion and multi-hour processing per star on long 4-year baselines, frequency factor is bounded. Full evaluation requires adequate CPU core allocation.

## Запуск в Google Colab

Для запуска проекта в среде Google Colab (Linux, 2 vCPU, риск обрыва сессии) рекомендуется сохранять выходные данные на примонтированный Google Drive:

### 1. Подключение Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Клонирование и установка зависимостей
```bash
!git clone https://github.com/avgrunn-jix/Project-Onyx.git
%cd Project-Onyx
!pip install -r requirements.txt
```

### 3. Запуск экстрактора с сохранением на Google Drive
```bash
!python main.py --mode extract --mission Kepler --workers 2 \
  --output /content/drive/MyDrive/onyx/extracted_features.csv \
  --cache-dir /tmp/fits_cache
```

### 4. Автономный запуск обучения по готовому CSV
Если файл с признаками уже готов или сохранён на Google Drive, машинное обучение можно запустить отдельно без обращения к MAST и без повторной экстракции через единую точку входа `main.py`:
```bash
!python main.py --mode train --features-file /content/drive/MyDrive/onyx/extracted_features.csv
```
(также доступен прямой вызов: `!python 2_ml_pipeline.py --mode train --input /content/drive/MyDrive/onyx/extracted_features.csv`).

> [!IMPORTANT]
> При запуске в Google Colab папки `outputs/` и `logs/` хранятся во временной файловой системе сессии и безвозвратно удаляются при её перезапуске. Обязательно передавайте путь на Google Drive в аргумент `--output` или сохраняйте результаты (`!cp -r outputs logs /content/drive/MyDrive/onyx/`).

### 5. Поведение при обрыве сессии (Возобновление)
Сессия Colab может завершиться в любой момент. В Project Onyx встроена отказоустойчивость:
* **Автосохранение:** строки каждой обработанной звезды сбрасываются на диск (`flush` + `fsync`) сразу по завершении анализа звезды.
* **Автовозобновление:** при повторном запуске той же команды скрипт считывает список уже готовых звёзд из `--output` и обрабатывает только оставшиеся.
* **Флаги управления отказоустойчивостью:**
  * `--no-resume`: принудительно начать обработку с нуля (перезаписать выходной CSV). По умолчанию возобновление включено.
  * `--star-timeout <СЕКУНДЫ>`: предел времени на одну звезду (по умолчанию `1800` с). Если загрузка или расчет зависает, звезда отправляется в файл `_failed.csv`, а конвейер переходит к следующей звезде. При повторном запуске упавшие звёзды повторяются один раз.
  * `--cache-dir <ПУТЬ>`: путь для скачивания временных FITS-файлов (по умолчанию `/tmp` через `tempfile.gettempdir()`). FITS автоматически удаляются сразу после извлечения признаков.

## References
1. Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.
2. Valizadegan, H., et al. (2022). ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier that Validates 301 New Exoplanets. *The Astrophysical Journal*, 926(2), 120.
3. Lundberg, S. M., & Lee, Su-In (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
4. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *2008 Eighth IEEE International Conference on Data Mining*.
