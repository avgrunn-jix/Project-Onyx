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

## Производительность (Performance)

* **Реальные измерения времени обработки:**
  * **Локальный ПК / ноутбук (1 ядро CPU, Kepler, стандартная сетка):** около 385 секунд (~6,4 минуты) на одну звезду при обработке полной 4-летней кривой (17 кварталов, ~65 000 точек).
  * **Google Colab (бесплатный тариф, Linux, 2 vCPU):** около 17,5 минут на одну звезду. Замедление обусловлено сетевой задержкой и троттлингом пакетных запросов к MAST API, а также однопоточными вычислениями BLS на виртуализованных ядрах.
  * **Быстрый режим (`--quick`):** около 1,5–2,5 минут на звезду за счёт огрубления шага BLS (`frequency_factor=5000`, 4 сетки длительностей).
* **Оговорка:** Данные замеры являются ориентировочными и не представляют собой строгих гарантий: фактическое время зависит от пропускной способности сети до MAST, загруженности серверов архива и производительности процессора.
* **Режим `--mode extract`:** Позволяет отделить этап длительного извлечения признаков от машинного обучения. Например, можно сначала извлечь признаки в фоновом режиме в файл `outputs/tess_features.csv`, а затем отдельно обучать или тестировать модели.

## Output Artifacts

| File / Folder | Description |
|---|---|
| `extracted_features.csv` | Main feature matrix. Excluded from Git. |
| `outputs/models/` | Serialized `pipeline_model.joblib`. |
| `outputs/figures/` | All plots: ROC, SHAP, Coverage, Transfer KS, Heatmap, Anomalies. Excluded from Git. |
| `outputs/predictions.csv` | Test set predictions with SHAP top factors. |
| `outputs/new_candidates.csv` | Default output from `predict` mode. |
| `outputs/scored_file.csv` | Default output from `score-file` mode. |
| `outputs/injection_results.csv` | Transit injection completeness table (or `_quick.csv`). |
| `outputs/folded_curves.npz` | Compressed arrays of phase-folded curves (201 bins) for all signals. Excluded from Git. |
| `outputs/metrics.json` | Final performance metrics. |

## Limitations & Threats to Validity
* **Coarse Centroid Shift vs Pixel-Level PRF:** Onyx вычисляет смещение центроида во время транзита по столбцам координат центра пятна (`centroid_col`, `centroid_row`), выровненным по каждому кварталу/сектору отдельно и переведённым в угловые секунды (Kepler ~3.98", TESS ~21.0"). Это позволяет отсекать грубые фоновые затменные двойные (BEB), однако не заменяет полноценную пиксельную диагностику PRF (Pixel Response Function) по файлам TPF. Для слабых звёзд и мелких транзитов фотометрический центроид является зашумлённым.
* **Несовместимость схем при обновлении признаков:** При изменении набора признаков (например, добавлении колонок центроида) старые CSV-файлы (`extracted_features.csv`) и сохранённые модели (`pipeline_model.joblib`) становятся несовместимыми. В экстрактор встроен контроль схемы заголовка: при несоответствии колонок выдаётся предупреждение о необходимости запуска с `--no-resume` или указания нового файла `--output`.
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
Если файл с признаками уже готов или сохранён на Google Drive, машинное обучение можно запустить отдельно без обращения к MAST и без повторной экстракции:
```bash
!python 2_ml_pipeline.py --mode train --input /content/drive/MyDrive/onyx/extracted_features.csv
```

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
