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

### 3. False Positive Classification (The "Why")
When Onyx rejects a candidate, it predicts the *likely reason* using a multi-label One-Vs-Rest XGBoost classifier trained on known FP flags (`nt`, `ss`, `co`, `ec`).
* **Honest Disclaimer on Centroid Offset (`co`):** Centroid offset requires pixel-level positional data (which star moved during transit). Because Onyx relies solely on 1D light curve flux features, the prediction of `co` is inherently limited and performs close to random guessing. This limitation is intentional to highlight the boundary of pure photometric analysis.

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
2. `predict`: Runs extraction on unlabelled target IDs and uses the saved model to classify them.
   * `python main.py --mode predict --targets-file my_targets.txt --output outputs/my_predictions.csv`
   * Default output: `outputs/new_candidates.csv`
3. `score-file`: Skips extraction and directly predicts on an existing CSV feature file.
   * `python main.py --mode score-file --input-file custom_features.csv --output outputs/scored_file.csv`
   * Default output: `outputs/scored_file.csv`
4. `retrain`: Retrains the model from scratch on combined data.
   * `python main.py --mode retrain --mission Kepler --limit 50`
5. `inject`: Runs the synthetic injection pipeline.
   * `python main.py --mode inject --injection-stars 100 --injections-per-star 12 --targets-file quiet_targets.txt`
   * Add `--quick` for rapid coarse-grid check (saves to `outputs/injection_results_quick.csv`).
6. `transfer`: Runs Kepler to TESS domain shift experiments (T1, T2, T3).
   * `python main.py --mode transfer --train-csv extracted_features.csv --test-csv outputs/tess_features.csv`

## Output Artifacts

| File / Folder | Description |
|---|---|
| `extracted_features.csv` | Main feature matrix. Excluded from Git. |
| `outputs/models/` | Serialized `pipeline_model.joblib`. |
| `outputs/figures/` | All plots: ROC, SHAP, Coverage, Transfer KS, Heatmap, Anomalies. |
| `outputs/predictions.csv` | Test set predictions with SHAP top factors. |
| `outputs/new_candidates.csv` | Default output from `predict` mode. |
| `outputs/scored_file.csv` | Default output from `score-file` mode. |
| `outputs/injection_results.csv` | Transit injection completeness table (or `_quick.csv`). |
| `outputs/folded_curves.npz` | Compressed arrays of phase-folded curves (201 bins) for all signals. |
| `outputs/metrics.json` | Final performance metrics. |

## Limitations & Threats to Validity
* **No Pixel-Level Data:** Lack of centroid tracking makes background eclipsing binaries (BEB) blending harder to detect.
* **Trapezoidal Injections:** The injection module ignores stellar limb darkening, leading to slight overestimation of completeness.
* **BLS Resolution vs Speed:** To avoid memory exhaustion and multi-hour processing per star on long 4-year baselines, frequency factor is bounded. Full evaluation requires adequate CPU core allocation.

## References
1. Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.
2. Valizadegan, H., et al. (2022). ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier that Validates 301 New Exoplanets. *The Astrophysical Journal*, 926(2), 120.
3. Lundberg, S. M., & Lee, Su-In (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
4. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *2008 Eighth IEEE International Conference on Data Mining*.
