# Project Onyx: Explainable Exoplanet Detection Pipeline

Project Onyx is a modular, transparent, and explainable machine learning pipeline designed for the detection of exoplanet candidates from Kepler and TESS photometric light curves. 

Unlike black-box deep learning systems, Onyx focuses on **honest evaluation, feature explainability (SHAP), and robust anomaly detection**. It implements rigorous data leakage prevention, confidence-based abstention, and handles multi-planet systems.

## Key Features & Extensions (v2)

### 1. Robust Data Processing & Storage Efficiency
* **Local Caching & Disk Saving:** Downloading Kepler short/long cadence data generates massive amounts of FITS files. Onyx downloads each star into isolated temporary directories and automatically removes them after feature extraction.
* **Cadence-agnostic Flattening:** The detrending window length is defined in physical days (--flatten-days=3.0), rather than data points, ensuring transferability between Kepler (29.4 min) and TESS (2 min) cadence.

### 2. Multi-Signal Iterative Search
Instead of finding only the strongest transit, Onyx iteratively masks out detected transits and re-runs the Box Least Squares (BLS) periodogram up to 4 times per star. 
* Found signals are matched against the KOI/TOI catalogs. Matches are classified as MATCHED, HARMONIC (period ratio 0.5 or 2.0), or UNMATCHED. 
* **Note:** UNMATCHED signals are NOT treated as false positives. They are excluded from supervised training to prevent contamination but are evaluated in the prediction and anomaly detection phases.

### 3. False Positive Classification (The "Why")
When Onyx rejects a candidate, it predicts the *likely reason* using a multi-label One-Vs-Rest XGBoost classifier trained on known FP flags (
t, ss, co, ec).
* **Honest Disclaimer on Centroid Offset (co):** Centroid offset requires pixel-level positional data (which star moved during transit). Because Onyx relies solely on 1D light curve flux features, the prediction of co is inherently limited and performs close to random guessing. This limitation is intentional to highlight the boundary of pure photometric analysis.

### 4. Explainability (SHAP)
Onyx employs TreeExplainer (Lundberg & Lee, 2017) to provide local and global interpretability.
* For every new prediction, Onyx outputs the **Top 3 contributing factors** (e.g., secondary_depth_ppm=450 (negative) meaning a deep secondary eclipse strongly pushed the model to reject the candidate).

### 5. Confidence and Abstention (Right to Refuse)
Onyx uses an ensemble of 20 XGBoost models trained on grouped bootstraps. It calculates p_mean and p_std. 
* If a signal falls into an ambiguous zone, the pipeline flags it as REVIEW instead of forcing a binary choice. Thresholds are automatically tuned on a validation set to ensure >95% precision for confident CANDIDATE and FALSE_POSITIVE calls.

### 6. Synthetic Transit Injections (Completeness Test)
To evaluate the true sensitivity of the pipeline, Onyx includes an --inject mode. It takes "quiet" stars and mathematically injects trapezoidal transits varying in radius and period.
* **Disclaimer:** Injections are idealized (no limb darkening, circular orbits, no transit starspots). Therefore, the resulting completeness heatmap (outputs/figures/injection_heatmap.png) represents an *upper bound* of the pipeline's true sensitivity.

### 7. Mission Transferability (Kepler → TESS)
We tested the model's ability to generalize across missions:
* **T1:** Direct transfer (All features).
* **T2:** Robust transfer (Removing aseline_d, 	ransit_count, gap_fraction, which are heavily mission-biased).
* **T3:** In-domain cross-validation on TESS.
Onyx computes the Kolmogorov-Smirnov (K-S) shift for each feature (outputs/figures/ks_shift_by_feature.png) to identify covariate shift.

### 8. Unsupervised Anomaly Detection
Using Isolation Forest (Liu, Ting & Zhou, 2008) and Local Outlier Factor, Onyx scans all signals (including UNMATCHED and HARMONIC) for weird behavior, filtering out confident known classes. 
* **Disclaimer:** An anomaly in feature space is NOT equivalent to a physical discovery. In >90% of cases, anomalies are instrumental artifacts, extreme eclipsing binaries, or variable stars.

## Pipeline Modes

Onyx operates in several distinct modes via main.py --mode <MODE>:

1. 	rain (Default): Extracts features, matches catalogs, trains models, outputs metrics and SHAP plots.
   * python main.py --mode train --mission Kepler --limit 300
2. predict: Runs extraction on unlabelled target IDs and uses the saved model to classify them.
   * python main.py --mode predict --targets-file my_targets.txt
3. score-file: Skips extraction and directly predicts on an existing CSV.
   * python main.py --mode score-file --input-file custom.csv
4. etrain: Retrains the model from scratch on combined data.
5. inject: Runs the synthetic injection pipeline.
   * python main.py --mode inject --injection-stars 50

## Output Artifacts

| File / Folder | Description |
|---|---|
| extracted_features.csv | Main feature matrix. Excluded from Git. |
| outputs/models/ | Serialized pipeline_model.joblib. |
| outputs/figures/ | All plots: ROC, SHAP, Coverage, Transfer KS, Heatmap, Anomalies. |
| outputs/predictions.csv | Test set predictions with SHAP top factors. |
| outputs/new_candidates.csv | Results from predict mode. |
| outputs/folded_curves.npz | Compressed arrays of phase-folded curves (201 bins) for all signals. |
| outputs/metrics.json | Final performance metrics. |

## Limitations & Threats to Validity
* **No Pixel-Level Data:** Lack of centroid tracking makes background eclipsing binaries (BEB) blending harder to detect.
* **Trapezoidal Injections:** The injection module ignores stellar limb darkening, leading to slight overestimation of completeness.
* **Hardware Constraints:** The requency_factor in BLS is capped to avoid OutOfMemory errors, which might slightly reduce period precision for very long Kepler baselines.

## References
1. Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.
2. Valizadegan, H., et al. (2022). ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier that Validates 301 New Exoplanets. *The Astrophysical Journal*, 926(2), 120.
3. Lundberg, S. M., & Lee, Su-In (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
4. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *2008 Eighth IEEE International Conference on Data Mining*.
