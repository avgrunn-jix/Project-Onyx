# Project Onyx Extension Progress Log

## Audit (Step 0)
- 1_data_extractor.py: Downloaded light curves via lightkurve. Flattens them, applies BLS and Lomb-Scargle, computes ~30 features, and writes to extracted_features.csv.
- 2_ml_pipeline.py: Pipeline with SimpleImputer, FunctionTransformer(log1p), RobustScaler. Does GMM, KMeans, PCA, UMAP, and binary classification.
- Split strategy: Random by 	arget_id (using StratifiedGroupKFold). Labels strictly used for target and not as features.

## Completed Extensions
- **Step A:** Temporarily download FITS to 	mp_* dirs and delete in inally. Handled disk limits.
- **Step B:** Switched to --flatten-days in days rather than points to support TESS cadence.
- **Step C:** Multi-signal iterative masking and BLS up to --max-signals. Added KOI/TOI matching (MATCHED/HARMONIC/UNMATCHED). Saved folded curves to olded_curves.npz.
- **Step D:** Added Multi-label XGBoost to predict FP flags (
t, ss, co, ec).
- **Step E:** Implemented SHAP TreeExplainer for global feature importance and top 3 factors for each prediction.
- **Step F:** Deployed ensemble of 20 XGBoost models for uncertainty prediction (p_mean, p_std), and implemented a confidence threshold for REVIEW abstention.
- **Step G:** Implemented main.py --mode parameters (	rain, predict, score-file, etrain).
- **Step H:** Wrote transit injection module (inject_transit) for synthetic completeness heatmaps.
- **Step I:** Enabled TESS compatibility and wrote 	ransfer_mode to calculate Kolmogorov-Smirnov shifts between missions and test robustness.
- **Step J:** Implemented unsupervised Anomaly Detection using IsolationForest and LocalOutlierFactor on all signals, rendering a grid of folded curves for the top 20 anomalies.
- **Step K:** README overhauled to reflect new methodology, constraints, and peer-reviewed citations. No docstrings or comments in Python files.
