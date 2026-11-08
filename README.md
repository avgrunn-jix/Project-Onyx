# Project Onyx: Explainable Exoplanet Detection Pipeline

Project Onyx is a modular, transparent, and explainable machine learning pipeline for finding exoplanet candidates in Kepler and TESS photometric light curves. It searches each star for several periodic transit-like signals, matches them against the KOI/TOI catalogs, and classifies them with an ensemble of XGBoost models.

Unlike black-box deep learning systems, Onyx focuses on honest evaluation, feature explainability (SHAP), confidence-based abstention, and anomaly detection. It applies star-level data splits to prevent leakage and handles multi-planet systems.

## Contents

- [Key features](#key-features)
- [Installation and quick start](#installation-and-quick-start)
- [Running tests](#running-tests)
- [Pipeline modes](#pipeline-modes)
- [Demo mode](#demo-mode)
- [Search settings](#search-settings)
- [Resumable runs and fault tolerance](#resumable-runs-and-fault-tolerance)
- [Output artifacts](#output-artifacts)
- [Performance](#performance)
- [Limitations and threats to validity](#limitations-and-threats-to-validity)
- [References](#references)

## Key features

- **Cadence-agnostic detrending.** The flattening window is set in physical days (`--flatten-days`, default 3.0), not in data points, so the same setting works for Kepler (29.4 min) and TESS (2 min) cadence.
- **Low disk usage.** Each star is downloaded into its own temporary directory, and the FITS files are deleted after feature extraction (unless `--keep-raw` is given).
- **Robust target lists.** Target list files in UTF-8, UTF-8 with BOM, and UTF-16 (LE/BE) are read automatically.
- **Multi-signal iterative search.** The Box Least Squares (BLS) periodogram is run repeatedly on each star, masking the transits found so far, for up to `--max-signals` signals per star (default 4). The search stops early when the strongest peak falls below SDE 7.0 or depth SNR 7.1.
- **Catalog matching.** Each signal is labelled `MATCHED` (period within 1% of a catalog period), `HARMONIC` (period ratio 0.5 or 2.0 to a catalog period), or `UNMATCHED`.
  > [!NOTE]
  > `UNMATCHED` does not mean false positive. `UNMATCHED` signals are excluded from supervised training, but they are scored in the prediction and anomaly-detection phases.
- **Centroid-shift diagnostics.** In-transit versus out-of-transit centroid shifts are computed from `centroid_col` and `centroid_row`, detrended per observation segment. The features are `centroid_shift_col_sigma`, `centroid_shift_row_sigma`, `centroid_shift_total_sigma`, and `centroid_shift_total_arcsec` (pixel scales of 3.98"/pixel for Kepler and 21.0"/pixel for TESS).
- **Likely false-positive reason.** One binary XGBoost classifier is trained per KOI false-positive flag (`koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`) when enough positive and negative examples exist. Predictions include `p_reason_*` and `likely_reason` columns.
- **Explainability (SHAP).** A `TreeExplainer` (Lundberg & Lee, 2017) reports the top 3 contributing features for every prediction, for example `secondary_depth_ppm=450 (negative)`, meaning a deep secondary eclipse pushed the model towards rejecting the candidate.
- **Confidence and abstention.** The `train` mode builds an ensemble of 20 XGBoost models trained on star-grouped bootstraps and reports `p_mean` and `p_std`. A signal with ambiguous or uncertain scores gets the verdict `REVIEW` instead of a forced binary decision. The thresholds are searched on a grid, and the first setting is used for which the confident verdicts reach at least 95% precision (the share of correct `CANDIDATE` verdicts, and the share of correct `FALSE_POSITIVE` verdicts) on the training split; if no setting reaches it, built-in defaults are used. Demo mode tunes its thresholds differently (see [Demo mode](#demo-mode)).
- **Synthetic transit injections.** The `inject` mode injects trapezoidal transits into quiet stars and measures how often the search recovers them (see [Pipeline modes](#pipeline-modes)).
- **Mission transferability (Kepler → TESS).** The `transfer` mode compares direct transfer, a transfer with mission-dependent features removed, and in-domain cross-validation on TESS. It also computes the Kolmogorov-Smirnov (K-S) shift per feature.
- **Unsupervised anomaly detection.** Isolation Forest (Liu, Ting & Zhou, 2008) and Local Outlier Factor scan all signals, including `UNMATCHED` and `HARMONIC` ones, and report signals that both methods rank among their top 50 and that did not receive a confident verdict.
  > [!IMPORTANT]
  > An anomaly in feature space is not a physical discovery. Anomalies may well be instrumental artifacts, extreme eclipsing binaries, or variable stars.

## Installation and quick start

Install the dependencies listed in `requirements.txt` (numpy, pandas, scipy, astropy, lightkurve, astroquery, scikit-learn, xgboost, shap, matplotlib, seaborn, joblib):

```bash
pip install -r requirements.txt
```

Quick end-to-end check of the demo pipeline on a handful of stars. It uses the coarse BLS grid and writes everything to `outputs/_smoke_test/`, leaving the main artifacts in `outputs/` untouched. It downloads data from MAST, so network access is required:

```bash
python main.py --mode demo --marked-stars 4 --unmarked-stars 4 --quick --smoke-test
```

> [!NOTE]
> A smoke test only checks that the code runs end to end. Its results are not scientifically meaningful.

Run `python main.py --help` for the full list of arguments.

## Running tests

The tests are in `tests/`, need no network access, and are run from the repository root with `pytest`. pytest is not listed in `requirements.txt`, so install it separately:

```bash
pip install pytest
pytest
```

The test files cover: outlier clipping and the extractor search flags (`test_clip_outliers.py`), period alias resolution (`test_alias_resolution.py`), feature column selection (`test_feature_columns.py`), the star-level train/control split without leakage (`test_star_split_no_leakage.py`), and prefix stability of the star selection and split when the sample size grows (`test_prefix_stability.py`).

## Pipeline modes

All modes are selected with `python main.py --mode <MODE>`. The default mode is `train`. Paths below are relative to the repository root; with `--smoke-test`, outputs go to `outputs/_smoke_test/` instead.

| Mode | What it does | Example command | Default output |
|---|---|---|---|
| `train` (default) | Extracts features, matches catalogs, trains the models, and writes metrics and SHAP plots. With `--features-file` it skips extraction and trains on that CSV. | `python main.py --mode train --mission Kepler --limit 300` | `extracted_features.csv`, `outputs/models/pipeline_model.joblib`, `outputs/predictions.csv`, `outputs/metrics.json` |
| `extract` | Runs feature extraction only, without training. Useful for preparing feature files. | `python main.py --mode extract --mission Kepler --limit 100 --output outputs/kepler_features.csv` | `extracted_features.csv` (Kepler) or `outputs/tess_features.csv` (TESS) |
| `predict` | Extracts features for the stars in `--targets-file` and classifies them with the saved model. Requires a trained model. | `python main.py --mode predict --targets-file my_targets.txt --output outputs/my_predictions.csv` | `outputs/new_candidates.csv` (features in `outputs/predict_features.csv`) |
| `score-file` | Skips extraction and classifies an existing feature CSV with the saved model. | `python main.py --mode score-file --input-file custom_features.csv --output outputs/scored_file.csv` | `outputs/scored_file.csv` |
| `retrain` | Follows the same code path as `train`: extracts features unless `--features-file` is given, then trains the models again. | `python main.py --mode retrain --mission Kepler --limit 50` | Same as `train` |
| `inject` | Runs the synthetic-injection completeness test. Add `--quick` for a coarse-grid check. | `python main.py --mode inject --injection-stars 100 --injections-per-star 12 --targets-file quiet_targets.txt` | `outputs/injection_results.csv` (`_quick` suffix with `--quick`) |
| `transfer` | Runs the Kepler → TESS experiments T1 (all features), T2 (mission-dependent features removed), and T3 (cross-validation on TESS). | `python main.py --mode transfer --train-csv extracted_features.csv --test-csv outputs/tess_features.csv` | `outputs/metrics_transfer.json`, `outputs/figures/ks_shift_by_feature.png`, `outputs/figures/transfer_roc.png` |
| `demo` | Runs the balanced demonstration pipeline (see [Demo mode](#demo-mode)). | `python main.py --mode demo --marked-stars 120 --unmarked-stars 120` | `outputs/demo/` |

Notes on `inject`:

- Injected signals have random periods of 1-30 d, radii of 0.8-15 Earth radii, and durations of 0.05-0.3 d.
- Recovery uses the same BLS settings as the real search, with period range $P \in [0.5, \min(100.0, \text{baseline}/2)]$ days.
  - **Normal grid (default):** 10 duration steps linearly spaced in $[0.04, 0.45]$ days (about 0.96 h to 10.8 h); frequency resolution $\Delta f = 500 \times \min(T) / \text{baseline}^2$ (`frequency_factor=500`).
  - **Quick grid (`--quick`):** 4 duration steps in $[0.05, 0.40]$ days; frequency resolution $\Delta f = 5000 \times \min(T) / \text{baseline}^2$ (`frequency_factor=5000`). Outputs get a `_quick` suffix and a warning is printed.
- Injections are idealized: no limb darkening, circular orbits, and no starspots crossing the transit. The resulting completeness map is therefore an upper bound on the true sensitivity of the pipeline.

## Demo mode

Demo mode (`--mode demo`) checks the pipeline reproducibly on a balanced sample of real Kepler stars.

### Methodology

- **Star-level 50/50 split.** The sample is split into a training half and a control half. Each half contains the same number of marked stars (stars that appear in the KOI label table with a `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE` disposition) and unmarked control stars (no known KOI). No star appears in both halves.
- **Matched control stars.** Unmarked stars are drawn from the Kepler stellar catalog (`Q1_Q17_DR25_KS`, `kepmag` 11.0-15.5, data span above 1000 d, at least 10 quarters), excluding all KOI hosts. Each marked star gets a one-to-one match within its KIC-number decile, chosen as the nearest star in normalized `kepmag` and number of quarters (`st_quarters`). This prevents trivially separating the groups by instrumental noise or observation length.
- **Isolated showcase system.** Kepler-80 (KIC 4852528) is forced into the control half and is excluded from all training, which rules out data leakage. Use `--showcase-star` to choose another star. The KOI label table lists five planets for this star (b, c, d, e, f), and the Kepler-80 recovery table in `summary.md` counts exactly these five, using their periods hard-coded in `3_demo_pipeline.py`. A sixth planet, Kepler-80 g (period about 14.65 d), was found by Shallue & Vanderburg (2018; reference 1) but is absent from the label table. It is therefore not counted as a known planet in the recovery table, and a signal near its period would not be matched to a catalog planet.
- **Two-way evaluation.**
  - **Direction A:** train on the training half, evaluate on the control half.
  - **Direction B:** train on the control half (Kepler-80 removed), evaluate on the training half.
- **Abstention thresholds.** Verdict thresholds are tuned with star-grouped (`GroupKFold`) out-of-fold predictions on the training data, choosing the thresholds with the largest coverage for which the accuracy on accepted signals (the share of correct verdicts among all `CANDIDATE` and `FALSE_POSITIVE` verdicts) reaches `--target-precision` (default 0.90). If no setting reaches it, the setting with the highest accuracy is used. Despite its name, `--target-precision` therefore targets accuracy on accepted signals, not class precision. Confidence intervals come from a bootstrap over stars (500 resamples, 5th-95th percentile).
- **Immutability check.** Before and after the run, the pipeline computes SHA-256 hashes of all files in `outputs/` (excluding `outputs/demo*` and `outputs/_smoke_test`) and compares them, so the run cannot silently modify existing artifacts.

### Running

Full run:

```bash
python main.py --mode demo --marked-stars 120 --unmarked-stars 120 --workers 4
```

Quick smoke test (coarse grid, isolated output folder):

```bash
python main.py --mode demo --marked-stars 4 --unmarked-stars 4 --quick --smoke-test
```

Other demo options: `--seed`, `--showcase-star`, `--unmarked-file`, `--output-dir`, `--max-download-workers`, `--target-precision`, `--min-depth-ppm`. Runtime estimates are given in [Performance](#performance).

### Showcase figure

`make_showcase.py` redraws the Kepler-80 (KIC 4852528) showcase figure for a finished run, with a choice of layout (`tall` or `wide`, the wide one is about 11 x 6.6 inches at 300 dpi and suits a journal page) and of label language (`ru` or `en`). It takes the detected signals from the features table of the run and, optionally, the verdicts and `p_mean` values from `control_signals_quick.csv`, matched by period within 1%; when no match is found, the panel title shows `p = n/a`. It downloads the Kepler light curve with the same settings as the quick extractor mode, so network access and the light curve cache are needed.

```bash
python make_showcase.py --features-file outputs/demo/features_demo_quick.csv --signals-file outputs/demo/control_signals_quick.csv --layout wide --lang en --output outputs/showcase_kepler80_wide_en.png
```

### Output files (`outputs/demo/`)

With `--quick`, file names get a `_quick` suffix. With `--smoke-test`, the folder is `outputs/_smoke_test/demo/`.

- `summary.md`: the full analytical report. **This file is generated in Russian.** It contains:
  - balance statistics of the stellar parameters per group;
  - photometry and BLS diagnostics for every star;
  - the classification results with a confusion matrix, precision, recall, ROC-AUC, the share of `REVIEW` verdicts, bootstrap confidence intervals, and comparison with baselines (majority class and an SNR threshold);
  - the distribution of predicted probabilities on unmarked stars, the false-alarm rate assuming they host no detectable planets, anomaly scores (Isolation Forest, LOF), and the top 5 validation candidates with SHAP factors;
  - the Kepler-80 recovery table, counting exact period matches and harmonic matches separately (format "N exact + M by harmonic out of K");
  - classification examples with SHAP explanations;
  - threats to validity and the SHA-256 immutability report.
- `diagnostics_stars.csv`: per-star photometric diagnostics (number of quarters, baseline, number of signals, `MATCHED`/`HARMONIC`/`UNMATCHED` counts, status, maximum SDE/SNR, and per-stage timings).
- `control_signals.csv` and `control_signals.md`: all periodic signals of the control half with their features, `p_mean`, `p_std`, verdicts (`CANDIDATE`, `FALSE_POSITIVE`, `REVIEW`), and SHAP factors.
- `showcase_kepler80.png`: a multi-panel figure with the raw photometry of all downloaded quarters, the detrended light curve, the BLS periodogram (SDE scale), and phase-folded profiles of the detected signals with the other transits masked.

## Search settings

Two flags of `1_data_extractor.py` control details of the signal search. `main.py` does not forward them, so pass them to the extractor directly:

```bash
python 1_data_extractor.py --mission Kepler --limit 100 --clip-upward-only --no-resolve-aliases
```

- **Outlier clipping (default: symmetric).** By default, outliers beyond 5 sigma are removed on both sides. `--clip-upward-only` (experimental) removes only upward outliers, keeping transit dips. On a small bench, upward-only clipping lost known planets in multi-planet systems, so it is off by default.
- **Alias resolution (default: on).** After each BLS peak, the extractor tests whether the true period is one half or one third of it (or twice it). A test is accepted only if the BLS power near the trial period and the box-model transit depth support it (within a 1% period tolerance). A signal whose period duplicates one already found (ratios 1, 2, 3, 1/2, or 1/3 within 1%) is not recorded, but its transits are masked and the iteration is used up. `--no-resolve-aliases` disables all of this.
- The number of corrected periods is written to the diagnostics file (`<output>_diagnostics.csv`) as `alias_corrections`, and the applied factors as `alias_factors`. Neither column enters the feature table.
- Bench: `bench/run_bench.bat` runs the extractor on 7 Kepler stars (`bench/targets_bench.csv`) with default settings and with `--no-resolve-aliases`, and `bench/compare_bench.py` compares the two. Recorded results are in [bench/BENCH_RESULTS.md](bench/BENCH_RESULTS.md).

## Resumable runs and fault tolerance

Long extraction runs can be interrupted and resumed. This applies to any environment.

- **Autosave.** The rows of each star are flushed to disk (`flush` + `fsync`) as soon as that star is finished.
- **Auto-resume.** Re-running the same command reads the finished stars from `--output` (or from the diagnostics file) and processes only the remaining ones. If the existing file lacks columns of the current feature schema (the centroid-shift columns), the extractor stops with an error: use `--no-resume` or a new `--output`.
- `--no-resume`: start from scratch. This deletes the existing output, `_failed`, diagnostics, and `folded_curves.npz` files. Resuming is on by default.
- `--star-timeout <SECONDS>`: time limit per star (default 1800 s). A star whose download or computation fails or hangs is recorded in `<output>_failed.csv` (for example `extracted_features_failed.csv`), and the pipeline moves on. On the next run, failed stars are retried once.
- `--cache-dir <PATH>`: where FITS files are downloaded (default: the system temporary folder, `tempfile.gettempdir()`). Files are deleted after feature extraction unless `--keep-raw` is set.
- **Training without MAST access.** If a feature CSV already exists, train directly on it without any extraction:

  ```bash
  python main.py --mode train --features-file extracted_features.csv
  ```

  The same can be done with `python 2_ml_pipeline.py --mode train --input extracted_features.csv`.

## Output artifacts

| File / folder | Description |
|---|---|
| `extracted_features.csv` | Main feature matrix produced by the extractor on real stars (`_quick` suffix with `--quick`). Excluded from Git. |
| `<output>_diagnostics.csv`, `<output>_failed.csv` | Per-star diagnostics and failed stars of an extraction run. |
| `outputs/models/` | The serialized model `pipeline_model.joblib`. Excluded from Git. |
| `outputs/figures/` | Plots: SHAP, coverage versus precision, transfer K-S shift and ROC, anomalies. Excluded from Git. |
| `outputs/predictions.csv` | Test-set predictions with SHAP top factors. |
| `outputs/metrics.json`, `outputs/metrics_fp_reasons.json`, `outputs/metrics_transfer.json` | Metrics of training, false-positive-reason classifiers, and transfer experiments. |
| `outputs/anomalies.csv` | Signals flagged by both anomaly detectors. |
| `outputs/new_candidates.csv` | Default output of `predict` mode. |
| `outputs/scored_file.csv` | Default output of `score-file` mode. |
| `outputs/injection_results.csv` | Injection completeness table (`_quick` suffix with `--quick`). |
| `outputs/folded_curves.npz` | Phase-folded curves (201 bins) of all signals. Excluded from Git. |
| `outputs/demo/` | Artifacts of demo mode (`--mode demo`); see [Demo mode](#demo-mode). `summary.md` there is in Russian. |
| `outputs/_smoke_test/` | Isolated output of `--smoke-test` runs. Excluded from Git. |

## Performance

Timings depend strongly on the star, the number of quarters, and the machine. The figures below come from specific measurements.

**Per-star profile, normal mode.** Star KIC 10801249 (KOI-4611, `FALSE POSITIVE`), 18 files covering 18 unique quarters (Q00-Q17), about 65,000 points, 1 worker process on a local machine, normal mode (no `--quick`, `frequency_factor=500`, 10 duration steps). The stages add up to **about 365 s (about 6.1 min)** per star; the total with overhead was about 385 s:

| Stage | Time | Share |
|---|---|---|
| FITS download from MAST (`dl`) | 107.0 s | about 29% |
| Stitching and flattening of flux and centroids (`stitch`) | 1.4 s | under 1% |
| Iterative BLS search (`bls`, 3 iterations) | 255.8 s | about 70% |
| Feature extraction (`feats`) | 1.0 s | under 1% |

BLS iterations of this run (each on a grid of 215,146 frequencies):

| Iteration | Time | Period | SDE | SNR |
|---|---|---|---|---|
| 1 | 91.3 s | $P = 1.6750$ d | 17.1 | 18.1 |
| 2 | 82.9 s | $P = 95.5922$ d | 15.9 | 17.9 |
| 3 | 81.5 s | $P = 0.5216$ d | 17.2 | 15.1 |

**Per-star profile, quick mode.** `--quick` uses `frequency_factor=5000` and 4 duration steps. Measured on Kepler-80 (KIC 4852528), 14 files covering 14 unique quarters, 1 CPU process, fresh cache, `--no-resume`: the stages add up to **106.3 s (about 1.8 min)** per star (`dl` = 82.8 s, `stitch` = 1.1 s, `bls` = 21.6 s, `feats` = 0.8 s). The total `total_time` with overhead was 107.6 s and the CPU time 113.1 s.

**Multi-process run.** A quick-mode demo run on 60 stars took about 16.5 min with 12 parallel worker processes (177 s per star on average, of which about 157 s was the FITS download from MAST).

**Choosing `--workers`.** The BLS search is CPU-bound, so by default the number of workers should be close to the number of physical cores. Because about 29% of the time per star in normal mode is spent on the network download (I/O-bound), a worker count slightly above the number of physical cores can shorten the total run. The best value is found by a short trial run. `--max-download-workers` (default 4) limits concurrent downloads.

**Demo runtime estimate.** These figures are extrapolated from the per-star profiles above, not measured end to end.

- Normal mode: about 6.1 min per star on one CPU process. The full sample of 240 stars (120 marked + 120 unmarked) needs about 1,460 min (about 24.4 CPU-hours).
- With parallel processing: about 6.1 h on 4 physical cores, about 3.1 h on 8, and about 1.5-2 h on 16 (including parallel FITS downloads).
- Quick mode (`--quick`): about 1.5-2 min per star.

**Related options.**

- `--mode extract` separates the long photometric extraction from model training.
- `--max-sectors` is a debugging flag only. It truncates the observation baseline, which distorts the features and the astrophysical parameters of the signals. Files produced with it must not be used to train the main model. The script prints a warning and adds a `_sec{N}` suffix to the output file name.
- `--smoke-test` redirects all outputs of a run to `outputs/_smoke_test/`, so the main artifacts in `outputs/` are not touched.

## Limitations and threats to validity

- **Coarse centroid shift versus pixel-level PRF.** The centroid shift is computed from the flux-weighted centroid columns (`centroid_col`, `centroid_row`), detrended per quarter or sector and converted to arcseconds. This helps to identify background eclipsing binaries but does not replace pixel-level PRF diagnostics on target pixel files. For faint stars and shallow transits the photometric centroid is noisy. In `inject` mode, the centroid features are computed from the star's real detrended centroid at the injected transit times, which keeps the natural instrumental noise and avoids overestimating completeness.
- **Feature schema changes.** When the feature set changes (for example, new centroid columns), old feature files and saved models become incompatible. The extractor checks the header of an existing output file and stops with an error that suggests `--no-resume` or a new `--output`. `predict` mode likewise reports missing feature columns.
- **Trapezoidal injections.** The injection module ignores limb darkening, which slightly overestimates completeness.
- **BLS resolution versus speed.** To avoid memory exhaustion and multi-hour processing of single stars on long baselines, the BLS frequency factor is bounded (500 in normal mode, 5000 in quick mode). Full evaluation needs adequate CPU resources.
- **Quick mode is coarse.** The `--quick` grid is coarser than the normal one and is meant for fast checks only.
- **Debug and test flags.** Output from `--max-sectors` runs and from `--smoke-test` runs must not be used as scientific results or for training the main model.
- **Anomalies are not discoveries.** See the anomaly-detection note under [Key features](#key-features).

## References

1. Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.
2. Valizadegan, H., et al. (2022). ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier that Validates 301 New Exoplanets. *The Astrophysical Journal*, 926(2), 120.
3. Lundberg, S. M., & Lee, Su-In (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
4. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *2008 Eighth IEEE International Conference on Data Mining*.
