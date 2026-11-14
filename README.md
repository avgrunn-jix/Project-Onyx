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
- **Multi-signal iterative search.** The Box Least Squares (BLS) periodogram is run repeatedly on each star, masking the transits found so far, for up to `--max-signals` signals per star (default 4). Each peak is refined on a fine local period grid, its depth SNR is read at the power peak, and its SDE is measured on the spectrum after a running median is subtracted (red-noise correction). The search stops early when the strongest peak falls below SDE 7.0 or depth SNR 7.1; a peak with fewer than 3 observed transits is masked and not recorded. See [Search grid and detection](#search-grid-and-detection).
- **Catalog matching.** Each signal is labelled `MATCHED` (period within 1% of a catalog period), `HARMONIC` (period ratio 0.5, 2, 1/3 or 3 to a catalog period), or `UNMATCHED`.
  > [!NOTE]
  > `UNMATCHED` does not mean false positive. `UNMATCHED` signals are excluded from supervised training, but they are scored in the prediction and anomaly-detection phases.
- **Eclipsing-binary features.** `odd_even_z` compares the depths of odd and even transits in full symmetric windows, in units of their combined standard error. `secondary_z` and `secondary_phase` give the strongest dip of the same duration at phases 0.2-0.8 and its significance. `implied_radius_rjup` converts the depth into a companion radius using `star_radius`, and `duration_ratio` compares the found duration with the duration expected for a central transit on a circular orbit around a star with the catalog `star_radius` and `star_logg` (NaN without stellar parameters). The older `odd_even_sigma` and `secondary_snr` are still written but are not model features. `eb_period_match` marks a period that matches the same star in the Kepler eclipsing-binary catalog (`data/labels/kepler_eb_catalog.csv`, ratios 0.5, 1 or 2 within 1%); it is used only in reports.
- **Search diagnostics as features.** `n_transits_observed` (transits whose window holds at least half of the expected points), `single_event_fraction` (share of the summed signal that comes from the strongest transit), and `dur_at_grid_edge` (1 when the best duration is the shortest or longest allowed one).
- **Inverted-light-curve control (`--inverted-control`).** After the main search, the same search runs again in the same process on `2 - flux` of the already processed light curve, with the signals found by the main search masked first. Real transits become bumps there and are not searched for, so every dip found is a noise or systematics detection. These signals are written to `<output>_inverted.csv` with `match_status = INVERTED`; they never enter the regular feature table, tables or metrics.
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
- The recovery test in `inject` mode still uses its own single BLS call with the previous grid, period range $P \in [0.5, \min(100.0, \text{baseline}/2)]$ days, and no period refinement, so its completeness map does not yet describe the current search (see [Search grid and detection](#search-grid-and-detection)).
  - **Normal grid (default):** 10 duration steps linearly spaced in $[0.04, 0.45]$ days; frequency resolution $\Delta f = 500 \times \min(T) / \text{baseline}^2$ (`frequency_factor=500`).
  - **Quick grid (`--quick`):** 4 duration steps in $[0.05, 0.40]$ days; frequency resolution $\Delta f = 5000 \times \min(T) / \text{baseline}^2$ (`frequency_factor=5000`). Outputs get a `_quick` suffix and a warning is printed.
- Injections are idealized: no limb darkening, circular orbits, and no starspots crossing the transit. The resulting completeness map is therefore an upper bound on the true sensitivity of the pipeline.

## Demo mode

Demo mode (`--mode demo`) checks the pipeline reproducibly on a balanced sample of real Kepler stars.

### Methodology

- **Star-level 50/50 split.** The sample is split into a training half and a control half. The planned groups are marked stars (stars that appear in the KOI label table with a `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE` disposition) and unmarked control stars (no known KOI), each split into a training and a control part. No stars are shared between the halves. Stars whose extraction failed are not replaced, and the remaining groups are not trimmed to the size of the smallest one; add `--balance-halves` to restore the old behaviour of trimming all four groups to equal size. The processing order of the planned stars is shuffled with the seed, which does not change which stars are selected or which group they belong to.
- **Matched control stars.** Unmarked stars are drawn from the Kepler stellar catalog (`Q1_Q17_DR25_KS`, `kepmag` 11.0-15.5, data span above 1000 d, at least 10 quarters), excluding all KOI hosts. Each marked star gets a one-to-one match within its KIC-number decile, chosen as the nearest star in normalized `kepmag` and number of quarters (`st_quarters`). This prevents trivially separating the groups by instrumental noise or observation length.
- **Isolated showcase system.** Kepler-80 (KIC 4852528) is forced into the control half and is excluded from all training; no stars are shared between the halves. Use `--showcase-star` to choose another star. The KOI label table lists five planets for this star (b, c, d, e, f), and the Kepler-80 recovery table in `summary.md` counts exactly these five, using their periods hard-coded in `3_demo_pipeline.py`. A sixth planet, Kepler-80 g (period about 14.65 d), was found by Shallue & Vanderburg (2018; reference 1) but is absent from the label table. It is therefore not counted as a known planet in the recovery table, and a signal near its period would not be matched to a catalog planet.
- **Two-way evaluation.**
  - **Direction A:** train on the training half, evaluate on the control half.
  - **Direction B:** train on the control half (Kepler-80 removed), evaluate on the training half.
- **Abstention thresholds.** Verdict thresholds are tuned with star-grouped (`GroupKFold`) out-of-fold predictions on the training data, choosing the thresholds with the largest coverage for which the accuracy on accepted signals (the share of correct verdicts among all `CANDIDATE` and `FALSE_POSITIVE` verdicts) reaches `--target-precision` (default 0.90). If no setting reaches it, the setting with the highest accuracy is used. Despite its name, `--target-precision` therefore targets accuracy on accepted signals, not class precision. The out-of-fold models used to tune the thresholds have the same hyperparameters and the same number of models as the final ensemble. All confidence intervals are 95% intervals (2.5th-97.5th percentile) from a bootstrap over stars (500 resamples); proportions of recovered KOIs use 95% Wilson intervals.
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

Other demo options: `--seed`, `--showcase-star`, `--unmarked-file`, `--output-dir`, `--max-download-workers`, `--target-precision`, `--min-depth-ppm`.

- `--threshold-criterion {point,wilson}` (default `point`; `main.py` forwards it to the demo): how the abstention thresholds must reach `--target-precision` on the star-grouped out-of-fold predictions of the training half. The thresholds are searched on a grid (upper threshold 0.50 to 0.90, lower threshold 0.50 to 0.10, both in 9 steps, maximum ensemble spread 0.15 to 1.0); among the triples that pass, the one with the highest coverage is taken. `point` requires the point estimate of the precision on accepted verdicts to be at least the target; `wilson` requires the lower bound of its 95% Wilson interval to be at least the target. If no triple passes, the triple with the highest point precision is taken and the summary says that the target is not confirmed.

Three more flags change how a run is resumed or evaluated:

- `--report-only`: rebuilds the report and tables from the existing features and split files in `--output-dir`. It does no extraction, no retries and no downloads, and it does not change the existing files; the new outputs get a `_report` suffix (for example `summary_quick_report.md`). The Kepler-80 showcase figure is not rebuilt, because that needs a download; the summary says so. Files without a `pipeline_version` column work, with a warning.
- `--allow-legacy`: lets extraction resume into a features file whose `pipeline_version` differs or is missing. Without it, the run stops with an error (see [Pipeline version](#pipeline-version)).
- `--balance-halves`: trims the four star groups to the size of the smallest one. Without it, a failed star does not shrink the other groups.
- `--inverted-control`: passed to the extractor; it also writes `features_demo*_inverted.csv` (see [Key features](#key-features)). It is needed for the signal-or-noise stage below. Stars that were already extracted without it are not re-run, so use it on a fresh run or with `--no-resume` on the extractor.

**Signal-or-noise stage.** When the inverted file exists, the demo trains a separate bootstrap XGBoost ensemble that separates `MATCHED` signals of the training half (class 1) from `INVERTED` signals of training-half stars (class 0), without the stellar features. Its threshold on `p_real` is chosen on star-grouped out-of-fold predictions so that at least 95% of the training `MATCHED` signals stay above it. On the control half, a `CANDIDATE` verdict is kept only when `p_real` reaches the threshold; otherwise the signal becomes `REVIEW`, and the earlier verdict is kept in `decision_before_noise_gate`. Without the inverted file the stage is skipped with a warning and the verdicts are as before.

The metrics in `summary.md` are conditional on detection and on matching a KOI: they are computed only on signals that the search found and that matched a catalog KOI period (`MATCHED`). KOIs that the search missed, or found only at a harmonic, are not in them; the KOI recovery table in the summary reports those separately. Runtime estimates are given in [Performance](#performance).

### Showcase figure

`make_showcase.py` redraws the Kepler-80 (KIC 4852528) showcase figure for a finished run, with a choice of layout (`tall` or `wide`, the wide one is about 11 x 6.6 inches at 300 dpi and suits a journal page) and of label language (`ru` or `en`). It takes the detected signals from the features table of the run and, optionally, the verdicts and `p_mean` values from `control_signals_quick.csv`, matched by period within 1%; when no match is found, the panel title shows `p = n/a`. With `--lang both` the light curve and periodogram are computed once and two files, `showcase_kepler80_wide_ru.png` and `showcase_kepler80_wide_en.png`, are written to `--output-dir`; `--output` writes a single file instead. Verdicts are shown as `CANDIDATE`, `FALSE_POSITIVE` or `REVIEW`, and anything else as `n/a`. It downloads the Kepler light curve with the same settings as the quick extractor mode, so network access and the light curve cache are needed.

```bash
python make_showcase.py --features-file outputs/demo/features_demo_quick.csv --signals-file outputs/demo/control_signals_quick.csv --layout wide --lang both --output-dir outputs/showcase
```

### Output files (`outputs/demo/`)

With `--quick`, file names get a `_quick` suffix. With `--smoke-test`, the folder is `outputs/_smoke_test/demo/`.

- `summary.md`: the full analytical report. **This file is generated in Russian.** It contains:
  - balance statistics of the stellar parameters per group;
  - photometry and BLS diagnostics for every star;
  - the KOI recovery table on the control marked stars (exact, harmonic, not found, by disposition, period bin, catalog transit depth `koi_depth` in quartiles, and number of KOIs per star (1, 2-3, 4 or more), with 95% Wilson intervals, the number of KOIs outside the search range, and the end-to-end completeness: found exactly and verdict `CANDIDATE`). Catalog columns are used only in this table and never as features. The depth breakdown needs a `koi_depth` column in `data/labels/kepler_labels.csv`; the label query now requests it, but a label file downloaded earlier lacks it until it is fetched again;
  - with `--inverted-control`, the share of stars with at least one `INVERTED` signal, separately for KOI and unmarked stars (an estimate of the false-alarm frequency of the search), and the signal-or-noise stage: its threshold, the `p_real` distribution of control `MATCHED`, `UNMATCHED` and `INVERTED` signals, and the number of `CANDIDATE` verdicts among unmarked-star signals before and after the `p_real` condition;
  - the classification results with a confusion matrix, precision, recall, ROC-AUC, accuracy on accepted verdicts, the share of `REVIEW` verdicts, 95% bootstrap confidence intervals over stars, the Brier score with a five-bin reliability table, comparison with baselines (majority class, an SNR threshold with the direction found on the training half, and a logistic regression), and ablations (without the stellar features, without `transit_count` and `gap_fraction`, with the coverage features) with ROC-AUC intervals;
  - the distribution of the model score (`p_mean`) on unmarked stars, the share of `CANDIDATE` verdicts among their signals (the model is not trained on noise signals, so this is not a false-alarm rate), and the top 5 signals with SHAP factors;
  - the Kepler-80 recovery table, counting exact period matches and harmonic matches separately (format "N exact + M by harmonic out of K");
  - two correct and two wrong classification examples with SHAP explanations;
  - threats to validity and the SHA-256 immutability report.
- `diagnostics_stars.csv`: per-star photometric diagnostics (number of quarters, baseline, number of signals, `MATCHED`/`HARMONIC`/`UNMATCHED` counts, the number of `INVERTED` signals or `-` when the control did not run, status, maximum SDE/SNR, and per-stage timings).
- `control_signals.csv` and `control_signals.md`: all periodic signals of the control half with their features, `p_mean`, `p_std`, `p_real` and `decision_before_noise_gate` (when the signal-or-noise stage ran), verdicts (`CANDIDATE`, `FALSE_POSITIVE`, `REVIEW`), and SHAP factors.
- `features_demo_inverted.csv` (with `--inverted-control`): the `INVERTED` signals, with the same columns as the feature table.
- `showcase_kepler80.png`: a multi-panel figure with the raw photometry of all downloaded quarters, the detrended light curve, the BLS periodogram (SDE scale), and phase-folded profiles of the detected signals with the other transits masked.

## Search settings

### Search grid and detection

- **Period grid.** Unchanged: the astropy `autoperiod` grid over $P \in [0.5, \min(100, \text{baseline}/2)]$ d with `frequency_factor` 500 (normal) or 5000 (quick).
- **Duration grid.** Logarithmic from 0.04 to 0.5 d: 6 values in quick mode, 12 in normal mode (previously 4 linear values in $[0.05, 0.40]$ and 10 in $[0.04, 0.45]$). Durations longer than 0.3 of the trial period are not tried: the spectrum is computed in period bands, each with the durations allowed for it. A single call with long durations at $P \approx 0.5$ d produces a degenerate peak with a duty cycle near 1 that outranks real transits, and astropy rejects durations equal to the minimum period. astropy reports durations rounded to its internal bin of 0.004 d.
- **Period refinement.** After the strongest peak is chosen and before alias resolution, BLS is evaluated on a local grid of $\pm 1.5$ coarse steps with step $0.25 \times \text{duration} \times P / \text{baseline}$, on the same series; the best point gives the period, epoch, duration, depth and SNR. If the refined power is not higher, the coarse values are kept.
- **SNR and SDE.** `bls_depth_snr` is the depth SNR at the power peak, not the largest SNR anywhere in the spectrum. `bls_sde` is computed after a 501-point running median is subtracted from the spectrum; the uncorrected value is kept as `bls_sde_raw` and is not a feature.
- **Known limitation of quick mode.** With `frequency_factor=5000` on a 4-year Kepler baseline, the coarse grid step at long periods is many transit durations wide. On synthetic 1460-day series, quick mode recovered 0.7 d and 3 d planets exactly but found a 12 d planet at P/3 and a 40 d planet at P/2 or P/3 (now labelled `HARMONIC`); normal mode recovered all four exactly. Refinement cannot fix a peak that lands on a harmonic, and alias resolution tests only P/2, P/3 and 2P against the coarse spectrum.

### Extractor flags

Four flags of `1_data_extractor.py` control details of the signal search. `main.py` and the demo do not forward them, so pass them to the extractor directly:

```bash
python 1_data_extractor.py --mission Kepler --limit 100 --clip-upward-only --no-resolve-aliases
```

- **Outlier clipping (default: symmetric).** By default, outliers beyond 5 sigma are removed on both sides. `--clip-upward-only` (experimental) removes only upward outliers, keeping transit dips. On a small bench, upward-only clipping lost known planets in multi-planet systems, so it is off by default.
- **Isolated-outlier clipping (default: off).** `--clip-isolated` (experimental) replaces symmetric 5-sigma clipping with a robust one: the level is the median flux and sigma is 1.4826 times the median absolute deviation. It removes points more than 5 sigma above the level, and points more than 5 sigma below it only when they are isolated, that is, when neither neighbour within two median cadences is itself more than 2.5 sigma below the level. Dips of two or more consecutive low points are kept. It can produce false periods from multi-point downward artifacts. It cannot be combined with `--clip-upward-only`: the extractor stops with an error.
- **Restoring transit points (default: on).** Clipping can cut the bottom of real transits. Downward points removed by clipping that have a low neighbour (the same 2.5-sigma, two-cadence rule) are kept aside in a restore pool and detrended with the trend of the kept points, interpolated in time. For each found signal, the pool points inside windows of 1.25 transit durations around its transits are returned to the light curve only when they are at least 5% of all points in those windows and their number is at least 3 times the number expected by chance (pool size times the window share of the period). When the returned points are at least 20% of the points in the windows, duration, epoch and depth are re-estimated with BLS at the found period on the restored curve; otherwise, when points were returned, only the depth is re-measured with a box model. The transit count, depth and other signal features are then measured on that curve. When the transits of a signal are masked for the next search iteration, the pool points in the same windows are dropped as well. `--no-restore-transit-points` turns all of this off: the features are measured on the clipped light curve.
- Three columns of the diagnostics file (`<output>_diagnostics.csv`) describe the restoration: `restore_pool_points` is the size of the restore pool of the star (0 when restoration is off or nothing was clipped downward with a low neighbour); `restore_candidates` lists, for each recorded signal in order of `signal_index` and separated by `;`, the number of pool points inside its transit windows before the admission check; `restored_points` lists, in the same order, the number of points actually returned (0 when the check rejected them). For a star without diagnostics timings both lists are empty strings. None of these columns enters the feature table.
- **Alias resolution (default: on).** After each BLS peak, the extractor tests whether the true period is one half or one third of it (or twice it). A test is accepted only if the BLS power near the trial period and the box-model transit depth support it (within a 1% period tolerance). A signal whose period duplicates one already found (ratios 1, 2, 3, 1/2, or 1/3 within 1%) is not recorded, but its transits are masked and the iteration is used up. `--no-resolve-aliases` disables all of this.
- The number of corrected periods is written to the diagnostics file (`<output>_diagnostics.csv`) as `alias_corrections`, and the applied factors as `alias_factors`. Neither column enters the feature table.
- Bench: `bench/run_bench.bat` runs the extractor on 7 Kepler stars (`bench/targets_bench.csv`) with default settings and with `--no-resolve-aliases`, and `bench/compare_bench.py` compares the two. Optional arguments: `run_bench.bat [limit or -] [tag] [extra extractor flags]` writes to `outputs\bench_new<tag>` and `outputs\bench_old<tag>` and passes the extra flags to both runs, for example `bench\run_bench.bat - _v4 --inverted-control`. Use a new tag after a `PIPELINE_VERSION` change: resuming into the earlier bench folders stops with a version error. Recorded results are in [bench/BENCH_RESULTS.md](bench/BENCH_RESULTS.md).

## Resumable runs and fault tolerance

Long extraction runs can be interrupted and resumed. This applies to any environment.

- **Autosave.** The rows of each star are flushed to disk (`flush` + `fsync`) as soon as that star is finished.
- **Auto-resume.** Re-running the same command reads the finished stars from `--output` (or from the diagnostics file) and processes only the remaining ones. If the existing file lacks columns of the current feature schema (the centroid-shift columns), the extractor stops with an error: use `--no-resume` or a new `--output`.
- <a id="pipeline-version"></a>**Pipeline version.** Every feature row and diagnostics row has a `pipeline_version` column: the `PIPELINE_VERSION` constant of `onyx_features.py` (currently `8`) plus the `quick`, `flatten_days` and `max_signals` settings. When resuming, the extractor stops with an error if the existing file has a different version or no such column. `--allow-legacy` turns that stop into a warning. `pipeline_version` is never used as a model feature.
- `--no-resume`: start from scratch. This deletes the existing output, `_failed`, diagnostics, and `folded_curves.npz` files. Resuming is on by default.
- `--star-timeout <SECONDS>`: time limit per star (default 1800 s). A star whose download or computation fails or hangs is recorded in `<output>_failed.csv` (for example `extracted_features_failed.csv`), and the pipeline moves on. Each row of the failed table keeps the original error text, the stage, and the number of attempts; the demo also fills `star_type` and `split`. After the run, the failed stars get one more attempt in a retry pass; on the next run a star is retried until it has failed 3 times. If the disk is below `--min-free-gb`, a star waits up to 10 minutes (checked every 30 s) before it is recorded as failed. A star that ignores its own deadline is abandoned after three times `--star-timeout`, counted from the moment its worker started, and recorded as failed.
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

**Effect of the current search on runtime.** The profiles above were measured with the previous duration grid. On a synthetic 1460-day series (about 71,600 points), one quick-mode BLS iteration took about 8.0 s with the old grid and about 9.8 s with the new banded 6-duration grid (+20-25%); in normal mode, 12 instead of 10 durations took about 106.6 s instead of 100.8 s (+6%). Period refinement and the detrended SDE add well under a second per iteration. `--inverted-control` adds the BLS iterations of the inverted search, usually one or two per star (it stops at the first peak below the thresholds). Because the download dominates a quick-mode star (about 83 s of 106 s for Kepler-80), the time per star in quick mode should grow by roughly 5-15 s without the control and by roughly 10-30 s with it; these are estimates, not measurements on real stars.

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
