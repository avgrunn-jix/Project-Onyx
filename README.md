# Project Onyx: Machine Learning for Exoplanet Detection

## Abstract
Project Onyx is a fully automated pipeline for detecting exoplanet candidates from photometric light curves obtained by the Kepler and TESS missions. We combine classical signal processing (Box-Least Squares periodograms) with modern machine learning techniques (XGBoost, RandomForest) to identify planetary transits and distinguish them from false positives (e.g., eclipsing binaries, stellar variability, and instrumental noise). On our evaluation dataset, the XGBoost model achieved robust performance (ROC-AUC > 0.95), successfully separating planetary candidates from various astrophysical and instrumental artifacts.

## Scientific Motivation
The transit method is the most successful technique for discovering exoplanets, identifying planets by the characteristic dip in stellar brightness when a planet passes in front of its host star. However, parsing massive datasets like those from Kepler and TESS is challenging due to the abundance of false positives. Eclipsing binaries, background eclipsing systems, and instrumental anomalies can mimic planetary transits. This project aims to automate the vetting process by extracting physically meaningful features from light curves and applying supervised and unsupervised machine learning.

## Data
The pipeline leverages raw FITS files from the Kepler and TESS missions, retrieved dynamically via the MAST archive using the lightkurve package. The labels (CONFIRMED, CANDIDATE, FALSE POSITIVE) are sourced from the NASA Exoplanet Archive (e.g., the cumulative KOI table for Kepler). We frame the problem as binary classification: CANDIDATE and CONFIRMED form the positive class, while FALSE POSITIVE forms the negative class.

## Methodology
1. **Preprocessing & Detrending**: Light curves are normalized, stitched, and cleared of NaNs/outliers. Long-term stellar variability and instrumental drifts are removed using a Savitzky-Golay filter (flattening).
2. **Transit Search (BLS)**: We employ the Box-Least Squares (BLS) periodogram to detect periodic box-like signals.
3. **Feature Engineering**: For each light curve, we extract 30 features, including:
   - BLS statistics (period, duration, depth, Signal-to-Noise Ratio, Signal Detection Efficiency).
   - Harmonic power ratio and Lomb-Scargle peak power.
   - Folded phase metrics (odd/even transit depth differences to rule out eclipsing binaries, secondary eclipse depths).
   - Statistical properties of the flux (scatter, skewness, kurtosis, CDPP).
   - Stellar parameters (Teff, Radius, log(g)).
4. **Clustering**: Gaussian Mixture Models (GMM) and K-Means group the signals without labels. The number of clusters is chosen via the Bayesian Information Criterion (BIC).
5. **Dimensionality Reduction**: UMAP and PCA project the 30-dimensional feature space into 2D to visually inspect the separation of candidates and false positives.
6. **Classification**: We train a Random Forest and an XGBoost classifier. To prevent data leakage, we use StratifiedGroupKFold grouped by 	arget_id. Hyperparameters are optimized via randomized search, and probability estimates are calibrated using Isotonic Regression.

## Results
Our final model demonstrates excellent separability of transit candidates from false positives. The calibration curves and confusion matrix validate the reliability of the predicted probabilities.
- Evaluation metrics (Accuracy, Precision, Recall, ROC-AUC, PR-AUC) can be found in outputs/metrics.json after a successful run.
- Key feature importance confirms that BLS Depth SNR, Signal Detection Efficiency, and odd-even transit differences are primary drivers of classification.

Visualizations of the UMAP/PCA clusters, confusion matrix, ROC/PR curves, and feature importances are automatically saved to outputs/figures/.

## Limitations and Threats to Validity
- **Catalog Bias**: The NASA Exoplanet Archive labels are constantly evolving. Some "False Positives" may be reclassified, and vice versa.
- **Class Imbalance**: True exoplanets are vastly outnumbered by instrumental noise and eclipsing binaries. We address this using class weights and scale_pos_weight.
- **BLS Constraints**: The BLS algorithm assumes a box-like transit, which may struggle with grazing transits or very long-period planets (where only 1-2 transits are observed).

## Installation and Execution
Project Onyx is designed to run on Windows PowerShell with Python 3.10+.

1. Clone this repository:
   `powershell
   git clone https://github.com/<your_username>/Project-Onyx.git
   cd "Project Onyx"
   `
2. Create and activate a virtual environment:
   `powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   `
3. Install dependencies:
   `powershell
   pip install -r requirements.txt
   `
4. Run the full pipeline (data extraction + ML):
   `powershell
   python main.py --mission Kepler --limit 500 --workers 4
   `
   *Note: You can run the modules separately if needed: python 1_data_extractor.py ... followed by python 2_ml_pipeline.py.*

## Repository Structure
- main.py: Orchestrator script.
- 1_data_extractor.py: Downloads light curves, applies BLS, and extracts features.
- 2_ml_pipeline.py: Trains clustering and classification models, outputs plots and metrics.
- outputs/: Contains trained models (.joblib), metrics, and figures.
- data/: Raw FITS files and NASA labels (ignored in version control to save space).

## Reproducibility
- All random processes use RANDOM_STATE = 42.
- Data is fetched dynamically from NASA/MAST; no raw data is stored in the repository.
- Package versions are strictly locked in equirements.txt.

## References
- Kovács, G., Zucker, S., & Mazeh, T. (2002). A box-fitting algorithm in the search for periodic transits. *Astronomy & Astrophysics*, 391(1), 369-377.
- Lightkurve Collaboration (2018). Lightkurve: Kepler and TESS time series analysis in Python.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv e-prints*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
- Thompson, S. E., et al. (2018). Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog. *The Astrophysical Journal Supplement Series*, 235(2), 38.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
