import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
import umap.umap_ as umap
from sklearn.decomposition import PCA
import joblib

warnings.filterwarnings('ignore')

def apply_log1p(X):
    return np.log1p(np.maximum(X, 0))

def get_pipeline(model=None):
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(apply_log1p)),
        ('scaler', RobustScaler())
    ]
    if model is not None:
        steps.append(('model', model))
    return Pipeline(steps)

def run_clustering(df_raw, feature_cols):
    X = df_raw[feature_cols].values
    pipe = get_pipeline()
    X_scaled = pipe.fit_transform(X)
    best_bic = np.inf
    best_gmm = None
    for n in range(2, 11):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    gmm_labels = best_gmm.predict(X_scaled)
    best_sil = -1
    best_kmeans = None
    for n in range(2, 11):
        km = KMeans(n_clusters=n, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        if sil > best_sil:
            best_sil = sil
            best_kmeans = km
    df_raw['cluster_gmm'] = gmm_labels
    df_raw['cluster_kmeans'] = best_kmeans.labels_
    summary = []
    for c in sorted(df_raw['cluster_gmm'].unique()):
        subset = df_raw[df_raw['cluster_gmm'] == c]
        total = len(subset)
        conf = len(subset[subset['label'] == 'CONFIRMED'])
        cand = len(subset[subset['label'] == 'CANDIDATE'])
        fp = len(subset[subset['label'] == 'FALSE POSITIVE'])
        med_odd_even = subset['odd_even_sigma'].median()
        med_sec_snr = subset['secondary_snr'].median()
        med_flux_std = subset['flux_std_ppm'].median()
        summary.append({
            'cluster': c,
            'size': total,
            'confirmed_frac': conf / total if total > 0 else 0.0,
            'candidate_frac': cand / total if total > 0 else 0.0,
            'fp_frac': fp / total if total > 0 else 0.0,
            'median_odd_even_sigma': med_odd_even,
            'median_sec_snr': med_sec_snr,
            'median_flux_std_ppm': med_flux_std
        })
    pd.DataFrame(summary).to_csv('outputs/cluster_summary.csv', index=False)
    return df_raw, X_scaled

def run_dim_reduction(X_scaled, labels_series, cluster_series):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=cluster_series, palette='tab10', alpha=0.7)
    plt.savefig('outputs/figures/umap_clusters.png', dpi=300)
    plt.close()
    plt.figure(figsize=(10, 8))
    valid_mask = labels_series.isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])
    sns.scatterplot(x=X_umap[valid_mask, 0], y=X_umap[valid_mask, 1], hue=labels_series[valid_mask], alpha=0.7)
    plt.savefig('outputs/figures/umap_labels.png', dpi=300)
    plt.close()
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_series, palette='tab10', alpha=0.7)
    plt.savefig('outputs/figures/pca_clusters.png', dpi=300)
    plt.close()
    pca_full = PCA(random_state=42).fit(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.savefig('outputs/figures/pca_explained_variance.png', dpi=300)
    plt.close()

def build_and_evaluate(df, feature_cols):
    df_labeled = df[df['label'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])].copy()
    if len(df_labeled) == 0:
        return
    df_labeled['target_bin'] = df_labeled['label'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
    groups = df_labeled['target_id'].values
    X_raw = df_labeled[feature_cols].values
    y = df_labeled['target_bin'].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(cv.split(X_raw, y, groups))
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
    xgb_base = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=pos_weight)
    pipe_xgb = get_pipeline(xgb_base)
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.01, 0.1],
        'model__subsample': [0.8, 1.0]
    }
    search = RandomizedSearchCV(pipe_xgb, param_grid, n_iter=5, scoring='roc_auc', cv=cv, random_state=42, n_jobs=1)
    search.fit(X_train_raw, y_train, groups=groups[train_idx])
    best_xgb_pipe = search.best_estimator_
    
    try:
        from sklearn.frozen import FrozenEstimator
        xgb_cal_kwargs = {'estimator': FrozenEstimator(best_xgb_pipe), 'method': 'isotonic'}
    except ImportError:
        xgb_cal_kwargs = {'estimator': best_xgb_pipe, 'cv': 'prefit', 'method': 'isotonic'}
        
    calibrated = CalibratedClassifierCV(**xgb_cal_kwargs)
    calibrated.fit(X_test_raw, y_test)
    y_pred_prob = calibrated.predict_proba(X_test_raw)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    pipe_rf = get_pipeline(rf)
    pipe_rf.fit(X_train_raw, y_train)
    
    try:
        from sklearn.frozen import FrozenEstimator
        rf_cal_kwargs = {'estimator': FrozenEstimator(pipe_rf), 'method': 'sigmoid'}
    except ImportError:
        rf_cal_kwargs = {'estimator': pipe_rf, 'cv': 'prefit', 'method': 'sigmoid'}
        
    rf_calibrated = CalibratedClassifierCV(**rf_cal_kwargs)
    rf_calibrated.fit(X_test_raw, y_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_prob)
    pr_auc = average_precision_score(y_test, y_pred_prob)
    metrics = {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1': float(f1),
        'ROC-AUC': float(auc),
        'PR-AUC': float(pr_auc)
    }
    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig('outputs/figures/confusion_matrix.png', dpi=300)
    plt.close()
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC AUC = {:.3f}'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.savefig('outputs/figures/roc_curve.png', dpi=300)
    plt.close()
    pr, rc, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(rc, pr, label='PR AUC = {:.3f}'.format(pr_auc))
    plt.legend()
    plt.savefig('outputs/figures/pr_curve.png', dpi=300)
    plt.close()
    importance = best_xgb_pipe.named_steps['model'].feature_importances_
    idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importance[idx][:15], y=np.array(feature_cols)[idx][:15])
    plt.savefig('outputs/figures/feature_importance.png', dpi=300)
    plt.close()
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='XGBoost Calibrated')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.savefig('outputs/figures/calibration_curve.png', dpi=300)
    plt.close()
    joblib.dump(best_xgb_pipe, 'outputs/models/xgboost_model.joblib')
    joblib.dump(pipe_rf, 'outputs/models/rf_model.joblib')
    joblib.dump(calibrated, 'outputs/models/calibrated_xgb.joblib')
    full_probs = calibrated.predict_proba(df[feature_cols].values)[:, 1]
    df['p_candidate'] = full_probs
    df.to_csv('outputs/predictions.csv', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='extracted_features.csv')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    feature_cols = [c for c in df.columns if c not in ['target_id', 'mission', 'label', 'p_candidate']]
    missing_ratio = df[feature_cols].isnull().mean(axis=1)
    df = df[missing_ratio <= 0.3].copy()
    df, X_scaled = run_clustering(df, feature_cols)
    run_dim_reduction(X_scaled, df['label'], df['cluster_gmm'])
    build_and_evaluate(df, feature_cols)

if __name__ == '__main__':
    main()
