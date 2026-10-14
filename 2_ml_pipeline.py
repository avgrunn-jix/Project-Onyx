import argparse
import json
import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import shap

warnings.filterwarnings('ignore')

def setup_logger():
    os.makedirs('logs', exist_ok=True)
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger = logging.getLogger('ml_pipeline')
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    return logger

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

def get_explanations(df, X_scaled, explainer, feature_cols, top_n=3):
    shap_vals = explainer.shap_values(X_scaled)
    factors = []
    for i in range(len(df)):
        sv = shap_vals[i]
        top_idx = np.argsort(np.abs(sv))[-top_n:][::-1]
        reason_strs = []
        for idx in top_idx:
            val = df.iloc[i][feature_cols[idx]]
            impact = 'positive' if sv[idx] > 0 else 'negative'
            reason_strs.append(f"{feature_cols[idx]}={val:.4g}({impact})")
        factors.append(' | '.join(reason_strs))
    return factors

def run_ml(args, logger, is_retrain=False):
    base_outputs = getattr(args, 'base_outputs', os.path.join('outputs', '_smoke_test') if getattr(args, 'smoke_test', False) else 'outputs')
    if not os.path.exists(args.input):
        logger.error(f'Input file {args.input} not found')
        return
    df = pd.read_csv(args.input)
    df_matched = df[df['match_status'] == 'MATCHED'].copy()
    if len(df_matched) == 0:
        logger.error('No matched signals found for training')
        return
    df_matched['target_bin'] = df_matched['label'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
    exclude_cols = ['signal_id', 'target_id', 'signal_index', 'mission', 'label', 'match_status', 'target_bin', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'p_candidate', 'n_signals_star']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X_raw = df_matched[feature_cols].values
    y = df_matched['target_bin'].values
    groups = df_matched['target_id'].values
    unique_groups = np.unique(groups)
    unique_classes = np.unique(y)
    class_counts = {int(k): int(v) for k, v in pd.Series(y).value_counts().items()}
    if len(df_matched) < 10 or len(unique_groups) < 5 or len(unique_classes) < 2 or min(class_counts.values()) < 2:
        msg = f"Insufficient data for training: found {len(df_matched)} matched signals across {len(unique_groups)} stars with class distribution {class_counts}. StratifiedGroupKFold requires at least 5 stars and 2 examples per class. Run extraction on more targets or check your labels."
        logger.error(msg)
        print(f"ERROR: {msg}", flush=True)
        return
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(cv.split(X_raw, y, groups))
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    scale_pipe = get_pipeline()
    X_train_scaled = scale_pipe.fit_transform(X_train_raw)
    X_test_scaled = scale_pipe.transform(X_test_raw)
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1.0
    xgb = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=ratio)
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.1]
    }
    pipe_xgb = get_pipeline(xgb)
    search = RandomizedSearchCV(pipe_xgb, param_grid, n_iter=2, scoring='roc_auc', cv=cv, random_state=42, n_jobs=1)
    search.fit(X_train_raw, y_train, groups=groups[train_idx])
    best_xgb_pipe = search.best_estimator_
    try:
        from sklearn.frozen import FrozenEstimator
        xgb_cal_kwargs = {'estimator': FrozenEstimator(best_xgb_pipe), 'method': 'isotonic'}
    except ImportError:
        xgb_cal_kwargs = {'estimator': best_xgb_pipe, 'cv': 'prefit', 'method': 'isotonic'}
    calibrated = CalibratedClassifierCV(**xgb_cal_kwargs)
    calibrated.fit(X_test_raw, y_test)
    xgb_shap = XGBClassifier(**best_xgb_pipe.named_steps['model'].get_params())
    xgb_shap.fit(X_train_scaled, y_train)
    explainer = shap.TreeExplainer(xgb_shap)
    shap_values = explainer.shap_values(X_test_scaled)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_cols, show=False)
    plt.savefig(os.path.join(base_outputs, 'figures', 'shap_summary.png'), bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_cols, plot_type='bar', show=False)
    plt.savefig(os.path.join(base_outputs, 'figures', 'shap_bar.png'), bbox_inches='tight')
    plt.close()
    ensemble_models = []
    n_models = 20
    for i in range(n_models):
        unique_groups = np.unique(groups[train_idx])
        boot_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        boot_idx = []
        for g in boot_groups:
            boot_idx.extend(np.where(groups[train_idx] == g)[0])
        boot_idx = np.array(boot_idx)
        m = XGBClassifier(**best_xgb_pipe.named_steps['model'].get_params())
        m.fit(X_train_scaled[boot_idx], y_train[boot_idx])
        ensemble_models.append(m)
    val_preds = np.array([m.predict_proba(X_train_scaled)[:, 1] for m in ensemble_models])
    val_mean = np.mean(val_preds, axis=0)
    val_std = np.std(val_preds, axis=0)
    t_high, t_low, s_max = 0.8, 0.2, 0.1
    for t_h in np.linspace(0.5, 0.95, 10):
        for s_m in np.linspace(0.05, 0.3, 6):
            accepted = (val_mean >= t_h) & (val_std <= s_m)
            if np.sum(accepted) > 0:
                prec = precision_score(y_train[accepted], np.ones(np.sum(accepted)), zero_division=0)
                if prec >= 0.95:
                    t_high, s_max = t_h, s_m
                    break
    for t_l in np.linspace(0.5, 0.05, 10):
        accepted = (val_mean <= t_l) & (val_std <= s_max)
        if np.sum(accepted) > 0:
            prec = precision_score(1 - y_train[accepted], np.ones(np.sum(accepted)), zero_division=0)
            if prec >= 0.95:
                t_low = t_l
                break
    test_preds = np.array([m.predict_proba(X_test_scaled)[:, 1] for m in ensemble_models])
    test_mean = np.mean(test_preds, axis=0)
    test_std = np.std(test_preds, axis=0)
    decision = np.array(['REVIEW'] * len(y_test), dtype=object)
    decision[(test_mean >= t_high) & (test_std <= s_max)] = 'CANDIDATE'
    decision[(test_mean <= t_low) & (test_std <= s_max)] = 'FALSE_POSITIVE'
    accepted_mask = decision != 'REVIEW'
    cov = np.sum(accepted_mask) / len(y_test)
    if np.sum(accepted_mask) > 0:
        pred_acc = (decision[accepted_mask] == 'CANDIDATE').astype(int)
        acc_prec = accuracy_score(y_test[accepted_mask], pred_acc)
    else:
        acc_prec = 0.0
    plt.figure()
    covs, precs = [], []
    for t in np.linspace(0.5, 0.99, 20):
        acc = (val_mean >= t) & (val_std <= s_max)
        if np.sum(acc) > 0:
            covs.append(np.sum(acc) / len(y_train))
            precs.append(precision_score(y_train[acc], np.ones(np.sum(acc)), zero_division=0))
    plt.plot(covs, precs, marker='o')
    plt.xlabel('Coverage')
    plt.ylabel('Precision on Accepted')
    plt.savefig(os.path.join(base_outputs, 'figures', 'coverage_vs_precision.png'))
    plt.close()
    df_fp = df_matched[df_matched['target_bin'] == 0].copy()
    fp_flags = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    fp_models = {}
    fp_metrics = {}
    if len(df_fp) > 0:
        X_fp = df_fp[feature_cols].values
        X_fp_scaled = scale_pipe.transform(X_fp)
        for flag in fp_flags:
            if flag in df_fp.columns:
                y_flag = df_fp[flag].fillna(0).astype(int).values
                if np.sum(y_flag) >= 30 and np.sum(y_flag == 0) >= 30:
                    m = XGBClassifier(random_state=42, eval_metric='logloss')
                    m.fit(X_fp_scaled, y_flag)
                    fp_models[flag] = m
                    fp_metrics[flag] = {
                        'count': int(np.sum(y_flag)),
                        'roc_auc': float(roc_auc_score(y_flag, m.predict_proba(X_fp_scaled)[:, 1]))
                    }
                else:
                    logger.info(f'Skipping {flag}: not enough positive/negative examples ({np.sum(y_flag)})')
    with open(os.path.join(base_outputs, 'metrics_fp_reasons.json'), 'w', encoding='utf-8') as f:
        json.dump(fp_metrics, f, indent=4)
    joblib.dump({
        'scale_pipe': scale_pipe,
        'best_xgb_pipe': best_xgb_pipe,
        'calibrated': calibrated,
        'ensemble': ensemble_models,
        'fp_models': fp_models,
        'feature_cols': feature_cols,
        'thresholds': {'t_high': t_high, 't_low': t_low, 's_max': s_max},
        'explainer': explainer
    }, os.path.join(base_outputs, 'models', 'pipeline_model.joblib'))
    df_test = df_matched.iloc[test_idx].copy()
    df_test['p_mean'] = test_mean
    df_test['p_std'] = test_std
    df_test['decision'] = decision
    for flag, m in fp_models.items():
        df_test[f'p_reason_{flag[-2:]}'] = m.predict_proba(X_test_scaled)[:, 1]
    reason_cols = [c for c in df_test.columns if c.startswith('p_reason_')]
    if reason_cols:
        def get_likely(row):
            max_c = max(reason_cols, key=lambda c: row[c])
            if row[max_c] > 0.5: return max_c[-2:].upper()
            return 'UNKNOWN'
        df_test['likely_reason'] = df_test.apply(get_likely, axis=1)
    df_test['top_factors'] = get_explanations(df_test, X_test_scaled, explainer, feature_cols)
    df_test.to_csv(os.path.join(base_outputs, 'predictions.csv'), index=False, encoding='utf-8')
    metrics = {
        'ROC-AUC': float(roc_auc_score(y_test, test_mean)),
        'Coverage': float(cov),
        'Precision_on_Accepted': float(acc_prec)
    }
    with open(os.path.join(base_outputs, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    logger.info('Running anomaly detection on all signals (Step J)...')
    try:
        X_all = df[feature_cols].values
        X_all_scaled = scale_pipe.transform(X_all)
        preds_all = np.array([m.predict_proba(X_all_scaled)[:, 1] for m in ensemble_models])
        p_mean_all = np.mean(preds_all, axis=0)
        p_std_all = np.std(preds_all, axis=0)
        decision_all = np.array(['REVIEW'] * len(df), dtype=object)
        decision_all[(p_mean_all >= t_high) & (p_std_all <= s_max)] = 'CANDIDATE'
        decision_all[(p_mean_all <= t_low) & (p_std_all <= s_max)] = 'FALSE_POSITIVE'
        iso = IsolationForest(n_estimators=500, random_state=42)
        lof = LocalOutlierFactor(n_neighbors=20)
        iso.fit(X_all_scaled)
        iso_scores = iso.decision_function(X_all_scaled)
        lof.fit(X_all_scaled)
        lof_scores = lof.negative_outlier_factor_
        df_anom = df.copy()
        df_anom['iso_score'] = iso_scores
        df_anom['lof_score'] = lof_scores
        df_anom['decision'] = decision_all
        iso_top = np.argsort(iso_scores)[:50]
        lof_top = np.argsort(lof_scores)[:50]
        intersection = np.intersect1d(iso_top, lof_top)
        anomaly_idx = [i for i in intersection if decision_all[i] not in ['CANDIDATE', 'FALSE_POSITIVE']]
        df_anomalies = df_anom.iloc[anomaly_idx].copy()
        df_anomalies.to_csv(os.path.join(base_outputs, 'anomalies.csv'), index=False, encoding='utf-8')
        npz_path = os.path.join(base_outputs, 'folded_curves.npz')
        if not os.path.exists(npz_path) and getattr(args, 'smoke_test', False):
            npz_path = os.path.join('outputs', 'folded_curves.npz')
        if os.path.exists(npz_path) and len(df_anomalies) > 0:
            npz = np.load(npz_path)
            fig, axes = plt.subplots(4, 5, figsize=(20, 16))
            axes = axes.flatten()
            plot_idx = 0
            for i, row in df_anomalies.iterrows():
                if plot_idx >= 20: break
                sid = str(row['signal_id'])
                if sid in npz:
                    curve = npz[sid]
                    phase_curve = curve[0]
                    axes[plot_idx].plot(np.linspace(-0.5, 0.5, 201), phase_curve)
                    axes[plot_idx].set_title(sid)
                    plot_idx += 1
            plt.tight_layout()
            plt.savefig(os.path.join(base_outputs, 'figures', 'anomalies_grid.png'))
            plt.close()
    except Exception as e:
        logger.error(f'Anomaly detection failed: {e}')
    logger.info('Training completed.')

def transfer_mode(args, logger):
    base_outputs = getattr(args, 'base_outputs', os.path.join('outputs', '_smoke_test') if getattr(args, 'smoke_test', False) else 'outputs')
    if not args.train_csv or not args.test_csv:
        logger.error('--train-csv and --test-csv required for transfer mode')
        return
    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)
    df_train = df_train[df_train['match_status'] == 'MATCHED'].copy()
    df_test = df_test[df_test['match_status'] == 'MATCHED'].copy()
    df_train['target_bin'] = df_train['label'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
    df_test['target_bin'] = df_test['label'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
    exclude_cols = ['signal_id', 'target_id', 'signal_index', 'mission', 'label', 'match_status', 'target_bin', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'p_candidate', 'n_signals_star']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols and c in df_test.columns]
    X_train = df_train[feature_cols].values
    y_train = df_train['target_bin'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['target_bin'].values
    from scipy.stats import ks_2samp
    shifts = []
    for i, c in enumerate(feature_cols):
        stat, pval = ks_2samp(X_train[:, i], X_test[:, i])
        shifts.append((c, stat))
    shifts.sort(key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(10, 15))
    names = [x[0] for x in shifts]
    vals = [x[1] for x in shifts]
    plt.barh(names, vals)
    plt.gca().invert_yaxis()
    plt.title('KS Shift by Feature (Kepler vs TESS)')
    plt.savefig(os.path.join(base_outputs, 'figures', 'ks_shift_by_feature.png'))
    plt.close()
    robust_exclude = ['baseline_d', 'transit_count', 'gap_fraction', 'bls_epoch', 'transit_depth_scatter_ppm']
    robust_cols = [c for c in feature_cols if c not in robust_exclude]
    robust_idx = [feature_cols.index(c) for c in robust_cols]
    scale_pipe = get_pipeline()
    X_train_scaled = scale_pipe.fit_transform(X_train)
    X_test_scaled = scale_pipe.transform(X_test)
    xgb1 = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb1.fit(X_train_scaled, y_train)
    preds1 = xgb1.predict_proba(X_test_scaled)[:, 1]
    auc1 = roc_auc_score(y_test, preds1)
    scale_pipe2 = get_pipeline()
    X_train_rob = scale_pipe2.fit_transform(X_train[:, robust_idx])
    X_test_rob = scale_pipe2.transform(X_test[:, robust_idx])
    xgb2 = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb2.fit(X_train_rob, y_train)
    preds2 = xgb2.predict_proba(X_test_rob)[:, 1]
    auc2 = roc_auc_score(y_test, preds2)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    xgb3 = XGBClassifier(random_state=42, eval_metric='logloss')
    groups_test = df_test['target_id'].values
    cv_preds = np.zeros(len(y_test))
    for tr, ts in cv.split(X_test, y_test, groups_test):
        s = get_pipeline()
        xtr = s.fit_transform(X_test[tr])
        xts = s.transform(X_test[ts])
        xgb3.fit(xtr, y_test[tr])
        cv_preds[ts] = xgb3.predict_proba(xts)[:, 1]
    auc3 = roc_auc_score(y_test, cv_preds)
    res = {
        'T1_Kepler_TESS_All': float(auc1),
        'T2_Kepler_TESS_Robust': float(auc2),
        'T3_TESS_CV': float(auc3)
    }
    with open(os.path.join(base_outputs, 'metrics_transfer.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)
    from sklearn.metrics import roc_curve
    fpr1, tpr1, _ = roc_curve(y_test, preds1)
    fpr2, tpr2, _ = roc_curve(y_test, preds2)
    fpr3, tpr3, _ = roc_curve(y_test, cv_preds)
    plt.figure()
    plt.plot(fpr1, tpr1, label=f'T1 (All): {auc1:.3f}')
    plt.plot(fpr2, tpr2, label=f'T2 (Robust): {auc2:.3f}')
    plt.plot(fpr3, tpr3, label=f'T3 (CV): {auc3:.3f}')
    plt.legend()
    plt.savefig(os.path.join(base_outputs, 'figures', 'transfer_roc.png'))
    plt.close()
    logger.info('Transfer experiments completed.')

def predict_mode(args, logger):
    base_outputs = getattr(args, 'base_outputs', os.path.join('outputs', '_smoke_test') if getattr(args, 'smoke_test', False) else 'outputs')
    model_path = os.path.join(base_outputs, 'models', 'pipeline_model.joblib')
    if not os.path.exists(model_path) and getattr(args, 'smoke_test', False):
        model_path = os.path.join('outputs', 'models', 'pipeline_model.joblib')
    if not os.path.exists(model_path):
        logger.error(f'No trained model found at {model_path}! Run --mode train first.')
        return
    if not os.path.exists(args.input):
        logger.error(f'Input file {args.input} not found')
        return
    model_dict = joblib.load(model_path)
    scale_pipe = model_dict['scale_pipe']
    ensemble_models = model_dict['ensemble']
    fp_models = model_dict['fp_models']
    feature_cols = model_dict['feature_cols']
    thresholds = model_dict['thresholds']
    explainer = model_dict.get('explainer', None)
    df = pd.read_csv(args.input)
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        msg = f"Input file {args.input} is missing {len(missing_cols)} feature columns required by trained model: {', '.join(missing_cols)}. Please re-extract features with the matching pipeline schema."
        logger.error(msg)
        print(f"ERROR: {msg}", flush=True)
        return
    X = df[feature_cols].values
    X_scaled = scale_pipe.transform(X)
    preds = np.array([m.predict_proba(X_scaled)[:, 1] for m in ensemble_models])
    p_mean = np.mean(preds, axis=0)
    p_std = np.std(preds, axis=0)
    decision = np.array(['REVIEW'] * len(df), dtype=object)
    decision[(p_mean >= thresholds['t_high']) & (p_std <= thresholds['s_max'])] = 'CANDIDATE'
    decision[(p_mean <= thresholds['t_low']) & (p_std <= thresholds['s_max'])] = 'FALSE_POSITIVE'
    df['p_mean'] = p_mean
    df['p_std'] = p_std
    df['decision'] = decision
    for flag, m in fp_models.items():
        df[f'p_reason_{flag[-2:]}'] = m.predict_proba(X_scaled)[:, 1]
    reason_cols = [c for c in df.columns if c.startswith('p_reason_')]
    if reason_cols:
        def get_likely(row):
            max_c = max(reason_cols, key=lambda c: row[c])
            if row[max_c] > 0.5: return max_c[-2:].upper()
            return 'UNKNOWN'
        df['likely_reason'] = df.apply(get_likely, axis=1)
    if explainer is not None:
        df['top_factors'] = get_explanations(df, X_scaled, explainer, feature_cols)
    df.sort_values(by='p_mean', ascending=False, inplace=True)
    out_file = args.output
    if not out_file:
        if args.mode == 'score-file':
            out_file = os.path.join(base_outputs, 'scored_file.csv')
        else:
            out_file = os.path.join(base_outputs, 'new_candidates.csv')
    elif getattr(args, 'smoke_test', False) and not '_smoke_test' in out_file:
        out_file = os.path.join(base_outputs, os.path.basename(out_file))
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    df.to_csv(out_file, index=False, encoding='utf-8')
    logger.info(f'Prediction completed. Saved to {out_file}')

def score_file_mode(args, logger):
    predict_mode(args, logger)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='extracted_features.csv')
    parser.add_argument('--output', default='')
    parser.add_argument('--train-csv', default='')
    parser.add_argument('--test-csv', default='')
    parser.add_argument('--mode', default='train', choices=['train', 'predict', 'score-file', 'retrain', 'transfer'])
    parser.add_argument('--smoke-test', action='store_true', help='[TEST ONLY] Run pipeline in isolated smoke test mode, writing outputs to outputs/_smoke_test/')
    args = parser.parse_args()
    logger = setup_logger()
    base_outputs = os.path.join('outputs', '_smoke_test') if args.smoke_test else 'outputs'
    os.makedirs(base_outputs, exist_ok=True)
    os.makedirs(os.path.join(base_outputs, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_outputs, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(base_outputs, 'explanations'), exist_ok=True)
    args.base_outputs = base_outputs
    if args.mode in ['train', 'retrain']:
        run_ml(args, logger, is_retrain=(args.mode == 'retrain'))
    elif args.mode in ['predict', 'score-file']:
        predict_mode(args, logger)
    elif args.mode == 'transfer':
        transfer_mode(args, logger)

if __name__ == '__main__':
    main()
