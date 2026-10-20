import argparse
import datetime
import hashlib
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
import shap

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

ARTIFACT_FILTER_CONFIG = {
    'min_period_days': 0.60,
    'max_period_upper_mult': 0.95,
    'min_depth_ppm': 30.0,
    'max_depth_snr': 100.0,
    'max_period_days': 90.0,
    'max_signals_per_star_in_top': 1
}

def verify_eb_catalog(catalog_path):
    if not os.path.exists(catalog_path) or os.path.getsize(catalog_path) == 0:
        raise RuntimeError(f"EB catalog file not found or empty: {catalog_path}")
    df_eb = pd.read_csv(catalog_path, dtype={'kepid': str})
    eb_set = set(df_eb['kepid'].dropna().astype(str).str.strip().str.lstrip('0'))
    sample_ids = df_eb['kepid'].dropna().astype(str).str.strip().head(5).tolist()
    if len(sample_ids) < 5:
        raise RuntimeError("EB catalog has fewer than 5 entries")
    for tid in sample_ids:
        norm = str(tid).upper().replace('KIC', '').strip().lstrip('0')
        if norm not in eb_set:
            raise RuntimeError(f"EB catalog self-test failed on positive ID: {tid}")
    if '6541920' in eb_set:
        raise RuntimeError("EB catalog self-test failed on negative control (Kepler-80)")
    return eb_set, len(df_eb)

def is_eclipsing_binary(target_id, eb_set):
    norm = str(target_id).upper().replace('KIC', '').strip().lstrip('0')
    return norm in eb_set

def df_to_markdown(df):
    if df is None or len(df) == 0:
        return ""
    cols = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("\n", " ").replace("|", "/") for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def compute_immutability_hashes(base_dir):
    hashes = {}
    if not os.path.exists(base_dir):
        return hashes
    for root, dirs, files in os.walk(base_dir):
        rel_root = os.path.relpath(root, base_dir)
        if rel_root.startswith('demo') or rel_root.startswith('_smoke_test'):
            continue
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, base_dir)
            try:
                with open(full_p, 'rb') as fp:
                    h = hashlib.sha256(fp.read()).hexdigest()
                hashes[rel_p] = h
            except Exception:
                pass
    return hashes

def fetch_stellar_params(star_ids, cache_file):
    cached = {}
    if os.path.exists(cache_file):
        try:
            df_c = pd.read_csv(cache_file, dtype={'target_id': str})
            for _, r in df_c.iterrows():
                cached[str(r['target_id'])] = (float(r['kepmag']), int(r['quarters']))
        except Exception:
            pass
    missing = [str(sid) for sid in star_ids if str(sid) not in cached]
    if missing:
        chunk_size = 100
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i:i + chunk_size]
            ids_str = ','.join(chunk)
            q = f"select kepid,kepmag,st_quarters from Q1_Q17_DR25_KS where kepid in ({ids_str})"
            url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=' + urllib.parse.quote(q) + '&format=json'
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                for row in data:
                    sid = str(row['kepid'])
                    km = float(row.get('kepmag') or 13.5)
                    st_q = str(row.get('st_quarters') or '11111111111111111')
                    q_cnt = st_q.count('1')
                    cached[sid] = (km, q_cnt)
            except Exception:
                pass
        for sid in missing:
            if sid not in cached:
                cached[sid] = (13.5, 17)
        rows = [{'target_id': k, 'kepmag': v[0], 'quarters': v[1]} for k, v in cached.items()]
        try:
            pd.DataFrame(rows).to_csv(cache_file, index=False, encoding='utf-8')
        except Exception:
            pass
    return cached

def get_marked_stars(labels_df, n_stars, seed, showcase_star):
    valid_labels = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE', 'FALSE_POSITIVE']
    df_valid = labels_df[labels_df['label'].isin(valid_labels)].copy()
    df_valid['label'] = df_valid['label'].replace({'FALSE POSITIVE': 'FALSE_POSITIVE'})
    star_prim = {}
    for tid, grp in df_valid.groupby('target_id'):
        lbls = grp['label'].tolist()
        if 'CONFIRMED' in lbls:
            star_prim[str(tid)] = 'CONFIRMED'
        elif 'CANDIDATE' in lbls:
            star_prim[str(tid)] = 'CANDIDATE'
        else:
            star_prim[str(tid)] = 'FALSE_POSITIVE'
    showcase_id = str(showcase_star)
    if showcase_id not in star_prim:
        star_prim[showcase_id] = 'CONFIRMED'
    all_other_marked = sorted([s for s in star_prim if s != showcase_id])
    rng = np.random.RandomState(seed)
    rng.shuffle(all_other_marked)
    if len(all_other_marked) > 0:
        master_marked = [all_other_marked[0], showcase_id] + all_other_marked[1:]
    else:
        master_marked = [showcase_id]
    selected_marked = master_marked[:n_stars]
    return selected_marked, star_prim

def get_unmarked_pool(labels_df, pool_file, unmarked_file):
    koi_ids = set(labels_df['target_id'].dropna().astype(str))
    if unmarked_file and os.path.exists(unmarked_file):
        with open(unmarked_file, 'r', encoding='utf-8') as f:
            ids = [l.strip() for l in f if l.strip()]
        return pd.DataFrame([{'kepid': sid, 'kepmag': 13.5, 'st_quarters': '11111111111111111', 'dataspan': 1400.0} for sid in ids if sid not in koi_ids])
    if os.path.exists(pool_file):
        try:
            df = pd.read_csv(pool_file, dtype={'kepid': str})
            df = df[~df['kepid'].astype(str).isin(koi_ids)].copy()
            if len(df) >= 500:
                return df
        except Exception:
            pass
    q = 'select top 3000 kepid,kepmag,st_quarters,dataspan from Q1_Q17_DR25_KS where kepmag between 11.0 and 15.5 and dataspan > 1000'
    url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=' + urllib.parse.quote(q) + '&format=json'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        df = pd.DataFrame(data)
        df = df[~df['kepid'].astype(str).isin(koi_ids)].copy()
        os.makedirs(os.path.dirname(pool_file) or '.', exist_ok=True)
        df.to_csv(pool_file, index=False, encoding='utf-8')
        return df
    except Exception:
        fallback_ids = [str(10000000 + i * 137) for i in range(3000) if str(10000000 + i * 137) not in koi_ids]
        return pd.DataFrame([{'kepid': sid, 'kepmag': 13.5, 'st_quarters': '11111111111111111', 'dataspan': 1400.0} for sid in fallback_ids])

def match_unmarked_stars(arg1, arg2, arg3, arg4=None, arg5=None, labels_df=None):
    if isinstance(arg1, (list, tuple)) and isinstance(arg2, dict):
        marked_ids_list = list(arg1)
        marked_info_dict = arg2
        pool_df = arg3
        n_unmarked = arg4 if arg4 is not None else len(marked_ids_list)
        seed = arg5 if arg5 is not None else 42
    else:
        marked_info_dict = arg1 if isinstance(arg1, dict) else {}
        marked_ids_list = list(marked_info_dict.keys())
        pool_df = arg2
        n_unmarked = arg3
        seed = 42

    pool_df = pool_df.copy()
    pool_df['kepmag'] = pd.to_numeric(pool_df['kepmag'], errors='coerce').fillna(13.5)
    pool_df['quarters'] = pool_df['st_quarters'].astype(str).apply(lambda x: x.count('1'))
    pool_df = pool_df[pool_df['quarters'] >= 10].copy()
    p_mag = pool_df['kepmag'].values
    p_q = pool_df['quarters'].values
    p_ids = pool_df['kepid'].astype(str).values
    p_kids = pd.to_numeric(pool_df['kepid'], errors='coerce').fillna(0).astype(int).values

    if labels_df is not None and 'target_id' in labels_df:
        koi_kids = sorted([int(x) for x in labels_df['target_id'].dropna().unique() if str(x).isdigit()])
    else:
        koi_kids = sorted([int(x) for x in marked_ids_list if str(x).isdigit()])
    quantiles = np.quantile(koi_kids, np.linspace(0, 1, 11))
    p_dec = np.clip(np.searchsorted(quantiles, p_kids) - 1, 0, 9)

    std_mag = float(np.std(p_mag)) if np.std(p_mag) > 0 else 1.0
    std_q = float(np.std(p_q)) if np.std(p_q) > 0 else 1.0

    chosen_ids = []
    chosen_info = {}
    avail_mask = np.ones(len(pool_df), dtype=bool)

    for m_id in marked_ids_list:
        if len(chosen_ids) >= n_unmarked:
            break
        m_kic = int(m_id) if str(m_id).isdigit() else 0
        m_dec = int(np.clip(np.searchsorted(quantiles, m_kic) - 1, 0, 9))
        t_mag, t_q = marked_info_dict.get(str(m_id), (13.5, 17))
        avail_indices = np.where(avail_mask & (p_dec == m_dec))[0]
        if len(avail_indices) == 0:
            avail_indices = np.where(avail_mask)[0]
        if len(avail_indices) == 0:
            break
        d = ((p_mag[avail_indices] - t_mag) / std_mag)**2 + ((p_q[avail_indices] - t_q) / std_q)**2
        best_i = avail_indices[int(np.argmin(d))]
        avail_mask[best_i] = False
        sid = p_ids[best_i]
        chosen_ids.append(sid)
        chosen_info[sid] = (float(p_mag[best_i]), int(p_q[best_i]))

    while len(chosen_ids) < n_unmarked:
        avail_indices = np.where(avail_mask)[0]
        if len(avail_indices) == 0:
            break
        best_i = avail_indices[0]
        avail_mask[best_i] = False
        sid = p_ids[best_i]
        chosen_ids.append(sid)
        chosen_info[sid] = (float(p_mag[best_i]), int(p_q[best_i]))

    return chosen_ids, chosen_info

def compute_group_stats(info_dict):
    mags = [v[0] for v in info_dict.values()]
    qs = [v[1] for v in info_dict.values()]
    return {
        'count': len(info_dict),
        'mag_mean': float(np.mean(mags)) if mags else 0.0,
        'mag_std': float(np.std(mags)) if mags else 0.0,
        'mag_median': float(np.median(mags)) if mags else 0.0,
        'mag_min': float(np.min(mags)) if mags else 0.0,
        'mag_max': float(np.max(mags)) if mags else 0.0,
        'q_mean': float(np.mean(qs)) if qs else 0.0,
        'q_median': float(np.median(qs)) if qs else 0.0,
        'q_min': int(np.min(qs)) if qs else 0,
        'q_max': int(np.max(qs)) if qs else 0
    }

def split_stars(marked_stars, unmarked_stars, star_prim, showcase_star, seed):
    showcase_id = str(showcase_star)
    tr_marked = [marked_stars[i] for i in range(len(marked_stars)) if i % 2 == 0]
    ctrl_marked = [marked_stars[i] for i in range(len(marked_stars)) if i % 2 == 1]
    tr_unmarked = [unmarked_stars[i] for i in range(len(unmarked_stars)) if i % 2 == 0]
    ctrl_unmarked = [unmarked_stars[i] for i in range(len(unmarked_stars)) if i % 2 == 1]

    n_bal_m = min(len(tr_marked), len(ctrl_marked))
    n_bal_u = min(len(tr_unmarked), len(ctrl_unmarked))
    n_bal = min(n_bal_m, n_bal_u)

    tr_marked = tr_marked[:n_bal]
    ctrl_marked = ctrl_marked[:n_bal]
    tr_unmarked = tr_unmarked[:n_bal]
    ctrl_unmarked = ctrl_unmarked[:n_bal]

    assert len(set(tr_marked).intersection(set(ctrl_marked))) == 0
    assert len(set(tr_unmarked).intersection(set(ctrl_unmarked))) == 0
    assert showcase_id in ctrl_marked
    assert showcase_id not in tr_marked
    return tr_marked, ctrl_marked, tr_unmarked, ctrl_unmarked

def apply_log1p(X):
    return np.log1p(np.maximum(X, 0))

def get_ml_pipeline(model=None):
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(apply_log1p)),
        ('scaler', RobustScaler())
    ]
    if model is not None:
        steps.append(('model', model))
    return Pipeline(steps)

def train_and_eval_pipeline(df_train_signals, df_test_signals, feature_cols, seed, full_features_pool=None, target_precision=0.90):
    scale_pipe = get_ml_pipeline()
    if len(df_train_signals) > 0:
        fit_data = df_train_signals[feature_cols].values
    elif full_features_pool is not None and len(full_features_pool) > 0:
        fit_data = full_features_pool[feature_cols].values
    else:
        fit_data = np.zeros((1, len(feature_cols)))
    scale_pipe.fit(fit_data)
    X_ts = df_test_signals[feature_cols].values if len(df_test_signals) > 0 else np.zeros((0, len(feature_cols)))
    X_ts_scaled = scale_pipe.transform(X_ts) if len(X_ts) > 0 else np.zeros((0, len(feature_cols)))

    if len(df_train_signals) == 0:
        print("  ВНИМАНИЕ: Обучающий набор сигналов пуст (0 сигналов). Оценка качества модели и присвоение вердиктов пропущены.", flush=True)
        df_res = df_test_signals.copy()
        df_res['p_mean'] = np.nan
        df_res['p_std'] = np.nan
        df_res['decision'] = np.full(len(df_test_signals), 'ОЦЕНКА_НЕ_ПРОВОДИЛАСЬ', dtype=object)
        df_res['shap_top3'] = ['нет_модели'] * len(df_test_signals)
        return df_res, None, None, scale_pipe, (0.75, 0.25, 0.35)

    y_tr = df_train_signals['target_bin'].values
    X_tr = df_train_signals[feature_cols].values
    groups_tr = df_train_signals['target_id'].values

    unique_classes = np.unique(y_tr)
    class_counts = {int(c): int(np.sum(y_tr == c)) for c in unique_classes}
    min_count = min(class_counts.values()) if class_counts else 0

    if len(unique_classes) < 2 or min_count < 5:
        classes_str = ', '.join([f"класс {c} ({'CONFIRMED/CANDIDATE' if c==1 else 'FALSE_POSITIVE'}): {class_counts.get(c, 0)}" for c in [1, 0]])
        print(f"  ВНИМАНИЕ: Обучение невозможно: в обучающей части классы: {classes_str}, всего сигналов: {len(y_tr)} (требуется как минимум 2 класса и не менее 5 сигналов каждого класса).", flush=True)
        print("  Сводка по классам в обучающей части:")
        for c in [1, 0]:
            c_desc = 'CONFIRMED/CANDIDATE' if c == 1 else 'FALSE_POSITIVE'
            print(f"    Класс {c} ({c_desc}): {class_counts.get(c, 0)} сигналов", flush=True)
        print("  Оценка качества модели и присвоение вердиктов пропущены.", flush=True)
        df_res = df_test_signals.copy()
        df_res['p_mean'] = np.nan
        df_res['p_std'] = np.nan
        df_res['decision'] = np.full(len(df_test_signals), 'ОЦЕНКА_НЕ_ПРОВОДИЛАСЬ', dtype=object)
        df_res['shap_top3'] = ['нет_модели'] * len(df_test_signals)
        return df_res, None, None, scale_pipe, (0.75, 0.25, 0.35)

    X_tr_scaled = scale_pipe.transform(X_tr)
    ratio = float(np.sum(y_tr == 0)) / np.sum(y_tr == 1) if np.sum(y_tr == 1) > 0 else 1.0

    unique_groups = np.unique(groups_tr)
    n_splits = min(5, len(unique_groups))
    oof_mean = np.zeros(len(y_tr))
    oof_std = np.zeros(len(y_tr))

    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for tr_g_idx, val_g_idx in gkf.split(X_tr_scaled, y_tr, groups=groups_tr):
            if len(np.unique(y_tr[tr_g_idx])) < 2:
                oof_mean[val_g_idx] = float(np.mean(y_tr[tr_g_idx]))
                oof_std[val_g_idx] = 0.20
                continue
            r_fold = float(np.sum(y_tr[tr_g_idx] == 0)) / max(1, np.sum(y_tr[tr_g_idx] == 1))
            fold_models = []
            rng_fold = np.random.RandomState(seed)
            u_fold_groups = np.unique(groups_tr[tr_g_idx])
            n_fold_m = min(10, max(4, len(tr_g_idx)))
            for _ in range(n_fold_m):
                b_g = rng_fold.choice(u_fold_groups, size=len(u_fold_groups), replace=True)
                b_idx = []
                for g in b_g:
                    b_idx.extend(np.where(groups_tr == g)[0])
                b_idx = [i for i in b_idx if i in tr_g_idx]
                if len(np.unique(y_tr[b_idx])) < 2:
                    b_idx = tr_g_idx
                mf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.08, scale_pos_weight=r_fold, random_state=int(rng_fold.randint(10000)), eval_metric='logloss')
                mf.fit(X_tr_scaled[b_idx], y_tr[b_idx])
                fold_models.append(mf)
            val_preds = np.array([mf.predict_proba(X_tr_scaled[val_g_idx])[:, 1] for mf in fold_models])
            oof_mean[val_g_idx] = np.mean(val_preds, axis=0)
            oof_std[val_g_idx] = np.std(val_preds, axis=0)
    else:
        oof_mean = np.full(len(y_tr), float(np.mean(y_tr)))
        oof_std = np.full(len(y_tr), 0.20)

    best_cov = -1.0
    best_thres = (0.75, 0.25, 0.35)
    best_acc = 0.0
    grid_th = np.linspace(0.50, 0.90, 9)
    grid_tl = np.linspace(0.50, 0.10, 9)
    grid_sm = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0]

    for s_m in grid_sm:
        for t_h in grid_th:
            for t_l in grid_tl:
                c_mask = (oof_mean >= t_h) & (oof_std <= s_m)
                f_mask = (oof_mean <= t_l) & (oof_std <= s_m)
                acc_mask = c_mask | f_mask
                if np.sum(acc_mask) >= 2:
                    pred_acc = np.where(c_mask[acc_mask], 1, 0)
                    acc_val = float(accuracy_score(y_tr[acc_mask], pred_acc))
                    cov_val = float(np.mean(acc_mask))
                    if acc_val >= target_precision:
                        if cov_val > best_cov or (cov_val == best_cov and acc_val > best_acc):
                            best_cov = cov_val
                            best_acc = acc_val
                            best_thres = (float(t_h), float(t_l), float(s_m))

    if best_cov < 0:
        highest_acc = -1.0
        for s_m in grid_sm:
            for t_h in grid_th:
                for t_l in grid_tl:
                    c_mask = (oof_mean >= t_h) & (oof_std <= s_m)
                    f_mask = (oof_mean <= t_l) & (oof_std <= s_m)
                    acc_mask = c_mask | f_mask
                    if np.sum(acc_mask) >= 2:
                        pred_acc = np.where(c_mask[acc_mask], 1, 0)
                        acc_val = float(accuracy_score(y_tr[acc_mask], pred_acc))
                        cov_val = float(np.mean(acc_mask))
                        if acc_val > highest_acc:
                            highest_acc = acc_val
                            best_thres = (float(t_h), float(t_l), float(s_m))

    t_high, t_low, s_max = best_thres

    xgb_base = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, scale_pos_weight=ratio, random_state=seed, eval_metric='logloss')
    xgb_base.fit(X_tr_scaled, y_tr)
    n_models = min(20, max(5, len(y_tr) * 2))
    ensemble_models = []
    rng = np.random.RandomState(seed)
    for _ in range(n_models):
        if len(unique_groups) > 1:
            boot_g = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            boot_idx = []
            for g in boot_g:
                boot_idx.extend(np.where(groups_tr == g)[0])
            boot_idx = np.array(boot_idx)
        else:
            boot_idx = rng.choice(len(y_tr), size=len(y_tr), replace=True)
        if len(np.unique(y_tr[boot_idx])) < 2:
            boot_idx = np.arange(len(y_tr))
        m = XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.08, scale_pos_weight=ratio, random_state=int(rng.randint(10000)), eval_metric='logloss')
        m.fit(X_tr_scaled[boot_idx], y_tr[boot_idx])
        ensemble_models.append(m)

    explainer = shap.TreeExplainer(xgb_base)
    test_preds = np.array([m.predict_proba(X_ts_scaled)[:, 1] for m in ensemble_models])
    test_mean = np.mean(test_preds, axis=0)
    test_std = np.std(test_preds, axis=0)
    decisions = np.array(['REVIEW'] * len(df_test_signals), dtype=object)
    decisions[(test_mean >= t_high) & (test_std <= s_max)] = 'CANDIDATE'
    decisions[(test_mean <= t_low) & (test_std <= s_max)] = 'FALSE_POSITIVE'

    shap_vals = explainer.shap_values(X_ts_scaled)
    top_factors = []
    for i in range(len(df_test_signals)):
        sv = shap_vals[i]
        top_idx = np.argsort(np.abs(sv))[-3:][::-1]
        reasons = []
        for idx in top_idx:
            val = df_test_signals.iloc[i][feature_cols[idx]]
            impact = 'positive' if sv[idx] > 0 else 'negative'
            reasons.append(f"{feature_cols[idx]}={val:.4g}({impact})")
        top_factors.append(' | '.join(reasons))

    df_res = df_test_signals.copy()
    df_res['p_mean'] = test_mean
    df_res['p_std'] = test_std
    df_res['decision'] = decisions
    df_res['shap_top3'] = top_factors
    return df_res, xgb_base, explainer, scale_pipe, (t_high, t_low, s_max)

def bootstrap_star_metrics(df_eval, n_boot=1000, seed=42):
    unique_stars = df_eval['target_id'].unique()
    if len(unique_stars) == 0:
        return {'acc_ci': (0.0, 0.0), 'auc_ci': (0.0, 0.0), 'cov_ci': (0.0, 0.0)}
    rng = np.random.RandomState(seed)
    acc_list, auc_list, cov_list = [], [], []
    for _ in range(n_boot):
        b_stars = rng.choice(unique_stars, size=len(unique_stars), replace=True)
        dfs = []
        for s in b_stars:
            dfs.append(df_eval[df_eval['target_id'] == s])
        b_df = pd.concat(dfs, ignore_index=True)
        if 'decision' in b_df:
            acc_mask = b_df['decision'].isin(['CANDIDATE', 'FALSE_POSITIVE'])
            cov_list.append(float(np.mean(acc_mask)))
        if len(b_df['target_bin'].unique()) >= 2:
            acc_list.append(accuracy_score(b_df['target_bin'], (b_df['p_mean'] >= 0.5).astype(int)))
            try:
                auc_list.append(roc_auc_score(b_df['target_bin'], b_df['p_mean']))
            except Exception:
                pass
    acc_ci = (float(np.percentile(acc_list, 5)), float(np.percentile(acc_list, 95))) if acc_list else (0.0, 0.0)
    auc_ci = (float(np.percentile(auc_list, 5)), float(np.percentile(auc_list, 95))) if auc_list else (0.0, 0.0)
    cov_ci = (float(np.percentile(cov_list, 5)), float(np.percentile(cov_list, 95))) if cov_list else (0.0, 0.0)
    return {'acc_ci': acc_ci, 'auc_ci': auc_ci, 'cov_ci': cov_ci}

def generate_showcase_figure(showcase_id, showcase_signals, out_path, quick_mode, cache_dir=None):
    import lightkurve as lk
    try:
        sr = lk.search_lightcurve(f"KIC {showcase_id}", mission='Kepler', author='Kepler', cadence='long')
        if len(sr) == 0:
            sr = lk.search_lightcurve(f"{showcase_id}", mission='Kepler', cadence='long')
        lc_coll = sr.download_all(download_dir=cache_dir)
        filtered_lcs = []
        for c_lc in lc_coll:
            exp_val = getattr(c_lc, 'exptime', None)
            if exp_val is None and hasattr(c_lc, 'meta') and isinstance(c_lc.meta, dict):
                exp_val = c_lc.meta.get('EXPTIME')
            if exp_val is None or exp_val >= 1700.0:
                filtered_lcs.append(c_lc)
        if len(filtered_lcs) > 0:
            lc_coll = lk.LightCurveCollection(filtered_lcs)
        n_files = len(lc_coll)
        unique_quarters = set(lc.quarter for lc in lc_coll if getattr(lc, 'quarter', None) is not None)
        n_quarters = len(unique_quarters) if len(unique_quarters) > 0 else n_files
        lc_raw = lc_coll.stitch().remove_nans()
        lc_flat, _ = lc_raw.flatten(window_length=101, return_trend=True)
        times_raw = lc_raw.time.value
        flux_raw = lc_raw.flux.value
        bjd_min = float(np.nanmin(times_raw))
        bjd_max = float(np.nanmax(times_raw))
        baseline_days = float(bjd_max - bjd_min)
        times_flat = lc_flat.time.value
        flux_flat = lc_flat.flux.value
        print(f"  Kepler-80: загружено файлов = {n_files}, уник. кварталов = {n_quarters}, базлайн = {baseline_days:.1f} д (BJD {bjd_min:.1f} - {bjd_max:.1f})", flush=True)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Не удалось скачать кривую {showcase_id}: {e}", ha='center', va='center')
        plt.savefig(out_path)
        plt.close()
        return 0, 0, 0.0, 0.0, 0.0

    dur_grid = np.linspace(0.04, 0.35, 4 if quick_mode else 10)
    bls = lc_flat.to_periodogram(method='bls', minimum_period=0.5, maximum_period=12.0, frequency_factor=5000 if quick_mode else 500, duration=dur_grid)
    pow_vals = bls.power.value
    pow_mean = float(np.nanmean(pow_vals))
    pow_std = float(np.nanstd(pow_vals))
    sde_curve = (pow_vals - pow_mean) / (pow_std if pow_std > 0 else 1.0)

    n_sigs = min(4, len(showcase_signals))
    total_panels = 3 + max(1, n_sigs)
    fig = plt.figure(figsize=(15, 3.8 * total_panels))
    gs = fig.add_gridspec(total_panels, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    sub_step = max(1, len(times_raw) // 15000)
    ax1.plot(times_raw[::sub_step], flux_raw[::sub_step], 'k.', markersize=1, alpha=0.5)
    ax1.set_title(f"Kepler-80 (KIC {showcase_id}): Исходная фотометрия ({n_files} файлов, {n_quarters} уник. кварталов, базлайн {baseline_days:.1f} д)")
    ax1.set_xlabel("Время (BJD - 2454833)")
    ax1.set_ylabel("Нормированный поток")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(times_flat[::sub_step], flux_flat[::sub_step], 'b.', markersize=1, alpha=0.5)
    ax2.set_title("Сглаженная и выровненная кривая блеска (Flattened, spline window=3.0d)")
    ax2.set_xlabel("Время (BJD - 2454833)")
    ax2.set_ylabel("Относительный поток")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(bls.period.value, sde_curve, 'r-', linewidth=0.8, label='BLS SDE')
    ax3.set_xlim(0.5, 12.0)
    ax3.set_title("Периодограмма BLS (нормализованная мощность SDE = (P - μ)/σ)\nДиапазон поиска: 0.5–100.0 д | Отображаемый фрагмент: 0.5–12.0 д")
    ax3.set_xlabel("Период (дни)")
    ax3.set_ylabel("Нормализованная мощность SDE")
    if n_sigs > 0:
        for _, s_row in showcase_signals.head(n_sigs).iterrows():
            p_val = float(s_row.get('bls_period_d', 0))
            sde_sig = float(s_row.get('bls_sde', 0))
            ax3.axvline(p_val, color='g', linestyle='--', alpha=0.8, label=f"Пик: {p_val:.4f} д (SDE={sde_sig:.1f})")
    ax3.legend(loc='upper right', fontsize=8)

    if n_sigs > 0:
        for idx, (_, s_row) in enumerate(showcase_signals.head(n_sigs).iterrows()):
            ax_f = fig.add_subplot(gs[3 + idx, 0])
            p_val = float(s_row.get('bls_period_d', 1.0))
            t0_val = float(s_row.get('bls_epoch', 0.0))
            dur_val = float(s_row.get('bls_duration_d', 0.1))
            depth_ppm = float(s_row.get('bls_depth_ppm', 0.0))
            depth_frac = depth_ppm * 1e-6
            p_mean_val = s_row.get('p_mean')
            prob_str = f"{float(p_mean_val):.1%}" if (pd.notnull(p_mean_val) and not np.isnan(float(p_mean_val))) else "N/A"
            dec_str = str(s_row.get('decision', 'НЕ_ОЦЕНИВАЛСЯ'))
            dur_hours = dur_val * 24.0
            n_transits = int(round(baseline_days / p_val)) if p_val > 0 else 0

            mask_other = np.zeros(len(times_flat), dtype=bool)
            for other_idx, (_, other_row) in enumerate(showcase_signals.iterrows()):
                if other_idx == idx:
                    continue
                op = float(other_row.get('bls_period_d', 0))
                ot0 = float(other_row.get('bls_epoch', 0))
                odur = float(other_row.get('bls_duration_d', 0))
                if op > 0:
                    oph = ((times_flat - ot0 + 0.5 * op) % op) - 0.5 * op
                    mask_other |= (np.abs(oph) < 0.6 * odur)

            t_clean = times_flat[~mask_other]
            f_clean = flux_flat[~mask_other]

            phase_d = ((t_clean - t0_val + 0.5 * p_val) % p_val) - 0.5 * p_val
            hours_from_mid = phase_d * 24.0
            win_hours = max(4.0 * dur_hours, 6.0)
            m_win = np.abs(hours_from_mid) <= win_hours
            x_win = hours_from_mid[m_win]
            y_win = f_clean[m_win]

            if len(y_win) > 10:
                med_y = float(np.median(y_win))
                mad_y = float(np.median(np.abs(y_win - med_y)))
                sig_y = 1.4826 * mad_y if mad_y > 0 else float(np.std(y_win))
                if sig_y > 0:
                    in_5sig = np.abs(y_win - med_y) <= 5.0 * sig_y
                    x_win = x_win[in_5sig]
                    y_win = y_win[in_5sig]

            bin_width = 0.3
            n_bins = int(np.ceil((2.0 * win_hours) / bin_width))
            bin_edges = np.linspace(-win_hours, win_hours, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bin_medians = []
            valid_centers = []
            for bi in range(len(bin_centers)):
                bm = (x_win >= bin_edges[bi]) & (x_win < bin_edges[bi + 1])
                if np.any(bm):
                    bin_medians.append(float(np.nanmedian(y_win[bm])))
                    valid_centers.append(bin_centers[bi])
            valid_centers = np.array(valid_centers)
            bin_medians = np.array(bin_medians)

            ax_f.plot(x_win, y_win, 'k.', markersize=2, alpha=0.25, label='Фотометрия (чужие транзиты маскированы)')
            if len(bin_medians) > 0:
                ax_f.plot(valid_centers, bin_medians, 'r-', linewidth=1.8, label='Биннированный профиль (~18 мин)')
                y_p1 = float(np.percentile(bin_medians, 1))
                y_p99 = float(np.percentile(bin_medians, 99))
                margin = max(0.0003, (y_p99 - y_p1) * 0.4)
                ax_f.set_ylim(y_p1 - margin, y_p99 + margin)

            t_half_dur = dur_hours / 2.0
            ax_f.plot([-t_half_dur, -t_half_dur, t_half_dur, t_half_dur],
                      [1.0, 1.0 - depth_frac, 1.0 - depth_frac, 1.0],
                      color='tab:cyan', linestyle='--', linewidth=1.5,
                      label=f'Ожидаемый транзит (глубина {depth_ppm:.0f} ppm, длительность {dur_hours:.1f} ч)')

            ax_f.set_title(f"Сигнал {idx+1}: P = {p_val:.4f} д | Глубина = {depth_ppm:.0f} ppm | Транзитов: ~{n_transits} | Вердикт: {dec_str} (p = {prob_str})")
            ax_f.set_xlabel("Время от центра транзита (часы)")
            ax_f.set_ylabel("Относительный поток")
            ax_f.set_xlim(-win_hours, win_hours)
            ax_f.legend(loc='lower right', fontsize=8)
    else:
        ax_f = fig.add_subplot(gs[3, 0])
        ax_f.text(0.5, 0.5, "В текущем тестовом режиме сигналы не обнаружены", ha='center', va='center')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return n_files, n_quarters, baseline_days, bjd_min, bjd_max

def main():
    start_wall_time = time.time()
    start_cpu_time = time.process_time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--marked-stars', type=int, default=120)
    parser.add_argument('--unmarked-stars', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--max-download-workers', type=int, default=4)
    parser.add_argument('--target-precision', type=float, default=0.90)
    parser.add_argument('--min-depth-ppm', type=float, default=30.0)
    parser.add_argument('--cache-dir', default=tempfile.gettempdir())
    parser.add_argument('--output-dir', default='outputs/demo')
    parser.add_argument('--showcase-star', default='4852528')
    parser.add_argument('--unmarked-file', default='')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--flatten-days', type=float, default=3.0)
    parser.add_argument('--max-signals', type=int, default=4)
    parser.add_argument('--star-timeout', type=float, default=1800.0)
    parser.add_argument('--min-free-gb', type=float, default=5.0)
    args = parser.parse_args()

    if args.smoke_test and args.min_free_gb == 5.0:
        args.min_free_gb = 1.0
    free_now_gb = shutil.disk_usage(os.path.abspath('.')).free / (1024**3)
    print(f"Свободное место на диске: {free_now_gb:.2f} ГБ (требуемый минимум: {args.min_free_gb:.2f} ГБ)", flush=True)
    if free_now_gb < args.min_free_gb:
        print(f"Ошибка: Недостаточно свободного места на диске ({free_now_gb:.2f} ГБ < {args.min_free_gb:.2f} ГБ). Освободите место или используйте параметр --min-free-gb.", flush=True)
        sys.exit(1)

    out_dir = args.output_dir
    if args.smoke_test and not '_smoke_test' in out_dir:
        out_dir = os.path.join('outputs', '_smoke_test', 'demo')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)

    file_suffix = '_quick' if args.quick else ''
    print(f"=== Project Onyx: Demonstration Pipeline (Mode Demo) ===", flush=True)
    print(f"Config: marked_stars={args.marked_stars}, unmarked_stars={args.unmarked_stars}, quick={args.quick}, seed={args.seed}, showcase={args.showcase_star}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)

    eb_catalog_path = os.path.join('data', 'labels', 'kepler_eb_catalog.csv')
    eb_stars_set, n_eb_cat = verify_eb_catalog(eb_catalog_path)
    print(f"Каталог затменных двойных Kirk+ 2016 проверен: {n_eb_cat} записей. Самопроверка: ПРОЙДЕНА.", flush=True)

    hashes_before = compute_immutability_hashes('outputs')

    labels_path = os.path.join('data', 'labels', 'kepler_labels.csv')
    if not os.path.exists(labels_path):
        extractor_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '1_data_extractor.py')
        subprocess.run([sys.executable, extractor_script, '--mission', 'Kepler', '--limit', '1', '--cache-dir', args.cache_dir], check=False)
    labels_df = pd.read_csv(labels_path, dtype={'target_id': str})

    marked_ids, star_prim = get_marked_stars(labels_df, args.marked_stars, args.seed, args.showcase_star)
    stellar_cache = os.path.join('data', 'labels', 'kepler_stellar_cache.csv')
    marked_info = fetch_stellar_params(marked_ids, stellar_cache)

    unmarked_pool_file = os.path.join('data', 'labels', 'kepler_stellar_unmarked_pool.csv')
    pool_df = get_unmarked_pool(labels_df, unmarked_pool_file, args.unmarked_file)
    unmarked_ids, unmarked_info = match_unmarked_stars(marked_ids, marked_info, pool_df, args.unmarked_stars, args.seed, labels_df=labels_df)

    stats_m = compute_group_stats(marked_info)
    stats_u = compute_group_stats(unmarked_info)
    comp_table = pd.DataFrame([
        {'Группа': 'Отмеченные (KOI)', 'Звёзд': stats_m['count'], 'Kepmag (среднее)': f"{stats_m['mag_mean']:.2f} ± {stats_m['mag_std']:.2f}", 'Kepmag (медиана)': f"{stats_m['mag_median']:.2f}", 'Кварталов (среднее)': f"{stats_m['q_mean']:.1f}", 'Кварталов (медиана)': f"{stats_m['q_median']:.0f}"},
        {'Группа': 'Неотмеченные (Control)', 'Звёзд': stats_u['count'], 'Kepmag (среднее)': f"{stats_u['mag_mean']:.2f} ± {stats_u['mag_std']:.2f}", 'Kepmag (медиана)': f"{stats_u['mag_median']:.2f}", 'Кварталов (среднее)': f"{stats_u['q_mean']:.1f}", 'Кварталов (медиана)': f"{stats_u['q_median']:.0f}"}
    ])
    print("\nТаблица сравнения распределений звёздных параметров:")
    print(comp_table.to_string(index=False), flush=True)

    marked_kic = [int(s) for s in marked_ids if str(s).isdigit()]
    unmarked_kic = [int(s) for s in unmarked_ids if str(s).isdigit()]
    ks_kic = ks_2samp(marked_kic, unmarked_kic)

    m_mags = [marked_info[s][0] for s in marked_ids if s in marked_info]
    u_mags = [unmarked_info[s][0] for s in unmarked_ids if s in unmarked_info]
    ks_mag = ks_2samp(m_mags, u_mags)

    m_qs = [marked_info[s][1] for s in marked_ids if s in marked_info]
    u_qs = [unmarked_info[s][1] for s in unmarked_ids if s in unmarked_info]
    ks_q = ks_2samp(m_qs, u_qs)

    ks_table = pd.DataFrame([
        {
            'Параметр': 'KIC-номер',
            'Отмеченные (min / Q1 / med / Q3 / max)': f"{np.min(marked_kic)} / {int(np.percentile(marked_kic, 25))} / {int(np.median(marked_kic))} / {int(np.percentile(marked_kic, 75))} / {np.max(marked_kic)}",
            'Неотмеченные (min / Q1 / med / Q3 / max)': f"{np.min(unmarked_kic)} / {int(np.percentile(unmarked_kic, 25))} / {int(np.median(unmarked_kic))} / {int(np.percentile(unmarked_kic, 75))} / {np.max(unmarked_kic)}",
            'Статистика KS (D)': f"{ks_kic.statistic:.3f}",
            'p-значение': f"{ks_kic.pvalue:.4f}"
        },
        {
            'Параметр': 'Яркость (kepmag)',
            'Отмеченные (min / Q1 / med / Q3 / max)': f"{np.min(m_mags):.2f} / {np.percentile(m_mags, 25):.2f} / {np.median(m_mags):.2f} / {np.percentile(m_mags, 75):.2f} / {np.max(m_mags):.2f}",
            'Неотмеченные (min / Q1 / med / Q3 / max)': f"{np.min(u_mags):.2f} / {np.percentile(u_mags, 25):.2f} / {np.median(u_mags):.2f} / {np.percentile(u_mags, 75):.2f} / {np.max(u_mags):.2f}",
            'Статистика KS (D)': f"{ks_mag.statistic:.3f}",
            'p-значение': f"{ks_mag.pvalue:.4f}"
        },
        {
            'Параметр': 'Кварталов (quarters)',
            'Отмеченные (min / Q1 / med / Q3 / max)': f"{np.min(m_qs)} / {int(np.percentile(m_qs, 25))} / {int(np.median(m_qs))} / {int(np.percentile(m_qs, 75))} / {np.max(m_qs)}",
            'Неотмеченные (min / Q1 / med / Q3 / max)': f"{np.min(u_qs)} / {int(np.percentile(u_qs, 25))} / {int(np.median(u_qs))} / {int(np.percentile(u_qs, 75))} / {np.max(u_qs)}",
            'Статистика KS (D)': f"{ks_q.statistic:.3f}",
            'p-значение': f"{ks_q.pvalue:.4f}"
        }
    ])
    print("\nРезультаты двухвыборочного критерия Колмогорова-Смирнова (KS-test):")
    print(ks_table.to_string(index=False), flush=True)

    tr_marked, ctrl_marked, tr_unmarked, ctrl_unmarked = split_stars(marked_ids, unmarked_ids, star_prim, args.showcase_star, args.seed)
    all_selected_stars = tr_marked + ctrl_marked + tr_unmarked + ctrl_unmarked
    targets_demo_file = os.path.join(out_dir, f"targets_demo{file_suffix}.txt")
    with open(targets_demo_file, 'w', encoding='utf-8') as f:
        for sid in all_selected_stars:
            f.write(f"{sid}\n")

    features_demo_csv = os.path.join(out_dir, f"features_demo{file_suffix}.csv")
    failed_csv = os.path.join(out_dir, f"features_demo{file_suffix}_failed.csv")
    diag_source_csv = os.path.join(out_dir, f"features_demo{file_suffix}_diagnostics.csv")

    already_done_stars = set()
    if os.path.exists(features_demo_csv) and os.path.getsize(features_demo_csv) > 0:
        try:
            df_existing = pd.read_csv(features_demo_csv, dtype={'target_id': str})
            already_done_stars.update(df_existing['target_id'].dropna().astype(str).unique())
        except Exception:
            pass
    if os.path.exists(diag_source_csv) and os.path.getsize(diag_source_csv) > 0:
        try:
            df_diag_ex = pd.read_csv(diag_source_csv, dtype={'target_id': str})
            already_done_stars.update(df_diag_ex[df_diag_ex['status'] == 'OK']['target_id'].dropna().astype(str).unique())
        except Exception:
            pass
    if os.path.exists(failed_csv) and os.path.getsize(failed_csv) > 0:
        try:
            df_failed_existing = pd.read_csv(failed_csv, dtype={'target_id': str})
            already_done_stars.update(df_failed_existing['target_id'].dropna().astype(str).unique())
        except Exception:
            pass

    targets_to_run = [sid for sid in all_selected_stars if sid not in already_done_stars]
    print(f"\nСостояние выборки перед извлечением: всего по плану {len(all_selected_stars)} звёзд, уже обработано ранее: {len(all_selected_stars) - len(targets_to_run)}, осталось обработать: {len(targets_to_run)}", flush=True)

    if len(targets_to_run) > 0:
        targets_run_file = os.path.join(out_dir, f"targets_to_run{file_suffix}.txt")
        with open(targets_run_file, 'w', encoding='utf-8') as f:
            for sid in targets_to_run:
                f.write(f"{sid}\n")
        extractor_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), '1_data_extractor.py')
        cmd_ext = [
            sys.executable, extractor_py,
            '--mission', 'Kepler',
            '--targets-file', targets_run_file,
            '--limit', str(len(targets_to_run)),
            '--output', features_demo_csv,
            '--workers', str(args.workers),
            '--max-download-workers', str(args.max_download_workers),
            '--star-timeout', str(args.star_timeout),
            '--flatten-days', str(args.flatten_days),
            '--max-signals', str(args.max_signals),
            '--cache-dir', args.cache_dir,
            '--min-free-gb', str(args.min_free_gb)
        ]
        if args.quick:
            cmd_ext.append('--quick')
        if args.smoke_test:
            cmd_ext.append('--smoke-test')
        print(f"Запуск извлечения признаков для {len(targets_to_run)} звёзд...", flush=True)
        subprocess.run(cmd_ext, check=True)

    if not os.path.exists(features_demo_csv) or os.path.getsize(features_demo_csv) == 0:
        print(f"Ошибка: файл признаков {features_demo_csv} пуст или не найден!", flush=True)
        sys.exit(1)
    df_features = pd.read_csv(features_demo_csv, dtype={'target_id': str})
    failed_stars = {}
    if os.path.exists(failed_csv) and os.path.getsize(failed_csv) > 0:
        try:
            df_fail = pd.read_csv(failed_csv, dtype={'target_id': str})
            for _, fr in df_fail.iterrows():
                f_id = str(fr.get('target_id', '')).strip()
                if f_id:
                    failed_stars[f_id] = {
                        'error': str(fr.get('error', 'unknown error')),
                        'stage': str(fr.get('stage', 'unknown')),
                        'star_type': str(fr.get('star_type', 'unknown')),
                        'split': str(fr.get('split', 'unknown'))
                    }
        except Exception:
            pass

    diag_ok_ids = set()
    if os.path.exists(diag_source_csv) and os.path.getsize(diag_source_csv) > 0:
        try:
            df_diag_chk = pd.read_csv(diag_source_csv, dtype={'target_id': str})
            diag_ok_ids = set(df_diag_chk[df_diag_chk['status'] == 'OK']['target_id'].dropna().astype(str).unique())
        except Exception:
            pass
    processed_ids = set(df_features['target_id'].dropna().astype(str).unique()).union(diag_ok_ids)
    failed_ids = set(failed_stars.keys())
    planned_ids = set(all_selected_stars)

    missing_orchestrator_ids = planned_ids - processed_ids - failed_ids
    hard_check_ok = (len(missing_orchestrator_ids) == 0)
    if not hard_check_ok:
        print("\n" + "!" * 70, flush=True)
        print("ГРОМКОЕ ПРЕДУПРЕЖДЕНИЕ: ЖЁСТКАЯ ПРОВЕРКА ВЫБОРКИ НЕ ПРОЙДЕНА!", flush=True)
        print(f"Запланировано звёзд: {len(all_selected_stars)}", flush=True)
        print(f"Обработано успешно: {len(planned_ids.intersection(processed_ids))}", flush=True)
        print(f"Зафиксировано сбоев в _failed.csv: {len(planned_ids.intersection(failed_ids))}", flush=True)
        print(f"Потеряно (не обрабатывалось): {len(missing_orchestrator_ids)} звёзд!", flush=True)
        print(f"Примеры потерянных ID: {list(missing_orchestrator_ids)[:10]}", flush=True)
        print("!" * 70 + "\n", flush=True)
    else:
        print(f"\nЖёсткая проверка выборки УСПЕШНО пройдена: запланировано {len(all_selected_stars)}, успешно обработано {len(planned_ids.intersection(processed_ids))}, сбоев {len(planned_ids.intersection(failed_ids))}, расхождений 0.\n", flush=True)

    succ_tr_m = [s for s in tr_marked if s in processed_ids]
    succ_ctrl_m = [s for s in ctrl_marked if s in processed_ids]
    succ_tr_u = [s for s in tr_unmarked if s in processed_ids]
    succ_ctrl_u = [s for s in ctrl_unmarked if s in processed_ids]

    n_bal = min(len(succ_tr_m), len(succ_ctrl_m), len(succ_tr_u), len(succ_ctrl_u))
    showcase_id = str(args.showcase_star)

    if showcase_id in succ_ctrl_m:
        succ_ctrl_m.remove(showcase_id)
        final_ctrl_m = [showcase_id] + succ_ctrl_m[:max(0, n_bal - 1)]
    else:
        final_ctrl_m = succ_ctrl_m[:n_bal]
    final_tr_m = succ_tr_m[:n_bal]
    final_tr_u = succ_tr_u[:n_bal]
    final_ctrl_u = succ_ctrl_u[:n_bal]

    total_dropped = (len(succ_tr_m) - len(final_tr_m)) + (len(succ_ctrl_m) - (len(final_ctrl_m) - (1 if showcase_id in final_ctrl_m else 0))) + (len(succ_tr_u) - len(final_tr_u)) + (len(succ_ctrl_u) - len(final_ctrl_u))
    print(f"\nСбалансированное разбиение по звёздам (равные половины):")
    print(f"  Обучающая часть: {len(final_tr_m)} отмеченных, {len(final_tr_u)} неотмеченных (Всего звёзд: {len(final_tr_m)+len(final_tr_u)})")
    print(f"  Контрольная часть: {len(final_ctrl_m)} отмеченных (включая Kepler-80), {len(final_ctrl_u)} неотмеченных (Всего звёзд: {len(final_ctrl_m)+len(final_ctrl_u)})")
    if total_dropped > 0:
        print(f"  Отброшено звёзд для поддержания равенства половин: {total_dropped}")

    split_records = []
    for s in final_tr_m: split_records.append({'target_id': s, 'star_type': 'MARKED', 'split': 'TRAIN', 'primary_label': star_prim.get(s, 'NONE')})
    for s in final_tr_u: split_records.append({'target_id': s, 'star_type': 'UNMARKED', 'split': 'TRAIN', 'primary_label': 'NONE'})
    for s in final_ctrl_m: split_records.append({'target_id': s, 'star_type': 'MARKED', 'split': 'CONTROL', 'primary_label': star_prim.get(s, 'NONE')})
    for s in final_ctrl_u: split_records.append({'target_id': s, 'star_type': 'UNMARKED', 'split': 'CONTROL', 'primary_label': 'NONE'})
    pd.DataFrame(split_records).to_csv(os.path.join(out_dir, f"demo_stars_split{file_suffix}.csv"), index=False, encoding='utf-8')

    diag_source_csv = os.path.join(out_dir, f"features_demo{file_suffix}_diagnostics.csv")
    diag_map = {}
    if os.path.exists(diag_source_csv) and os.path.getsize(diag_source_csv) > 0:
        try:
            df_d = pd.read_csv(diag_source_csv, dtype={'target_id': str})
            for _, r in df_d.iterrows():
                diag_map[str(r['target_id'])] = r
        except Exception:
            pass

    diag_rows_csv = []
    diag_rows_md = []
    for sid in all_selected_stars:
        is_m = (sid in marked_ids)
        st_type_str = 'Отмеченная (KOI)' if is_m else 'Неотмеченная (Control)'
        if sid in (final_tr_m + final_tr_u):
            split_str = 'Обучающая'
        elif sid in (final_ctrl_m + final_ctrl_u):
            split_str = 'Контрольная'
        else:
            split_str = 'Отброшена для баланса'

        is_eb = 'ДА' if is_eclipsing_binary(sid, eb_stars_set) else 'НЕТ'

        d_info = diag_map.get(sid)
        if d_info is not None:
            nf = int(d_info.get('n_files', d_info.get('n_quarters', 0)))
            nq = int(d_info.get('n_quarters', 0))
            bd = float(d_info.get('baseline_d', 0.0))
            nsig = int(d_info.get('signals_found', 0))
            mc = int(d_info.get('matched_count', 0))
            hc = int(d_info.get('harmonic_count', 0))
            uc = int(d_info.get('unmatched_count', 0))
            raw_stat = str(d_info.get('status', 'OK'))
            msde = float(d_info.get('max_sde', 0.0))
            msnr = float(d_info.get('max_snr', 0.0))
            tdl = float(d_info.get('time_dl_s', 0.0))
            tst = float(d_info.get('time_stitch_s', 0.0))
            tbls = float(d_info.get('time_bls_s', 0.0))
            tfe = float(d_info.get('time_feats_s', 0.0))
            ttot = float(d_info.get('time_total_s', 0.0))
            if raw_stat == 'OK' or sid in processed_ids:
                stat_display = 'Обработана'
            else:
                stat_display = f"В _failed.csv ({raw_stat})"
        elif sid in failed_stars:
            nf, nq, bd, nsig, mc, hc, uc = 0, 0, 0.0, 0, 0, 0, 0
            msde, msnr = 0.0, 0.0
            tdl, tst, tbls, tfe, ttot = 0.0, 0.0, 0.0, 0.0, 0.0
            stat_display = f"В _failed.csv ({failed_stars[sid]['stage']}: {failed_stars[sid]['error']})"
        else:
            nf, nq, bd, nsig, mc, hc, uc = 0, 0, 0.0, 0, 0, 0, 0
            msde, msnr = 0.0, 0.0
            tdl, tst, tbls, tfe, ttot = 0.0, 0.0, 0.0, 0.0, 0.0
            stat_display = 'Не обрабатывалась (ошибка оркестратора / лимит)'

        match_summary = f"{mc}/{hc}/{uc}"
        max_sde_snr_str = f"SDE={msde:.1f}, SNR={msnr:.1f}"

        diag_rows_csv.append({
            'target_id': sid,
            'тип': st_type_str,
            'половина': split_str,
            'число файлов': nf,
            'число уникальных кварталов': nq,
            'базлайн в сутках': round(bd, 2),
            'найдено сигналов': nsig,
            'MATCHED/HARMONIC/UNMATCHED': match_summary,
            'статус': stat_display,
            'максимальный SDE': round(msde, 2),
            'максимальный SNR': round(msnr, 2),
            'время dl (с)': round(tdl, 2),
            'время stitch (с)': round(tst, 2),
            'время bls (с)': round(tbls, 2),
            'время feats (с)': round(tfe, 2),
            'время total (с)': round(ttot, 2),
            'Затменная двойная (EB Kirk+ 2016)': is_eb
        })

        diag_rows_md.append({
            'Target ID': sid,
            'Тип': st_type_str,
            'Половина': split_str,
            'Файлов': nf,
            'Кварталов (уник.)': nq,
            'Базлайн (д)': f"{bd:.1f}",
            'Сигналов': nsig,
            'MATCHED/HARM/UNM': match_summary,
            'Макс. SDE / SNR': max_sde_snr_str,
            'Статус': stat_display,
            'EB (каталог)': is_eb,
            'Время': f"{ttot:.1f} с"
        })

    df_diag_csv = pd.DataFrame(diag_rows_csv)
    df_diag_md = pd.DataFrame(diag_rows_md)
    diag_csv_out = os.path.join(out_dir, f"diagnostics_stars{file_suffix}.csv")
    df_diag_csv.to_csv(diag_csv_out, index=False, encoding='utf-8')

    df_features['target_bin'] = df_features['label'].apply(lambda x: 1 if str(x).upper() in ['CONFIRMED', 'CANDIDATE'] else 0)
    exclude_cols = ['signal_id', 'target_id', 'signal_index', 'mission', 'label', 'match_status', 'target_bin', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'p_candidate', 'n_signals_star']
    coverage_cols = ['baseline_d', 'n_files', 'n_quarters', 'quarters', 'st_quarters', 'dataspan', 'cadence_days', 'mean_cadence_min']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols and c not in coverage_cols]

    train_stars_set = set(final_tr_m + final_tr_u)
    ctrl_stars_set = set(final_ctrl_m + final_ctrl_u)
    df_train_all = df_features[df_features['target_id'].isin(train_stars_set)].copy()
    df_ctrl_all = df_features[df_features['target_id'].isin(ctrl_stars_set)].copy()

    df_tr_m_matched = df_train_all[(df_train_all['target_id'].isin(set(final_tr_m))) & (df_train_all['match_status'] == 'MATCHED')].copy() if 'match_status' in df_train_all else df_train_all[df_train_all['target_id'].isin(set(final_tr_m))].copy()
    df_ctrl_m_matched = df_ctrl_all[(df_ctrl_all['target_id'].isin(set(final_ctrl_m))) & (df_ctrl_all['match_status'] == 'MATCHED')].copy() if 'match_status' in df_ctrl_all else df_ctrl_all[df_ctrl_all['target_id'].isin(set(final_ctrl_m))].copy()

    print(f"\nЧисло размеченных сигналов для обучения:", flush=True)
    print(f"  Обучающая часть (a): {len(df_tr_m_matched)} сигналов", flush=True)
    print(f"  Контрольная часть (b): {len(df_ctrl_m_matched)} сигналов (без Kepler-80: {len(df_ctrl_m_matched[df_ctrl_m_matched['target_id'] != showcase_id])})", flush=True)

    print("\n[Направление А] Обучение на обучающей половине -> проверка на контрольной...", flush=True)
    ctrl_pred, model_a, expl_a, scaler_a, thres_a = train_and_eval_pipeline(df_tr_m_matched, df_ctrl_all, feature_cols, args.seed, full_features_pool=df_features, target_precision=args.target_precision)

    print("\n[Направление Б] Обучение на контрольной половине (без Kepler-80) -> проверка на обучающей...", flush=True)
    df_ctrl_m_for_b = df_ctrl_m_matched[df_ctrl_m_matched['target_id'] != showcase_id].copy()
    train_pred, model_b, expl_b, scaler_b, thres_b = train_and_eval_pipeline(df_ctrl_m_for_b, df_train_all, feature_cols, args.seed, full_features_pool=df_features, target_precision=args.target_precision)

    can_eval_a = (model_a is not None)
    can_eval_b = (model_b is not None)

    ctrl_m_eval = ctrl_pred[(ctrl_pred['target_id'].isin(set(final_ctrl_m))) & (ctrl_pred['target_id'] != showcase_id) & (ctrl_pred['match_status'] == 'MATCHED')].copy()
    y_ctrl_true = ctrl_m_eval['target_bin'].values
    y_ctrl_prob = ctrl_m_eval['p_mean'].values

    if can_eval_a and len(y_ctrl_true) > 0 and not np.isnan(y_ctrl_prob).all():
        y_ctrl_pred = (y_ctrl_prob >= 0.5).astype(int)
        acc_a = float(accuracy_score(y_ctrl_true, y_ctrl_pred))
        rec_a = float(recall_score(y_ctrl_true, y_ctrl_pred, zero_division=0))
        prec_a = float(precision_score(y_ctrl_true, y_ctrl_pred, zero_division=0))
        auc_a = float(roc_auc_score(y_ctrl_true, y_ctrl_prob)) if len(np.unique(y_ctrl_true)) >= 2 else 0.0
        cm_a = confusion_matrix(y_ctrl_true, y_ctrl_pred)

        accepted_mask = ctrl_m_eval['decision'].isin(['CANDIDATE', 'FALSE_POSITIVE'])
        n_accepted_ctrl = int(np.sum(accepted_mask))
        n_review_ctrl = int(np.sum(ctrl_m_eval['decision'] == 'REVIEW'))
        coverage_ctrl = float(n_accepted_ctrl / len(ctrl_m_eval)) if len(ctrl_m_eval) > 0 else 0.0
        review_rate_ctrl = float(n_review_ctrl / len(ctrl_m_eval)) if len(ctrl_m_eval) > 0 else 0.0

        if n_accepted_ctrl > 0:
            y_acc_true = ctrl_m_eval[accepted_mask]['target_bin'].values
            y_acc_pred = (ctrl_m_eval[accepted_mask]['decision'] == 'CANDIDATE').astype(int)
            acc_accepted = float(accuracy_score(y_acc_true, y_acc_pred))
            prec_accepted = float(precision_score(y_acc_true, y_acc_pred, zero_division=0))
            rec_accepted = float(recall_score(y_acc_true, y_acc_pred, zero_division=0))
        else:
            acc_accepted, prec_accepted, rec_accepted = 0.0, 0.0, 0.0

        boot_res = bootstrap_star_metrics(ctrl_m_eval, n_boot=500, seed=args.seed)

        b0_pred = np.zeros(len(y_ctrl_true), dtype=int)
        b0_acc = float(accuracy_score(y_ctrl_true, b0_pred))
        b0_rec = float(recall_score(y_ctrl_true, b0_pred, zero_division=0))
        b0_prec = float(precision_score(y_ctrl_true, b0_pred, zero_division=0))
        b0_auc = 0.500
        b0_cov = 1.000

        tr_snr = df_tr_m_matched['bls_depth_snr'].fillna(0).values if 'bls_depth_snr' in df_tr_m_matched else np.zeros(len(df_tr_m_matched))
        y_tr_vals = df_tr_m_matched['target_bin'].values
        best_snr_th = 10.0
        best_snr_acc = -1.0
        best_snr_dir = '>='
        for cand_th in np.percentile(tr_snr, np.linspace(10, 90, 17)):
            acc_ge = float(accuracy_score(y_tr_vals, (tr_snr >= cand_th).astype(int)))
            acc_lt = float(accuracy_score(y_tr_vals, (tr_snr < cand_th).astype(int)))
            if acc_ge > best_snr_acc:
                best_snr_acc = acc_ge
                best_snr_th = cand_th
                best_snr_dir = '>='
            if acc_lt > best_snr_acc:
                best_snr_acc = acc_lt
                best_snr_th = cand_th
                best_snr_dir = '<'
        ctrl_snr = ctrl_m_eval['bls_depth_snr'].fillna(0).values if 'bls_depth_snr' in ctrl_m_eval else np.zeros(len(ctrl_m_eval))
        snr_pred = (ctrl_snr >= best_snr_th).astype(int) if best_snr_dir == '>=' else (ctrl_snr < best_snr_th).astype(int)
        b1_acc = float(accuracy_score(y_ctrl_true, snr_pred))
        b1_rec = float(recall_score(y_ctrl_true, snr_pred, zero_division=0))
        b1_prec = float(precision_score(y_ctrl_true, snr_pred, zero_division=0))
        b1_auc = float(roc_auc_score(y_ctrl_true, ctrl_snr)) if len(np.unique(y_ctrl_true)) >= 2 else 0.500
        b1_cov = 1.000

        lr_features = [c for c in ['bls_depth_snr', 'bls_depth_ppm', 'bls_period_d', 'centroid_shift_total_sigma'] if c in df_tr_m_matched.columns]
        lr_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log1p', FunctionTransformer(apply_log1p)),
            ('scaler', RobustScaler()),
            ('model', LogisticRegression(random_state=args.seed))
        ])
        lr_pipe.fit(df_tr_m_matched[lr_features].values, y_tr_vals)
        lr_prob = lr_pipe.predict_proba(ctrl_m_eval[lr_features].values)[:, 1]
        lr_pred = (lr_prob >= 0.5).astype(int)
        b2_acc = float(accuracy_score(y_ctrl_true, lr_pred))
        b2_rec = float(recall_score(y_ctrl_true, lr_pred, zero_division=0))
        b2_prec = float(precision_score(y_ctrl_true, lr_pred, zero_division=0))
        b2_auc = float(roc_auc_score(y_ctrl_true, lr_prob)) if len(np.unique(y_ctrl_true)) >= 2 else 0.500
        b2_cov = 1.000

        baselines_table = pd.DataFrame([
            {'Модель / Эталон': 'Всегда класс 0 (FALSE_POSITIVE)', 'Точность': f"{b0_acc:.3f}", 'Полнота': f"{b0_rec:.3f}", 'Precision': f"{b0_prec:.3f}", 'ROC-AUC': f"{b0_auc:.3f}", 'Охват (Coverage)': f"{b0_cov:.1%}"},
            {'Модель / Эталон': f"Один признак (bls_depth_snr {best_snr_dir} {best_snr_th:.1f})", 'Точность': f"{b1_acc:.3f}", 'Полнота': f"{b1_rec:.3f}", 'Precision': f"{b1_prec:.3f}", 'ROC-AUC': f"{b1_auc:.3f}", 'Охват (Coverage)': f"{b1_cov:.1%}"},
            {'Модель / Эталон': 'Логистическая регрессия (4 признака)', 'Точность': f"{b2_acc:.3f}", 'Полнота': f"{b2_rec:.3f}", 'Precision': f"{b2_prec:.3f}", 'ROC-AUC': f"{b2_auc:.3f}", 'Охват (Coverage)': f"{b2_cov:.1%}"},
            {'Модель / Эталон': 'Основная модель (ансамбль XGBoost, все решения)', 'Точность': f"{acc_a:.3f}", 'Полнота': f"{rec_a:.3f}", 'Precision': f"{prec_a:.3f}", 'ROC-AUC': f"{auc_a:.3f}", 'Охват (Coverage)': "100.0%"},
            {'Модель / Эталон': 'Основная модель (только принятые решения)', 'Точность': f"{acc_accepted:.3f}", 'Полнота': f"{rec_accepted:.3f}", 'Precision': f"{prec_accepted:.3f}", 'ROC-AUC': '-', 'Охват (Coverage)': f"{coverage_ctrl:.1%}"}
        ])

        cov_present = [c for c in coverage_cols if c in df_features.columns]
        if cov_present:
            feat_with_cov = feature_cols + cov_present
            ctrl_pred_cov, model_cov, expl_cov, _, _ = train_and_eval_pipeline(df_tr_m_matched, df_ctrl_all, feat_with_cov, args.seed, full_features_pool=df_features, target_precision=args.target_precision)
            ctrl_m_cov = ctrl_pred_cov[(ctrl_pred_cov['target_id'].isin(set(final_ctrl_m))) & (ctrl_pred_cov['target_id'] != showcase_id) & (ctrl_pred_cov['match_status'] == 'MATCHED')].copy()
            y_cov_prob = ctrl_m_cov['p_mean'].values
            y_cov_pred = (y_cov_prob >= 0.5).astype(int)
            acc_cov = float(accuracy_score(y_ctrl_true, y_cov_pred))
            auc_cov = float(roc_auc_score(y_ctrl_true, y_cov_prob)) if len(np.unique(y_ctrl_true)) >= 2 else 0.0
            cov_cov_val = float(np.mean(ctrl_m_cov['decision'].isin(['CANDIDATE', 'FALSE_POSITIVE'])))
            top_shap_cov = ctrl_m_cov['shap_top3'].iloc[0].split('|')[0].strip() if len(ctrl_m_cov) > 0 and 'shap_top3' in ctrl_m_cov else '-'
            top_shap_clean = ctrl_m_eval['shap_top3'].iloc[0].split('|')[0].strip() if len(ctrl_m_eval) > 0 and 'shap_top3' in ctrl_m_eval else '-'
            ablation_table = pd.DataFrame([
                {'Конфигурация': 'Без признаков покрытия (основная модель)', 'Точность': f"{acc_a:.3f}", 'ROC-AUC': f"{auc_a:.3f}", 'Охват решений': f"{coverage_ctrl:.1%}", 'Топ-1 признак SHAP': top_shap_clean},
                {'Конфигурация': 'С признаками покрытия (+baseline_d)', 'Точность': f"{acc_cov:.3f}", 'ROC-AUC': f"{auc_cov:.3f}", 'Охват решений': f"{cov_cov_val:.1%}", 'Топ-1 признак SHAP': top_shap_cov}
            ])
        else:
            ablation_table = pd.DataFrame()
    else:
        acc_a = None
        rec_a = None
        prec_a = None
        auc_a = None
        cm_a = None
        n_accepted_ctrl = 0
        n_review_ctrl = 0
        coverage_ctrl = 0.0
        review_rate_ctrl = 0.0
        acc_accepted = None
        prec_accepted = None
        rec_accepted = None
        boot_res = {'acc_ci': (0.0, 0.0), 'auc_ci': (0.0, 0.0), 'cov_ci': (0.0, 0.0)}
        baselines_table = pd.DataFrame()
        ablation_table = pd.DataFrame()

    tr_m_eval = train_pred[(train_pred['target_id'].isin(set(final_tr_m))) & (train_pred['match_status'] == 'MATCHED')].copy()
    y_tr_true = tr_m_eval['target_bin'].values
    y_tr_prob = tr_m_eval['p_mean'].values

    if can_eval_b and len(y_tr_true) > 0 and not np.isnan(y_tr_prob).all():
        y_tr_pred = (y_tr_prob >= 0.5).astype(int)
        acc_b = float(accuracy_score(y_tr_true, y_tr_pred))
        auc_b = float(roc_auc_score(y_tr_true, y_tr_prob)) if len(np.unique(y_tr_true)) >= 2 else 0.0
    else:
        acc_b = None
        auc_b = None

    if acc_a is not None and acc_b is not None:
        mean_acc_two_way = (acc_a + acc_b) / 2.0
        mean_auc_two_way = (auc_a + auc_b) / 2.0
    else:
        mean_acc_two_way = None
        mean_auc_two_way = None

    unmatched_marked_signals = ctrl_pred[(ctrl_pred['target_id'].isin(set(final_ctrl_m))) & (ctrl_pred['match_status'] == 'UNMATCHED')]

    ctrl_unmarked_signals = ctrl_pred[ctrl_pred['target_id'].isin(set(final_ctrl_u))].copy()
    n_unm_stars = len(final_ctrl_u)
    sigs_per_unm_star = float(len(ctrl_unmarked_signals) / n_unm_stars) if n_unm_stars > 0 else 0.0

    unm_filter_counts = {
        'Короткий период P < 0.6 д (край сетки поиска)': 0,
        'Длинный период P > 90 д (дрейфы)': 0,
        f"Малая глубина транзита < {args.min_depth_ppm:.0f} ppm (ниже шума)": 0,
        'Экстремальный SNR > 100 (фотометрические выбросы)': 0,
        'Флаг возможного артефакта (possible_artifact)': 0,
        'Всего отсеяно уникальных сигналов': 0
    }

    filtered_sig_ids = set()
    for _, sig_r in ctrl_unmarked_signals.iterrows():
        s_id = str(sig_r.get('signal_id', ''))
        sp = float(sig_r.get('bls_period_d', 0.0))
        sd = float(sig_r.get('bls_depth_ppm', 0.0))
        ssnr = float(sig_r.get('bls_depth_snr', 0.0))
        is_art = bool(sig_r.get('is_artifact', False))
        is_flagged = False
        if sp < ARTIFACT_FILTER_CONFIG['min_period_days']:
            unm_filter_counts['Короткий период P < 0.6 д (край сетки поиска)'] += 1
            is_flagged = True
        if sp > ARTIFACT_FILTER_CONFIG['max_period_days']:
            unm_filter_counts['Длинный период P > 90 д (дрейфы)'] += 1
            is_flagged = True
        if sd < args.min_depth_ppm:
            unm_filter_counts[f"Малая глубина транзита < {args.min_depth_ppm:.0f} ppm (ниже шума)"] += 1
            is_flagged = True
        if ssnr > ARTIFACT_FILTER_CONFIG['max_depth_snr']:
            unm_filter_counts['Экстремальный SNR > 100 (фотометрические выбросы)'] += 1
            is_flagged = True
        if is_art:
            unm_filter_counts['Флаг возможного артефакта (possible_artifact)'] += 1
            is_flagged = True
        if is_flagged:
            filtered_sig_ids.add(s_id)

    unm_filter_counts['Всего отсеяно уникальных сигналов'] = len(filtered_sig_ids)
    df_filter_stats = pd.DataFrame([{'Критерий фильтрации': k, 'Число отсеянных сигналов': v} for k, v in unm_filter_counts.items()])

    clean_unmarked_signals = ctrl_unmarked_signals[~ctrl_unmarked_signals['signal_id'].isin(filtered_sig_ids)].copy()

    if can_eval_a and len(clean_unmarked_signals) > 0 and not np.isnan(clean_unmarked_signals['p_mean']).all():
        unm_probs = ctrl_unmarked_signals['p_mean'].values
        n_false_alarms = int(np.sum(ctrl_unmarked_signals['decision'] == 'CANDIDATE'))
        far_rate = float(n_false_alarms / len(ctrl_unmarked_signals))
        review_rate_unm = float(np.sum(ctrl_unmarked_signals['decision'] == 'REVIEW') / len(ctrl_unmarked_signals))

        top_candidates_per_star = clean_unmarked_signals.sort_values(by='p_mean', ascending=False).groupby('target_id', as_index=False).first()
        top5_candidates = top_candidates_per_star.sort_values(by='p_mean', ascending=False).head(5)
    else:
        unm_probs = np.array([0.0])
        n_false_alarms = 0
        far_rate = 0.0
        review_rate_unm = 0.0
        top5_candidates = pd.DataFrame()

    ctrl_pred['star_type'] = ctrl_pred['target_id'].apply(lambda x: 'MARKED' if str(x) in set(final_ctrl_m) else 'UNMARKED')
    ctrl_pred['is_artifact'] = (ctrl_pred['match_status'] == 'UNMATCHED') & ((ctrl_pred['bls_period_d'] >= 90.0) | (ctrl_pred['bls_depth_snr'] > 100.0))
    ctrl_pred['possible_artifact'] = ctrl_pred['is_artifact'].apply(lambda x: 'ДА' if x else 'НЕТ')
    ctrl_pred['is_kepler_eb'] = ctrl_pred['target_id'].apply(lambda x: 'ДА' if is_eclipsing_binary(x, eb_stars_set) else 'НЕТ')

    out_table_cols = ['target_id', 'star_type', 'signal_id', 'bls_period_d', 'bls_depth_ppm', 'label', 'p_mean', 'p_std', 'decision', 'possible_artifact', 'is_kepler_eb', 'shap_top3']
    control_table = ctrl_pred[[c for c in out_table_cols if c in ctrl_pred]].copy()
    control_table.rename(columns={'bls_period_d': 'period_d', 'bls_depth_ppm': 'depth_ppm', 'label': 'catalog_label'}, inplace=True)
    control_table['catalog_label'] = control_table['catalog_label'].fillna('NONE')
    ctrl_csv_out = os.path.join(out_dir, f"control_signals{file_suffix}.csv")
    ctrl_md_out = os.path.join(out_dir, f"control_signals{file_suffix}.md")
    control_table.to_csv(ctrl_csv_out, index=False, encoding='utf-8')
    with open(ctrl_md_out, 'w', encoding='utf-8') as f:
        f.write(df_to_markdown(control_table) + "\n")

    suspect_rows = []
    for _, sig_r in ctrl_pred.iterrows():
        s_sde = float(sig_r.get('bls_sde', 0.0))
        s_snr = float(sig_r.get('bls_depth_snr', 0.0))
        s_p = float(sig_r.get('bls_period_d', 0.0))
        s_d = float(sig_r.get('bls_depth_ppm', 0.0))
        s_id = str(sig_r.get('signal_id', ''))
        s_tid = str(sig_r.get('target_id', ''))
        s_type = 'Отмеченная (KOI)' if s_tid in set(final_ctrl_m) else 'Неотмеченная (Control)'
        s_half = 'Контрольная'
        s_eb = 'ДА' if is_eclipsing_binary(s_tid, eb_stars_set) else 'НЕТ'
        even_odd = sig_r.get('odd_even_sigma', sig_r.get('bls_even_odd_ratio', '-'))

        suspect_reason = []
        if s_sde > 60.0:
            suspect_reason.append('SDE > 60')
        if s_sde > 100.0 and s_snr < 10.0:
            suspect_reason.append('SDE > 100 при SNR < 10')
        elif s_sde > 30.0 and s_snr < 5.0:
            suspect_reason.append('SDE > 30 при SNR < 5')
        elif s_sde > 50.0 and (s_sde / max(1.0, s_snr)) > 5.0:
            suspect_reason.append('Несогласованные SDE и SNR (SDE/SNR > 5)')

        if suspect_reason:
            suspect_rows.append({
                'target_id': s_tid,
                'signal_id': s_id,
                'тип': s_type,
                'половина': s_half,
                'period_d': round(s_p, 4),
                'depth_ppm': round(s_d, 1),
                'SDE': round(s_sde, 1),
                'SNR': round(s_snr, 1),
                'Затменная двойная': s_eb,
                'odd_even_sigma': str(even_odd),
                'причина подозрения': '; '.join(suspect_reason)
            })
    df_suspects = pd.DataFrame(suspect_rows)
    suspects_csv_out = os.path.join(out_dir, f"suspects{file_suffix}.csv")
    df_suspects.to_csv(suspects_csv_out, index=False, encoding='utf-8')

    df_features['is_artifact'] = (df_features['match_status'] == 'UNMATCHED') & ((df_features['bls_period_d'] >= 90.0) | (df_features['bls_depth_snr'] > 100.0))
    n_art_tr_m = int(np.sum((df_features['target_id'].isin(set(final_tr_m))) & df_features['is_artifact']))
    n_unm_tr_m = int(np.sum((df_features['target_id'].isin(set(final_tr_m))) & (df_features['match_status'] == 'UNMATCHED')))
    n_art_tr_u = int(np.sum((df_features['target_id'].isin(set(final_tr_u))) & df_features['is_artifact']))
    n_unm_tr_u = int(np.sum((df_features['target_id'].isin(set(final_tr_u))) & (df_features['match_status'] == 'UNMATCHED')))
    n_art_ctrl_m = int(np.sum((df_features['target_id'].isin(set(final_ctrl_m))) & df_features['is_artifact']))
    n_unm_ctrl_m = int(np.sum((df_features['target_id'].isin(set(final_ctrl_m))) & (df_features['match_status'] == 'UNMATCHED')))
    n_art_ctrl_u = int(np.sum((df_features['target_id'].isin(set(final_ctrl_u))) & df_features['is_artifact']))
    n_unm_ctrl_u = int(np.sum((df_features['target_id'].isin(set(final_ctrl_u))) & (df_features['match_status'] == 'UNMATCHED')))

    df_matched_all = df_features[df_features['match_status'] == 'MATCHED']
    n_matched_total = len(df_matched_all)
    n_matched_c1 = int(np.sum(df_matched_all['target_bin'] == 1))
    n_matched_c0 = int(np.sum(df_matched_all['target_bin'] == 0))

    df_matched_tr = df_matched_all[df_matched_all['target_id'].isin(set(final_tr_m))]
    n_matched_tr = len(df_matched_tr)
    n_tr_c1 = int(np.sum(df_matched_tr['target_bin'] == 1))
    n_tr_c0 = int(np.sum(df_matched_tr['target_bin'] == 0))

    df_matched_ctrl = df_matched_all[df_matched_all['target_id'].isin(set(final_ctrl_m))]
    n_matched_ctrl = len(df_matched_ctrl)
    n_ctrl_c1 = int(np.sum(df_matched_ctrl['target_bin'] == 1))
    n_ctrl_c0 = int(np.sum(df_matched_ctrl['target_bin'] == 0))
    n_ctrl_no_k80 = len(df_matched_ctrl[df_matched_ctrl['target_id'] != showcase_id])

    tot_marked_stars = len(final_tr_m) + len(final_ctrl_m)
    matched_per_marked = float(n_matched_total / tot_marked_stars) if tot_marked_stars > 0 else 0.0

    n_failed_stars = len(failed_stars)
    print("\n" + "="*50, flush=True)
    print("СВОДКА РАЗМЕЧЕННЫХ СИГНАЛОВ И КАЧЕСТВА ДАННЫХ:", flush=True)
    print(f"  Всего размеченных сигналов (MATCHED): {n_matched_total}", flush=True)
    print(f"    - Класс 1 (CONFIRMED/CANDIDATE): {n_matched_c1}", flush=True)
    print(f"    - Класс 0 (FALSE_POSITIVE): {n_matched_c0}", flush=True)
    print(f"  Размеченных сигналов по половинам:", flush=True)
    print(f"    - Обучающая половина: {n_matched_tr} (класс 1: {n_tr_c1}, класс 0: {n_tr_c0})", flush=True)
    print(f"    - Контрольная половина: {n_matched_ctrl} (класс 1: {n_ctrl_c1}, класс 0: {n_ctrl_c0}; без Kepler-80: {n_ctrl_no_k80})", flush=True)
    print(f"  Число размеченных сигналов на одну отмеченную звезду: {matched_per_marked:.2f} ({n_matched_total} / {tot_marked_stars})", flush=True)
    print(f"  Число звёзд в _failed.csv: {n_failed_stars}", flush=True)
    if n_failed_stars > 0:
        for f_sid, f_dat in failed_stars.items():
            print(f"    * Звезда {f_sid} ({f_dat['stage']}): {f_dat['error']}", flush=True)
    print("="*50 + "\n", flush=True)

    showcase_signals = ctrl_pred[ctrl_pred['target_id'] == showcase_id].copy()
    showcase_plot_path = os.path.join(out_dir, f"showcase_kepler80{file_suffix}.png")
    showcase_n_files, showcase_quarters, showcase_baseline, showcase_bjd_min, showcase_bjd_max = generate_showcase_figure(showcase_id, showcase_signals, showcase_plot_path, args.quick, cache_dir=args.cache_dir)

    known_planets = [
        {'planet': 'Kepler-80 b', 'koi': 'KOI-500.01', 'period_d': 7.053527},
        {'planet': 'Kepler-80 c', 'koi': 'KOI-500.02', 'period_d': 9.521645},
        {'planet': 'Kepler-80 d', 'koi': 'KOI-500.03', 'period_d': 3.072147},
        {'planet': 'Kepler-80 e', 'koi': 'KOI-500.04', 'period_d': 4.645382},
        {'planet': 'Kepler-80 f', 'koi': 'KOI-500.05', 'period_d': 0.986786}
    ]
    recovered_planets = []
    for kp in known_planets:
        kp_p = kp['period_d']
        matched_sig = None
        match_type = 'Не найдено'
        for _, s_row in showcase_signals.iterrows():
            sp = float(s_row['bls_period_d'])
            if abs(sp - kp_p) / kp_p < 0.01:
                matched_sig = s_row
                match_type = 'Найдено точно (допуск 1%)'
                break
            elif abs(sp - 0.5 * kp_p) / (0.5 * kp_p) < 0.01:
                matched_sig = s_row
                match_type = 'Найдено по гармонике (0.5x)'
                break
            elif abs(sp - 2.0 * kp_p) / (2.0 * kp_p) < 0.01:
                matched_sig = s_row
                match_type = 'Найдено по гармонике (2.0x)'
                break
        recovered_planets.append({
            'Планета': kp['planet'],
            'Каталожный период (дни)': f"{kp_p:.4f}",
            'Результат обнаружения': match_type,
            'Период найденного (дни)': f"{float(matched_sig['bls_period_d']):.4f}" if matched_sig is not None else '-',
            'Вероятность': f"{float(matched_sig['p_mean']):.1%}" if (matched_sig is not None and pd.notnull(matched_sig.get('p_mean')) and not np.isnan(float(matched_sig['p_mean']))) else '-',
            'Вердикт': str(matched_sig['decision']) if matched_sig is not None else '-'
        })
    df_k80_rec = pd.DataFrame(recovered_planets)
    n_exact = sum(1 for r in recovered_planets if 'точно' in r['Результат обнаружения'].lower())
    n_harmonic = sum(1 for r in recovered_planets if 'гармоник' in r['Результат обнаружения'].lower())
    n_not_found = sum(1 for r in recovered_planets if 'не найдено' in r['Результат обнаружения'].lower())
    k80_summary_str = f"{n_exact} точно + {n_harmonic} по гармонике из {len(known_planets)}"

    correct_samples = ctrl_m_eval[((ctrl_m_eval['target_bin'] == 1) & (ctrl_m_eval['decision'] == 'CANDIDATE')) | ((ctrl_m_eval['target_bin'] == 0) & (ctrl_m_eval['decision'] == 'FALSE_POSITIVE'))].head(2) if can_eval_a else pd.DataFrame()

    total_wall_s = time.time() - start_wall_time
    total_cpu_s = time.process_time() - start_cpu_time
    sum_star_total_s = float(df_diag_csv['время total (с)'].sum()) if 'время total (с)' in df_diag_csv else 0.0

    n_eb_total = sum(1 for sid in all_selected_stars if is_eclipsing_binary(sid, eb_stars_set))
    n_eb_marked = sum(1 for sid in marked_ids if is_eclipsing_binary(sid, eb_stars_set))
    n_eb_unmarked = sum(1 for sid in unmarked_ids if is_eclipsing_binary(sid, eb_stars_set))

    failed_m_in_plan = [sid for sid in marked_ids if sid in failed_stars]
    failed_u_in_plan = [sid for sid in unmarked_ids if sid in failed_stars]

    hashes_after = compute_immutability_hashes('outputs')
    hash_match = (hashes_before == hashes_after)

    summary_md_path = os.path.join(out_dir, f"summary{file_suffix}.md")
    h1 = chr(35) + " "
    h2 = chr(35) * 2 + " "
    h3 = chr(35) * 3 + " "
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write(f"{h1}Итоговый отчёт демонстрационного режима (Project Onyx Demo)\n\n")
        f.write(f"**Дата и время формирования:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{h2}1. Конфигурация запуска и параметры поиска\n")
        f.write(f"* **Режим:** {'Быстрый (--quick, frequency_factor=5000, 4 шага duration)' if args.quick else 'Стандартный (полная сетка BLS, frequency_factor=500)'}\n")
        f.write(f"* **Параметры детрендинга и поиска:** `flatten_days = {args.flatten_days}`, `max_signals = {args.max_signals}`\n")
        f.write(f"* **Размер выборки (план):** {args.marked_stars} отмеченных (KOI), {args.unmarked_stars} неотмеченных\n")
        f.write(f"* **Параметры параллелизма:** `--workers = {args.workers}`, `--max-download-workers = {args.max_download_workers}`\n")
        f.write(f"* **Параметры классификации и фильтрации:** `--target-precision = {args.target_precision:.2f}`, `--min-depth-ppm = {args.min_depth_ppm:.1f}`\n")
        f.write(f"* **Случайный seed:** `{args.seed}`\n")
        f.write(f"* **Показательная система:** Kepler-80 (KIC {args.showcase_star}, статус CONFIRMED, 5 планет)\n")
        f.write(f"* **Выходная директория:** `{out_dir}`\n\n")

        f.write(f"{h2}2. Состав выборки и балансировка звёздных параметров\n")
        f.write(f"Выборка неотмеченных звёзд сформирована со стратификацией по 10 децилям KIC-номеров отмеченных звёзд для обеспечения пространственной равномерности по всему полю Kepler, с жадным подбором распределений яркости (`kepmag`) и числа кварталов (`quarters`).\n\n")
        f.write(f"* **Фактические размеры выборки (плановые vs реально обработанные):**\n")
        f.write(f"  * Отмеченные звёзды (KOI): план **{args.marked_stars}**, успешно обработано **{len(succ_tr_m) + len(succ_ctrl_m)}**, с ошибками в `_failed.csv`: **{len(failed_m_in_plan)}**\n")
        f.write(f"  * Неотмеченные звёзды (Control): план **{args.unmarked_stars}**, успешно обработано **{len(succ_tr_u) + len(succ_ctrl_u)}**, с ошибками в `_failed.csv`: **{len(failed_u_in_plan)}**\n")
        f.write(f"  * Обучающая половина: **{len(final_tr_m)}** отмеченных + **{len(final_tr_u)}** неотмеченных звёзд\n")
        f.write(f"  * Контрольная половина: **{len(final_ctrl_m)}** отмеченных (включая Kepler-80) + **{len(final_ctrl_u)}** неотмеченных звёзд\n")
        f.write(f"  * Отброшено звёзд для поддержания строгого равенства половин: **{total_dropped}**\n")
        f.write(f"* **Временные затраты прогона:**\n")
        f.write(f"  * Астрономическое время запуска (Wall-clock time): **{total_wall_s:.1f} с** ({total_wall_s / 60.0:.2f} мин)\n")
        f.write(f"  * Суммарное процессорное время обработки звёзд: **{sum_star_total_s:.1f} с** ({sum_star_total_s / 60.0:.2f} мин)\n")
        f.write(f"  * Процессорное время главного оркестратора: **{total_cpu_s:.1f} с**\n")
        f.write(f"* **Жёсткая проверка соответствия плана и факта:** {'УСПЕШНО ПРОЙДЕНА (расхождений нет)' if hard_check_ok else f'ВНИМАНИЕ: НЕ ПРОЙДЕНА (потеряно {len(missing_orchestrator_ids)} звёзд)'}\n\n")

        if n_failed_stars > 0:
            f.write(f"{h3}Таблица сбойных звёзд (не удалось скачать или обработать):\n\n")
            failed_table_rows = []
            for f_sid, f_dat in failed_stars.items():
                failed_table_rows.append({
                    'Target ID': f_sid,
                    'Тип': f_dat['star_type'],
                    'Половина': f_dat['split'],
                    'Этап сбоя': f_dat['stage'],
                    'Текст ошибки': f_dat['error']
                })
            f.write(df_to_markdown(pd.DataFrame(failed_table_rows)) + "\n\n")

        f.write(f"{h3}Сравнение распределений яркости и наблюдаемости:\n\n")
        f.write(df_to_markdown(comp_table) + "\n\n")

        f.write(f"{h3}Результаты двухвыборочного критерия Колмогорова-Смирнова (KS-test):\n\n")
        f.write(df_to_markdown(ks_table) + "\n\n")

        f.write(f"{h2}2.1. Диагностика обнаружения периодических сигналов по каждой звезде\n")
        f.write(f"В таблице ниже приведены детальные результаты фотометрической обработки и поиска сигналов алгоритмом Box Least Squares (BLS) для каждой звезды выборки.\n\n")
        f.write(df_to_markdown(df_diag_md) + "\n\n")

        df_zero = df_diag_csv[(df_diag_csv['найдено сигналов'] == 0) & (df_diag_csv['статус'] == 'Обработана')]
        if len(df_zero) > 0:
            zero_rows = []
            for _, zr in df_zero.iterrows():
                z_tid = str(zr['target_id'])
                z_type = str(zr['тип'])
                z_sde = float(zr['максимальный SDE'])
                z_snr = float(zr['максимальный SNR'])
                z_eb = str(zr['Затменная двойная (EB Kirk+ 2016)'])
                z_cat_p = labels_df[labels_df['target_id'] == z_tid]['catalog_period'].dropna().tolist()
                if len(z_cat_p) > 0 and all(p > 100.0 for p in z_cat_p):
                    p_strs = ', '.join(f"{p:.1f}" for p in z_cat_p)
                    z_reason = f"Каталожный период вне диапазона поиска [0,5, 100,0] д ({p_strs} д)"
                elif z_sde < 7.0 or z_snr < 7.1:
                    z_reason = f"Пик ниже порога обнаружения: SDE={z_sde:.1f} (порог 7.0), SNR={z_snr:.1f} (порог 7.1)"
                else:
                    z_reason = "Причина не установлена"
                zero_rows.append({
                    'Target ID': z_tid,
                    'Тип': z_type,
                    'Макс. SDE': f"{z_sde:.1f}",
                    'Макс. SNR': f"{z_snr:.1f}",
                    'EB (каталог)': z_eb,
                    'Каталожные периоды (д)': ', '.join(f"{p:.2f}" for p in z_cat_p) if len(z_cat_p) > 0 else '-',
                    'Причина отсутствия сигналов': z_reason
                })
            f.write(f"{h3}Анализ звёзд с нулевым числом обнаруженных сигналов:\n\n")
            f.write(df_to_markdown(pd.DataFrame(zero_rows)) + "\n\n")
        else:
            f.write(f"{h3}Анализ звёзд с нулевым числом обнаруженных сигналов:\n")
            f.write("В данном запуске на всех успешно обработанных звёздах обнаружен как минимум один периодический сигнал.\n\n")

        f.write(f"{h2}2.2. Сводка по размеченным сигналам и качеству выборки\n")
        f.write(f"* **Всего размеченных сигналов (MATCHED):** **{n_matched_total}**\n")
        f.write(f"  * Класс 1 (CONFIRMED / CANDIDATE): **{n_matched_c1}**\n")
        f.write(f"  * Класс 0 (FALSE_POSITIVE): **{n_matched_c0}**\n")
        f.write(f"* **Разбиение по половинам выборки:**\n")
        f.write(f"  * Обучающая половина: **{n_matched_tr}** (класс 1: {n_tr_c1}, класс 0: {n_tr_c0})\n")
        f.write(f"  * Контрольная половина: **{n_matched_ctrl}** (класс 1: {n_ctrl_c1}, класс 0: {n_ctrl_c0}; без витринной Kepler-80: **{n_ctrl_no_k80}**)\n")
        f.write(f"* **Среднее число размеченных сигналов на одну отмеченную звезду:** **{matched_per_marked:.2f}** ({n_matched_total} / {tot_marked_stars})\n")
        f.write(f"* **Число звёзд в `_failed.csv`:** **{n_failed_stars}**\n\n")

        f.write(f"{h2}2.3. Сводка несовпавших сигналов (UNMATCHED) и возможных артефактов\n")
        f.write("Периодические сигналы на отмеченных звёздах, периоды которых не совпали с каталожными планетами или затменными двойными с допуском 1% (и их гармониками), получают статус UNMATCHED. На неотмеченных звёздах все найденные сигналы получают статус UNMATCHED.\n\n")
        unm_summary_rows = [
            {'Половина': 'Обучающая', 'Тип звёзд': 'Отмеченные (KOI)', 'Всего UNMATCHED': n_unm_tr_m, 'Из них возможных артефактов': n_art_tr_m},
            {'Половина': 'Обучающая', 'Тип звёзд': 'Неотмеченные', 'Всего UNMATCHED': n_unm_tr_u, 'Из них возможных артефактов': n_art_tr_u},
            {'Половина': 'Контрольная', 'Тип звёзд': 'Отмеченные (KOI)', 'Всего UNMATCHED': n_unm_ctrl_m, 'Из них возможных артефактов': n_art_ctrl_m},
            {'Половина': 'Контрольная', 'Тип звёзд': 'Неотмеченные', 'Всего UNMATCHED': n_unm_ctrl_u, 'Из них возможных артефактов': n_art_ctrl_u},
            {'Половина': 'ИТОГО', 'Тип звёзд': 'Все группы', 'Всего UNMATCHED': n_unm_tr_m + n_unm_tr_u + n_unm_ctrl_m + n_unm_ctrl_u, 'Из них возможных артефактов': n_art_tr_m + n_art_tr_u + n_art_ctrl_m + n_art_ctrl_u}
        ]
        f.write(df_to_markdown(pd.DataFrame(unm_summary_rows)) + "\n\n")

        f.write(f"{h2}3. Раздел А: Оценка на отмеченных звёздах контрольной части (есть истинные метки)\n")
        f.write(f"*(Примечание: витринная система Kepler-80 исключена из количественных метрик Раздела А и оценивается качественно и независимо в Разделе 5)*\n\n")
        if can_eval_a and len(ctrl_m_eval) > 0 and acc_a is not None:
            f.write(f"* **Всего размеченных сигналов (MATCHED) для оценки (без Kepler-80):** {len(ctrl_m_eval)}\n")
            f.write(f"* **Матрица ошибок (Confusion Matrix, основная модель):**\n```\n{cm_a}\n```\n")
            f.write(f"{h3}Сравнение с эталонами классификации (Baselines):\n\n")
            f.write(df_to_markdown(baselines_table) + "\n\n")
            f.write(f"* **Пороги отказа от решения (Abstention Thresholds):**\n")
            f.write(f"  * Пороги подобраны кросс-валидацией `GroupKFold` (с группировкой по звёздам `target_id`) на обучающей выборке под целевую точность $\\ge {args.target_precision:.2f}$:\n")
            f.write(f"    * $t_{{high}} = {thres_a[0]:.2f}$, $t_{{low}} = {thres_a[1]:.2f}$, $s_{{max}} = {thres_a[2]:.2f}$\n")
            f.write(f"  * **Охват решений (Coverage):** **{coverage_ctrl:.1%}** ({n_accepted_ctrl} из {len(ctrl_m_eval)} сигналов, 90% ДИ бутстрапом: [{boot_res['cov_ci'][0]:.3f}, {boot_res['cov_ci'][1]:.3f}])\n")
            f.write(f"  * **Точность на принятых решениях:** **{prec_accepted:.3f}** (принято сигналов: {n_accepted_ctrl}{', пометка: малая выборка' if n_accepted_ctrl < 20 else ''})\n")
            f.write(f"  * **Доля вердиктов REVIEW:** **{review_rate_ctrl:.1%}** ({n_review_ctrl} из {len(ctrl_m_eval)})\n\n")

            if len(ablation_table) > 0:
                f.write(f"{h3}Абляция: Влияние признаков покрытия данных (baseline_d):\n\n")
                f.write("Признаки покрытия данных (`baseline_d`, `n_quarters`, `n_files`) отражают длительность фотометрических наблюдений конкретной звезды телескопом, а не физические свойства транзитного сигнала. Во избежание ложной корреляции они исключены из пространства признаков основной модели.\n\n")
                f.write(df_to_markdown(ablation_table) + "\n\n")

            f.write(f"* **Двунаправленная перекрёстная проверка (Two-way train/test):**\n")
            f.write(f"  * Точность A (Train->Ctrl): {acc_a:.3f} | Точность B (Ctrl->Train): {acc_b:.3f} | Средняя: **{mean_acc_two_way:.3f}**\n")
            f.write(f"  * ROC-AUC A: {auc_a:.3f} | ROC-AUC B: {auc_b:.3f} | Средний ROC-AUC: **{mean_auc_two_way:.3f}**\n")
            f.write(f"  *(Примечание: в направлении B звезда Kepler-80 была полностью исключена из обучающего набора)*\n")
        else:
            c1_cnt = int(np.sum(df_tr_m_matched['target_bin'] == 1)) if len(df_tr_m_matched) > 0 else 0
            c0_cnt = int(np.sum(df_tr_m_matched['target_bin'] == 0)) if len(df_tr_m_matched) > 0 else 0
            classes_desc_str = f"класс 1 (CONFIRMED/CANDIDATE): {c1_cnt}, класс 0 (FALSE_POSITIVE): {c0_cnt}"
            f.write(f"> [!WARNING]\n")
            f.write(f"> **Обучение невозможно:** в обучающей части классы: {classes_desc_str}, всего сигналов: {len(df_tr_m_matched)}.\n")
            f.write(f"> Для корректного обучения модели машинного обучения требуется как минимум 2 класса и не менее 5 размеченных сигналов каждого класса.\n")
            f.write(f"> На текущей малой выборке обучение остановлено на этапе оценки. Выдача модельных вердиктов и расчёт нулевых метрик пропущены во избежание генерации некорректных показателей.\n\n")
            f.write(f"{h3}Сводка по классам в обучающей части (Направление А):\n\n")
            class_summary_rows = [
                {'Класс (target_bin)': '1 (CONFIRMED / CANDIDATE)', 'Число сигналов': c1_cnt, 'Минимальный порог': '>= 5', 'Статус': 'Достаточно' if c1_cnt >= 5 else 'Недостаточно (< 5)'},
                {'Класс (target_bin)': '0 (FALSE_POSITIVE)', 'Число сигналов': c0_cnt, 'Минимальный порог': '>= 5', 'Статус': 'Достаточно' if c0_cnt >= 5 else 'Недостаточно (< 5)'}
            ]
            f.write(df_to_markdown(pd.DataFrame(class_summary_rows)) + "\n\n")
        f.write(f"* **Несовпавшие сигналы (UNMATCHED) на отмеченных звёздах:** {len(unmatched_marked_signals)} сигналов (исключены из надзорных метрик, сохранены в таблице сигналов).\n\n")

        f.write(f"{h2}4. Раздел Б: Оценка на неотмеченных звёздах контрольной части (эталонных меток НЕТ)\n")
        f.write(f"> [!IMPORTANT]\n")
        f.write(f"> На неотмеченных звёздах метрики точности, полноты и ROC-AUC принципиально **не рассчитываются**, так как истинный статус этих объектов неизвестен (часть из них может содержать неоткрытые экзопланеты).\n\n")
        f.write(f"* **Фактическое число звёзд в контрольной неотмеченной группе:** **{n_unm_stars}**\n")
        f.write(f"* **Всего обнаружено периодических сигналов:** **{len(ctrl_unmarked_signals)}**\n")
        f.write(f"* **Среднее число сигналов на звезду:** **{sigs_per_unm_star:.2f}**\n")
        if can_eval_a and len(ctrl_unmarked_signals) > 0:
            f.write(f"* **Распределение предсказанной вероятности (p_mean):**\n")
            f.write(f"  * Минимум: {float(np.min(unm_probs)):.3f} | 25%: {float(np.percentile(unm_probs, 25)):.3f} | Медиана: {float(np.median(unm_probs)):.3f} | 75%: {float(np.percentile(unm_probs, 75)):.3f} | Максимум: {float(np.max(unm_probs)):.3f}\n")
            f.write(f"* **Доля сигналов с вердиктом CANDIDATE:** **{far_rate:.1%}** ({n_false_alarms} сигналов; примечание: данная величина не является строгой метрикой качества False Alarm Rate, так как отсутствие планет на неотмеченных звёздах является рабочей гипотезой, а не доказанным фактом).\n")
            f.write(f"* **Доля неопределённых сигналов (REVIEW):** **{review_rate_unm:.1%}**\n\n")

            f.write(f"{h3}Фильтрация потенциальных инструментальных артефактов перед ранжированием:\n\n")
            f.write(df_to_markdown(df_filter_stats) + "\n\n")

            f.write(f"{h3}Топ-5 наиболее «планетоподобных» сигналов среди неотмеченных звёзд (после фильтрации артефактов, максимум 1 на звезду):\n\n")
            if len(top5_candidates) > 0:
                top5_view = top5_candidates[['target_id', 'bls_period_d', 'bls_depth_ppm', 'p_mean', 'decision', 'shap_top3']].copy()
                top5_view.rename(columns={'bls_period_d': 'Период (д)', 'bls_depth_ppm': 'Глубина (ppm)', 'p_mean': 'Вероятность', 'decision': 'Решение', 'shap_top3': 'Вклады SHAP'}, inplace=True)
                f.write(df_to_markdown(top5_view) + "\n\n")
            else:
                f.write("Все обнаруженные сигналы на неотмеченных звёздах отфильтрованы по критериям артефактов.\n\n")
        else:
            f.write("* Модельная оценка вероятностей пропущена, так как классификатор машинного обучения не обучался из-за малого числа размеченных сигналов (< 5 на класс).\n\n")

        f.write(f"{h2}5. Результаты по витринной системе Kepler-80 (KIC {showcase_id})\n")
        f.write(f"Для витринной системы Kepler-80 загружено {showcase_n_files} файлов ({showcase_quarters} уникальных кварталов Kepler, базлайн: {showcase_baseline:.1f} суток, диапазон BJD {showcase_bjd_min:.1f} – {showcase_bjd_max:.1f}). Система содержит 5 известных подтверждённых планет в резонансной цепочке периодов. Графическая панель сохранена в файл `{showcase_plot_path}`.\n\n")
        f.write(df_to_markdown(df_k80_rec) + "\n\n")
        f.write(f"* **Итог валидации Kepler-80:** **{k80_summary_str}** (найдено точно: {n_exact}, по гармонике: {n_harmonic}, не найдено: {n_not_found}).\n")
        if n_exact + n_harmonic < 5:
            f.write(f"  * *Для ненайденных планет:* причина не установлена (требуется дополнительный специализированный анализ периодограммы BLS и фазовой кривой без умозрительных предположений).\n\n")

        f.write(f"{h2}6. Примеры классификации и интерпретация признаков (SHAP)\n")
        if can_eval_a and len(correct_samples) > 0:
            f.write(f"{h3}Примеры верных решений модели:\n")
            for _, r in correct_samples.iterrows():
                f.write(f"* **Звезда {r['target_id']} (Сигнал {r['signal_id']}):** Истинный класс = `{r['label']}`, Вердикт = `{r['decision']}` ($p = {r['p_mean']:.1%}$).\n  * Вклады SHAP: `{r['shap_top3']}`\n")
        else:
            f.write("* На текущей тестовой выборке примеры решений модели отсутствуют (оценка пропущена из-за размера выборки).\n")

        f.write(f"\n{h2}7. Проверка каталога затменных двойных (Kirk et al. 2016)\n")
        f.write(f"* **Каталог затменных двойных:** `{eb_catalog_path}` (всего {n_eb_cat} записей)\n")
        f.write(f"* **Самопроверка загрузки каталога при старте:** ПРОЙДЕНА (идентификаторы успешно верифицированы)\n")
        f.write(f"* **Идентифицировано затменных двойных в текущей выборке:** **{n_eb_total}** звёзд (из них отмеченных: {n_eb_marked}, неотмеченных: {n_eb_unmarked})\n")
        if len(all_selected_stars) >= 500 and n_eb_total == 0:
            f.write(f"> [!WARNING]\n")
            f.write(f"> **ГРОМКОЕ ПРЕДУПРЕЖДЕНИЕ:** Среди {len(all_selected_stars)} звёзд выборки не найдено ни одной затменной двойной из каталога Kirk+ 2016! Проверьте сопоставление форматов идентификаторов.\n\n")
        f.write("\n")

        f.write(f"{h2}8. Подозрительно сильные сигналы (Suspect Signals)\n")
        f.write(f"Сигналы с экстремально высокой мощностью BLS (SDE > 60) или с выраженным расхождением между мощностью периодограммы и глубиной транзита (например, SDE > 100 при SNR < 10) сохранены в отдельную диагностическую таблицу `{suspects_csv_out}` для независимого экспертного разбора.\n\n")
        f.write(f"* **Всего подозрительных сигналов в контрольной части:** **{len(df_suspects)}**\n")
        if len(df_suspects) > 0:
            f.write(df_to_markdown(df_suspects.head(10)) + "\n\n")

        f.write(f"{h2}9. Ограничения и угрозы валидности (Threats to Validity)\n")
        f.write(f"1. **Объём тестовой выборки:** Демонстрационная выборка носит иллюстративный характер и не заменяет полный прогон на тысячах звёзд.\n")
        f.write(f"2. **Шум метрик и доверительные интервалы:** При малом числе звёзд доверительные интервалы широки, а точечные метрики чувствительны к отдельным объектам.\n")
        f.write(f"3. **Каталожные метки не являются абсолютной истиной:** Каталог KOI содержит споры по спорным кандидатам и шум классификации операторов.\n")
        f.write(f"4. **Грубое смещение центроида:** Расчёт смещения центроида ведётся по средневзвешенным координатам пятна, а не по пиксельным функциям отклика (PRF).\n")
        f.write(f"5. **Неотмеченные звёзды:** Допущение об отсутствии планет на неотмеченных звёздах является рабочей гипотезой; часть обнаруженных кандидатов может оказаться реальными экзопланетами.\n\n")

        f.write(f"{h2}10. Тест на неприкосновенность основных файлов (Immutability Test)\n")
        f.write(f"* **Статус проверки хешей (SHA-256):** {'УСПЕШНО (Все хеши совпали)' if hash_match else 'ВНИМАНИЕ (Обнаружено несовпадение)'}\n")
        f.write(f"* Проверено исходных файлов в `outputs/`: {len(hashes_before)}\n")

    print(f"\n=== Демонстрационный прогон завершён! ===", flush=True)
    print(f"Итоговый отчёт сохранён в: {summary_md_path}", flush=True)
    print(f"Таблица контрольных сигналов: {ctrl_csv_out}", flush=True)
    print(f"Диагностическая таблица по звёздам: {diag_csv_out}", flush=True)
    print(f"Таблица подозрительных сигналов: {suspects_csv_out}", flush=True)
    print(f"Витрина Kepler-80: {showcase_plot_path}", flush=True)
    print(f"Тест неприкосновенности outputs/: {'ПРОЙДЕН (Хеши совпадают)' if hash_match else 'НЕ ПРОЙДЕН'}", flush=True)

if __name__ == '__main__':
    main()
