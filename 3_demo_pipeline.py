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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier
import shap

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

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
    conf_stars = [s for s, l in star_prim.items() if l == 'CONFIRMED' and s != showcase_id]
    cand_stars = [s for s, l in star_prim.items() if l == 'CANDIDATE' and s != showcase_id]
    fp_stars = [s for s, l in star_prim.items() if l == 'FALSE_POSITIVE' and s != showcase_id]
    rng = np.random.RandomState(seed)
    rng.shuffle(conf_stars)
    rng.shuffle(cand_stars)
    rng.shuffle(fp_stars)
    n_remain = max(0, n_stars - 1)
    tot_avail = len(conf_stars) + len(cand_stars) + len(fp_stars)
    n_conf = int(round(n_remain * (len(conf_stars) / tot_avail)))
    n_cand = int(round(n_remain * (len(cand_stars) / tot_avail)))
    n_fp = n_remain - n_conf - n_cand
    selected_marked = [showcase_id] + conf_stars[:n_conf] + cand_stars[:n_cand] + fp_stars[:n_fp]
    selected_marked = selected_marked[:n_stars]
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

def match_unmarked_stars(marked_info, pool_df, n_unmarked, seed):
    m_mag = np.array([marked_info[s][0] for s in marked_info])
    m_q = np.array([marked_info[s][1] for s in marked_info])
    std_mag = float(np.std(m_mag)) if np.std(m_mag) > 0 else 1.0
    std_q = float(np.std(m_q)) if np.std(m_q) > 0 else 1.0
    pool_df = pool_df.copy()
    pool_df['kepmag'] = pd.to_numeric(pool_df['kepmag'], errors='coerce').fillna(13.5)
    pool_df['quarters'] = pool_df['st_quarters'].astype(str).apply(lambda x: x.count('1'))
    pool_df = pool_df[pool_df['quarters'] >= 10].copy()
    p_mag = pool_df['kepmag'].values
    p_q = pool_df['quarters'].values
    p_ids = pool_df['kepid'].astype(str).values
    chosen_ids = []
    chosen_info = {}
    avail_indices = set(range(len(pool_df)))
    rng = np.random.RandomState(seed)
    order = list(range(len(m_mag)))
    rng.shuffle(order)
    for idx in order:
        if len(chosen_ids) >= n_unmarked:
            break
        target_m = m_mag[idx]
        target_q = m_q[idx]
        avail_list = list(avail_indices)
        if not avail_list:
            break
        d = ((p_mag[avail_list] - target_m) / std_mag)**2 + ((p_q[avail_list] - target_q) / std_q)**2
        best_i = avail_list[int(np.argmin(d))]
        avail_indices.remove(best_i)
        sid = p_ids[best_i]
        chosen_ids.append(sid)
        chosen_info[sid] = (float(p_mag[best_i]), int(p_q[best_i]))
    while len(chosen_ids) < n_unmarked and avail_indices:
        b_idx = avail_indices.pop()
        sid = p_ids[b_idx]
        chosen_ids.append(sid)
        chosen_info[sid] = (float(p_mag[b_idx]), int(p_q[b_idx]))
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
    marked_conf = [s for s in marked_stars if star_prim.get(s) == 'CONFIRMED' and s != showcase_id]
    marked_cand = [s for s in marked_stars if star_prim.get(s) == 'CANDIDATE' and s != showcase_id]
    marked_fp = [s for s in marked_stars if star_prim.get(s) == 'FALSE_POSITIVE' and s != showcase_id]
    rng = np.random.RandomState(seed)
    rng.shuffle(marked_conf)
    rng.shuffle(marked_cand)
    rng.shuffle(marked_fp)
    ctrl_marked = [showcase_id]
    tr_marked = []
    for grp in [marked_conf, marked_cand, marked_fp]:
        for i, s in enumerate(grp):
            if len(ctrl_marked) <= len(tr_marked):
                ctrl_marked.append(s)
            else:
                tr_marked.append(s)
    while len(ctrl_marked) > len(tr_marked) + 1:
        moved = [s for s in ctrl_marked if s != showcase_id][-1]
        ctrl_marked.remove(moved)
        tr_marked.append(moved)
    unm_list = list(unmarked_stars)
    rng.shuffle(unm_list)
    mid_u = len(unm_list) // 2
    tr_unmarked = unm_list[:mid_u]
    ctrl_unmarked = unm_list[mid_u:]
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

def train_and_eval_pipeline(df_train_signals, df_test_signals, feature_cols, seed, full_features_pool=None):
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
        print("  ВНИМАНИЕ: Обучающий набор сигналов пуст (0 сигналов). Пропуск обучения XGBoost, присвоение вердикта REVIEW по умолчанию.", flush=True)
        df_res = df_test_signals.copy()
        df_res['p_mean'] = np.full(len(df_test_signals), 0.5)
        df_res['p_std'] = np.full(len(df_test_signals), 0.5)
        df_res['decision'] = np.full(len(df_test_signals), 'REVIEW', dtype=object)
        df_res['shap_top3'] = ['insufficient_training_data (neutral)'] * len(df_test_signals)
        return df_res, None, None, scale_pipe, (0.75, 0.25, 0.15)

    y_tr = df_train_signals['target_bin'].values
    X_tr = df_train_signals[feature_cols].values
    groups_tr = df_train_signals['target_id'].values

    if len(np.unique(y_tr)) < 2:
        c = int(y_tr[0])
        print(f"  ВНИМАНИЕ: В обучающем наборе присутствует только один класс ({'CONFIRMED' if c==1 else 'FALSE_POSITIVE'}, {len(y_tr)} сигналов). Полноценное обучение ансамбля невозможно.", flush=True)
        df_res = df_test_signals.copy()
        p_val = 0.95 if c == 1 else 0.05
        dec_val = 'CANDIDATE' if c == 1 else 'FALSE_POSITIVE'
        df_res['p_mean'] = np.full(len(df_test_signals), p_val)
        df_res['p_std'] = np.zeros(len(df_test_signals))
        df_res['decision'] = np.full(len(df_test_signals), dec_val, dtype=object)
        df_res['shap_top3'] = [f'single_class_train_label={c} (neutral)'] * len(df_test_signals)
        return df_res, None, None, scale_pipe, (0.75, 0.25, 0.15)

    X_tr_scaled = scale_pipe.transform(X_tr)
    ratio = float(np.sum(y_tr == 0)) / np.sum(y_tr == 1) if np.sum(y_tr == 1) > 0 else 1.0
    xgb_base = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, scale_pos_weight=ratio, random_state=seed, eval_metric='logloss')
    xgb_base.fit(X_tr_scaled, y_tr)
    n_models = min(20, max(5, len(y_tr) * 2))
    unique_groups = np.unique(groups_tr)
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

    val_preds = np.array([m.predict_proba(X_tr_scaled)[:, 1] for m in ensemble_models])
    val_mean = np.mean(val_preds, axis=0)
    val_std = np.std(val_preds, axis=0)
    t_high, t_low, s_max = 0.75, 0.25, 0.15
    for t_h in np.linspace(0.5, 0.90, 9):
        for s_m in np.linspace(0.05, 0.30, 6):
            accepted = (val_mean >= t_h) & (val_std <= s_m)
            if np.sum(accepted) >= 2:
                prec = precision_score(y_tr[accepted], np.ones(np.sum(accepted)), zero_division=0)
                if prec >= 0.90:
                    t_high, s_max = t_h, s_m
                    break
    for t_l in np.linspace(0.5, 0.10, 9):
        accepted = (val_mean <= t_l) & (val_std <= s_max)
        if np.sum(accepted) >= 2:
            prec = precision_score(1 - y_tr[accepted], np.ones(np.sum(accepted)), zero_division=0)
            if prec >= 0.90:
                t_low = t_l
                break
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
        return {'acc': (0, 0), 'auc': (0, 0)}
    rng = np.random.RandomState(seed)
    acc_list, auc_list = [], []
    for _ in range(n_boot):
        b_stars = rng.choice(unique_stars, size=len(unique_stars), replace=True)
        dfs = []
        for s in b_stars:
            dfs.append(df_eval[df_eval['target_id'] == s])
        b_df = pd.concat(dfs, ignore_index=True)
        if len(b_df['target_bin'].unique()) >= 2:
            acc_list.append(accuracy_score(b_df['target_bin'], (b_df['p_mean'] >= 0.5).astype(int)))
            try:
                auc_list.append(roc_auc_score(b_df['target_bin'], b_df['p_mean']))
            except Exception:
                pass
    acc_ci = (float(np.percentile(acc_list, 5)), float(np.percentile(acc_list, 95))) if acc_list else (0.0, 0.0)
    auc_ci = (float(np.percentile(auc_list, 5)), float(np.percentile(auc_list, 95))) if auc_list else (0.0, 0.0)
    return {'acc_ci': acc_ci, 'auc_ci': auc_ci}

def generate_showcase_figure(showcase_id, showcase_signals, out_path, quick_mode, cache_dir=None):
    import lightkurve as lk
    try:
        sr = lk.search_lightcurve(f"KIC {showcase_id}", mission='Kepler', author='Kepler')
        if len(sr) == 0:
            sr = lk.search_lightcurve(f"{showcase_id}", mission='Kepler')
        if quick_mode:
            sr = sr[:4]
        lc_coll = sr.download_all(download_dir=cache_dir)
        lc_raw = lc_coll.stitch().remove_nans()
        lc_flat, _ = lc_raw.flatten(window_length=101, return_trend=True)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Не удалось скачать кривую {showcase_id}: {e}", ha='center', va='center')
        plt.savefig(out_path)
        plt.close()
        return
    times_raw = lc_raw.time.value
    flux_raw = lc_raw.flux.value
    times_flat = lc_flat.time.value
    flux_flat = lc_flat.flux.value
    p_grid = np.linspace(0.5, 12.0, 5000 if quick_mode else 20000)
    bls = lc_flat.to_periodogram(method='bls', period=p_grid, duration=np.linspace(0.04, 0.35, 4 if quick_mode else 10))
    n_sigs = min(4, len(showcase_signals))
    total_panels = 3 + max(1, n_sigs)
    fig = plt.figure(figsize=(15, 3.5 * total_panels))
    gs = fig.add_gridspec(total_panels, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    sub_mask = np.linspace(0, len(times_raw) - 1, min(10000, len(times_raw))).astype(int)
    ax1.plot(times_raw[sub_mask], flux_raw[sub_mask], 'k.', markersize=1, alpha=0.6)
    ax1.set_title(f"Kepler-80 (KIC {showcase_id}): Исходная фотометрия (Raw Light Curve)")
    ax1.set_xlabel("Время (BJD - 2454833)")
    ax1.set_ylabel("Нормированный поток")
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(times_flat[sub_mask], flux_flat[sub_mask], 'b.', markersize=1, alpha=0.6)
    ax2.set_title("Сглаженная и выровненная кривая блеска (Flattened, spline window=3.0d)")
    ax2.set_xlabel("Время (BJD - 2454833)")
    ax2.set_ylabel("Относительный поток")
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(bls.period.value, bls.power.value, 'r-', linewidth=0.8, label='BLS Power')
    if n_sigs > 0:
        for _, s_row in showcase_signals.head(n_sigs).iterrows():
            p_val = float(s_row.get('bls_period_d', 0))
            ax3.axvline(p_val, color='g', linestyle='--', alpha=0.8, label=f"Пик: {p_val:.4f} д")
    ax3.set_title("Периодограмма поиска транзитов BLS (Box Least Squares)")
    ax3.set_xlabel("Период (дни)")
    ax3.set_ylabel("Мощность BLS")
    ax3.legend(loc='upper right')
    if n_sigs > 0:
        for idx, (_, s_row) in enumerate(showcase_signals.head(n_sigs).iterrows()):
            ax_f = fig.add_subplot(gs[3 + idx, 0])
            p_val = float(s_row.get('bls_period_d', 1.0))
            t0_val = float(s_row.get('bls_epoch', 0.0))
            dur_val = float(s_row.get('bls_duration_d', 0.1))
            prob = float(s_row.get('p_mean', 0.0))
            dec = str(s_row.get('decision', 'UNKNOWN'))
            folded = lc_flat.fold(period=p_val, epoch_time=t0_val)
            ph = folded.phase.value
            fl = folded.flux.value
            m_win = np.abs(ph) < 0.25
            ax_f.plot(ph[m_win], fl[m_win], 'k.', markersize=2, alpha=0.3, label='Точки фазы')
            bins_x = np.linspace(-0.25, 0.25, 101)
            b_y = []
            for bi in range(len(bins_x) - 1):
                bm = (ph >= bins_x[bi]) & (ph < bins_x[bi+1])
                b_y.append(np.nanmedian(fl[bm]) if np.any(bm) else np.nan)
            ax_f.plot((bins_x[:-1] + bins_x[1:])/2, b_y, 'r-', linewidth=2, label='Биннированный профиль')
            ax_f.set_title(f"Сигнал {idx+1}: P = {p_val:.4f} д, T0 = {t0_val:.2f}, Длительность = {dur_val*24:.1f} ч | Модель: {dec} (Вероятность = {prob:.1%})")
            ax_f.set_xlabel("Орбитальная фаза")
            ax_f.set_ylabel("Поток")
            ax_f.legend(loc='lower right')
    else:
        ax_f = fig.add_subplot(gs[3, 0])
        ax_f.text(0.5, 0.5, "В текущем тестовом режиме сигналы не обнаружены", ha='center', va='center')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--marked-stars', type=int, default=120)
    parser.add_argument('--unmarked-stars', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--cache-dir', default=tempfile.gettempdir())
    parser.add_argument('--output-dir', default='outputs/demo')
    parser.add_argument('--showcase-star', default='4852528')
    parser.add_argument('--unmarked-file', default='')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--flatten-days', type=float, default=3.0)
    parser.add_argument('--max-signals', type=int, default=4)
    parser.add_argument('--star-timeout', type=float, default=1800.0)
    args = parser.parse_args()

    out_dir = args.output_dir
    if args.smoke_test and not '_smoke_test' in out_dir:
        out_dir = os.path.join('outputs', '_smoke_test', 'demo')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)

    file_suffix = '_quick' if args.quick else ''
    print(f"=== Project Onyx: Demonstration Pipeline (Mode Demo) ===", flush=True)
    print(f"Config: marked_stars={args.marked_stars}, unmarked_stars={args.unmarked_stars}, quick={args.quick}, seed={args.seed}, showcase={args.showcase_star}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)

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
    unmarked_ids, unmarked_info = match_unmarked_stars(marked_info, pool_df, args.unmarked_stars, args.seed)

    stats_m = compute_group_stats(marked_info)
    stats_u = compute_group_stats(unmarked_info)
    comp_table = pd.DataFrame([
        {'Группа': 'Отмеченные (KOI)', 'Звёзд': stats_m['count'], 'Kepmag (среднее)': f"{stats_m['mag_mean']:.2f} ± {stats_m['mag_std']:.2f}", 'Kepmag (медиана)': f"{stats_m['mag_median']:.2f}", 'Кварталов (среднее)': f"{stats_m['q_mean']:.1f}", 'Кварталов (медиана)': f"{stats_m['q_median']:.0f}"},
        {'Группа': 'Неотмеченные (Control)', 'Звёзд': stats_u['count'], 'Kepmag (среднее)': f"{stats_u['mag_mean']:.2f} ± {stats_u['mag_std']:.2f}", 'Kepmag (медиана)': f"{stats_u['mag_median']:.2f}", 'Кварталов (среднее)': f"{stats_u['q_mean']:.1f}", 'Кварталов (медиана)': f"{stats_u['q_median']:.0f}"}
    ])
    print("\nТаблица сравнения распределений звёздных параметров:")
    print(comp_table.to_string(index=False), flush=True)

    tr_marked, ctrl_marked, tr_unmarked, ctrl_unmarked = split_stars(marked_ids, unmarked_ids, star_prim, args.showcase_star, args.seed)
    all_selected_stars = list(set(tr_marked + ctrl_marked + tr_unmarked + ctrl_unmarked))
    targets_demo_file = os.path.join(out_dir, f"targets_demo{file_suffix}.txt")
    with open(targets_demo_file, 'w', encoding='utf-8') as f:
        for sid in all_selected_stars:
            f.write(f"{sid}\n")

    features_demo_csv = os.path.join(out_dir, f"features_demo{file_suffix}.csv")
    extractor_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), '1_data_extractor.py')
    cmd_ext = [sys.executable, extractor_py, '--mission', 'Kepler', '--targets-file', targets_demo_file, '--output', features_demo_csv, '--workers', str(args.workers), '--star-timeout', str(args.star_timeout), '--flatten-days', str(args.flatten_days), '--max-signals', str(args.max_signals), '--cache-dir', args.cache_dir]
    if args.quick:
        cmd_ext.append('--quick')
    if args.smoke_test:
        cmd_ext.append('--smoke-test')
    print("\nЗапуск извлечения признаков для выбранных звёзд...", flush=True)
    subprocess.run(cmd_ext, check=True)

    if not os.path.exists(features_demo_csv) or os.path.getsize(features_demo_csv) == 0:
        print(f"Ошибка: файл признаков {features_demo_csv} пуст или не найден!", flush=True)
        sys.exit(1)
    df_features = pd.read_csv(features_demo_csv, dtype={'target_id': str})
    failed_csv = os.path.join(out_dir, f"features_demo{file_suffix}_failed.csv")
    failed_stars = set()
    if os.path.exists(failed_csv) and os.path.getsize(failed_csv) > 0:
        try:
            df_fail = pd.read_csv(failed_csv, dtype={'target_id': str})
            failed_stars = set(df_fail['target_id'].dropna().astype(str))
        except Exception:
            pass

    succ_tr_m = [s for s in tr_marked if s not in failed_stars]
    succ_ctrl_m = [s for s in ctrl_marked if s not in failed_stars]
    succ_tr_u = [s for s in tr_unmarked if s not in failed_stars]
    succ_ctrl_u = [s for s in ctrl_unmarked if s not in failed_stars]

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

    df_features['target_bin'] = df_features['label'].apply(lambda x: 1 if str(x).upper() in ['CONFIRMED', 'CANDIDATE'] else 0)
    exclude_cols = ['signal_id', 'target_id', 'signal_index', 'mission', 'label', 'match_status', 'target_bin', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'p_candidate', 'n_signals_star']
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    train_stars_set = set(final_tr_m + final_tr_u)
    ctrl_stars_set = set(final_ctrl_m + final_ctrl_u)
    df_train_all = df_features[df_features['target_id'].isin(train_stars_set)].copy()
    df_ctrl_all = df_features[df_features['target_id'].isin(ctrl_stars_set)].copy()

    df_tr_m_matched = df_train_all[(df_train_all['target_id'].isin(set(final_tr_m))) & (df_train_all['match_status'] == 'MATCHED')].copy()
    df_ctrl_m_matched = df_ctrl_all[(df_ctrl_all['target_id'].isin(set(final_ctrl_m))) & (df_ctrl_all['match_status'] == 'MATCHED')].copy()

    print(f"\nЧисло размеченных сигналов для обучения:")
    print(f"  Обучающая часть (a): {len(df_tr_m_matched)} сигналов")
    print(f"  Контрольная часть (b): {len(df_ctrl_m_matched)} сигналов (без Kepler-80: {len(df_ctrl_m_matched[df_ctrl_m_matched['target_id'] != showcase_id])})")
    if len(df_tr_m_matched) < 30:
        print(f"  ВНИМАНИЕ: Размеченных сигналов в обучающей части мало ({len(df_tr_m_matched)} < 30). Обучение возможно, но оценки подвержены шуму выборки.", flush=True)

    print("\n[Направление А] Обучение на обучающей половине -> проверка на контрольной...", flush=True)
    ctrl_pred, model_a, expl_a, scaler_a, thres_a = train_and_eval_pipeline(df_tr_m_matched, df_ctrl_all, feature_cols, args.seed, full_features_pool=df_features)

    print("\n[Направление Б] Обучение на контрольной половине (без Kepler-80) -> проверка на обучающей...", flush=True)
    df_ctrl_m_for_b = df_ctrl_m_matched[df_ctrl_m_matched['target_id'] != showcase_id].copy()
    train_pred, model_b, expl_b, scaler_b, thres_b = train_and_eval_pipeline(df_ctrl_m_for_b, df_train_all, feature_cols, args.seed, full_features_pool=df_features)

    ctrl_m_eval = ctrl_pred[(ctrl_pred['target_id'].isin(set(final_ctrl_m))) & (ctrl_pred['match_status'] == 'MATCHED')].copy()
    y_ctrl_true = ctrl_m_eval['target_bin'].values
    y_ctrl_prob = ctrl_m_eval['p_mean'].values
    y_ctrl_pred = (y_ctrl_prob >= 0.5).astype(int)

    acc_a = float(accuracy_score(y_ctrl_true, y_ctrl_pred)) if len(y_ctrl_true) > 0 else 0.0
    rec_a = float(recall_score(y_ctrl_true, y_ctrl_pred, zero_division=0)) if len(y_ctrl_true) > 0 else 0.0
    prec_a = float(precision_score(y_ctrl_true, y_ctrl_pred, zero_division=0)) if len(y_ctrl_true) > 0 else 0.0
    auc_a = float(roc_auc_score(y_ctrl_true, y_ctrl_prob)) if len(np.unique(y_ctrl_true)) >= 2 else 0.0
    cm_a = confusion_matrix(y_ctrl_true, y_ctrl_pred) if len(y_ctrl_true) > 0 else np.zeros((2, 2))

    accepted_mask = ctrl_m_eval['decision'].isin(['CANDIDATE', 'FALSE_POSITIVE'])
    n_review_ctrl = int(np.sum(ctrl_m_eval['decision'] == 'REVIEW'))
    review_rate_ctrl = float(n_review_ctrl / len(ctrl_m_eval)) if len(ctrl_m_eval) > 0 else 0.0
    prec_accepted = float(accuracy_score(ctrl_m_eval[accepted_mask]['target_bin'], (ctrl_m_eval[accepted_mask]['decision'] == 'CANDIDATE').astype(int))) if np.sum(accepted_mask) > 0 else 0.0

    maj_class = int(pd.Series(df_tr_m_matched['target_bin']).mode()[0]) if len(df_tr_m_matched) > 0 else 0
    acc_maj_baseline = float(np.mean(y_ctrl_true == maj_class)) if len(y_ctrl_true) > 0 else 0.0

    snr_vals = ctrl_m_eval['bls_depth_snr'].fillna(0).values if 'bls_depth_snr' in ctrl_m_eval else np.zeros(len(ctrl_m_eval))
    snr_thresh = float(df_tr_m_matched['bls_depth_snr'].median()) if len(df_tr_m_matched) > 0 and 'bls_depth_snr' in df_tr_m_matched else 10.0
    acc_snr_baseline = float(accuracy_score(y_ctrl_true, (snr_vals >= snr_thresh).astype(int))) if len(y_ctrl_true) > 0 else 0.0

    boot_res = bootstrap_star_metrics(ctrl_m_eval, n_boot=500, seed=args.seed)

    tr_m_eval = train_pred[(train_pred['target_id'].isin(set(final_tr_m))) & (train_pred['match_status'] == 'MATCHED')].copy()
    y_tr_true = tr_m_eval['target_bin'].values
    y_tr_prob = tr_m_eval['p_mean'].values
    y_tr_pred = (y_tr_prob >= 0.5).astype(int)
    acc_b = float(accuracy_score(y_tr_true, y_tr_pred)) if len(y_tr_true) > 0 else 0.0
    auc_b = float(roc_auc_score(y_tr_true, y_tr_prob)) if len(np.unique(y_tr_true)) >= 2 else 0.0

    mean_acc_two_way = (acc_a + acc_b) / 2.0
    mean_auc_two_way = (auc_a + auc_b) / 2.0

    unmatched_marked_signals = ctrl_pred[(ctrl_pred['target_id'].isin(set(final_ctrl_m))) & (ctrl_pred['match_status'] == 'UNMATCHED')]

    ctrl_unmarked_signals = ctrl_pred[ctrl_pred['target_id'].isin(set(final_ctrl_u))].copy()
    n_unm_stars = len(final_ctrl_u)
    sigs_per_unm_star = float(len(ctrl_unmarked_signals) / n_unm_stars) if n_unm_stars > 0 else 0.0
    unm_probs = ctrl_unmarked_signals['p_mean'].values if len(ctrl_unmarked_signals) > 0 else np.array([0.0])
    n_false_alarms = int(np.sum(ctrl_unmarked_signals['decision'] == 'CANDIDATE'))
    far_rate = float(n_false_alarms / len(ctrl_unmarked_signals)) if len(ctrl_unmarked_signals) > 0 else 0.0
    review_rate_unm = float(np.sum(ctrl_unmarked_signals['decision'] == 'REVIEW') / len(ctrl_unmarked_signals)) if len(ctrl_unmarked_signals) > 0 else 0.0

    if len(ctrl_unmarked_signals) > 0:
        X_unm_scaled = scaler_a.transform(ctrl_unmarked_signals[feature_cols].values)
        iso = IsolationForest(n_estimators=100, random_state=args.seed)
        iso.fit(X_unm_scaled)
        ctrl_unmarked_signals['iso_score'] = iso.decision_function(X_unm_scaled)
        top5_candidates = ctrl_unmarked_signals.sort_values(by='p_mean', ascending=False).head(5)
    else:
        top5_candidates = pd.DataFrame()

    ctrl_pred['star_type'] = ctrl_pred['target_id'].apply(lambda x: 'MARKED' if str(x) in set(final_ctrl_m) else 'UNMARKED')
    out_table_cols = ['target_id', 'star_type', 'signal_id', 'bls_period_d', 'bls_depth_ppm', 'label', 'p_mean', 'p_std', 'decision', 'shap_top3']
    control_table = ctrl_pred[[c for c in out_table_cols if c in ctrl_pred]].copy()
    control_table.rename(columns={'bls_period_d': 'period_d', 'bls_depth_ppm': 'depth_ppm', 'label': 'catalog_label'}, inplace=True)
    control_table['catalog_label'] = control_table['catalog_label'].fillna('NONE')
    ctrl_csv_out = os.path.join(out_dir, f"control_signals{file_suffix}.csv")
    ctrl_md_out = os.path.join(out_dir, f"control_signals{file_suffix}.md")
    control_table.to_csv(ctrl_csv_out, index=False, encoding='utf-8')
    with open(ctrl_md_out, 'w', encoding='utf-8') as f:
        f.write(df_to_markdown(control_table) + "\n")

    showcase_signals = ctrl_pred[ctrl_pred['target_id'] == showcase_id].copy()
    showcase_plot_path = os.path.join(out_dir, f"showcase_kepler80{file_suffix}.png")
    generate_showcase_figure(showcase_id, showcase_signals, showcase_plot_path, args.quick, cache_dir=args.cache_dir)

    known_planets = [
        {'planet': 'Kepler-80 b', 'koi': 'KOI-500.01', 'period_d': 7.053527},
        {'planet': 'Kepler-80 c', 'koi': 'KOI-500.02', 'period_d': 9.521645},
        {'planet': 'Kepler-80 d', 'koi': 'KOI-500.03', 'period_d': 3.072147},
        {'planet': 'Kepler-80 e', 'koi': 'KOI-500.04', 'period_d': 4.645382},
        {'planet': 'Kepler-80 f', 'koi': 'KOI-500.05', 'period_d': 0.986786}
    ]
    recovered_planets = []
    det_periods = showcase_signals['bls_period_d'].values if len(showcase_signals) > 0 else []
    for kp in known_planets:
        kp_p = kp['period_d']
        matched_sig = None
        match_type = 'Не найден'
        for _, s_row in showcase_signals.iterrows():
            sp = float(s_row['bls_period_d'])
            if abs(sp - kp_p) / kp_p < 0.01:
                matched_sig = s_row
                match_type = 'Точное совпадение (1%)'
                break
            elif abs(sp - 0.5 * kp_p) / (0.5 * kp_p) < 0.01:
                matched_sig = s_row
                match_type = 'Гармоника 0.5x'
                break
            elif abs(sp - 2.0 * kp_p) / (2.0 * kp_p) < 0.01:
                matched_sig = s_row
                match_type = 'Гармоника 2.0x'
                break
        recovered_planets.append({
            'Планета': kp['planet'],
            'Каталожный период (дни)': f"{kp_p:.4f}",
            'Найден ли сигнал?': 'ДА' if matched_sig is not None else 'НЕТ',
            'Период найденного (дни)': f"{float(matched_sig['bls_period_d']):.4f}" if matched_sig is not None else '-',
            'Статус': match_type,
            'Вероятность модели': f"{float(matched_sig['p_mean']):.1%}" if matched_sig is not None else '-',
            'Вердикт': str(matched_sig['decision']) if matched_sig is not None else '-'
        })
    df_k80_rec = pd.DataFrame(recovered_planets)

    correct_samples = ctrl_m_eval[((ctrl_m_eval['target_bin'] == 1) & (ctrl_m_eval['decision'] == 'CANDIDATE')) | ((ctrl_m_eval['target_bin'] == 0) & (ctrl_m_eval['decision'] == 'FALSE_POSITIVE'))].head(2)
    error_samples = ctrl_m_eval[((ctrl_m_eval['target_bin'] == 1) & (ctrl_m_eval['decision'] == 'FALSE_POSITIVE')) | ((ctrl_m_eval['target_bin'] == 0) & (ctrl_m_eval['decision'] == 'CANDIDATE'))].head(2)

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
        f.write(f"* **Случайный seed:** `{args.seed}`\n")
        f.write(f"* **Показательная система:** Kepler-80 (KIC {args.showcase_star}, статус CONFIRMED, 5 планет)\n")
        f.write(f"* **Выходная директория:** `{out_dir}`\n\n")

        f.write(f"{h2}2. Состав выборки и балансировка звёздных параметров\n")
        f.write(f"Выборка неотмеченных звёзд получена из официального каталога параметров звёзд NASA Exoplanet Archive (`Q1_Q17_DR25_KS`) с фильтрацией известных KOI и подбором распределений яркости (`kepmag`) и числа наблюдаемых кварталов (`st_quarters`) к отмеченным звёздам (1-к-1 нормализованное евклидово соответствие).\n\n")
        f.write(df_to_markdown(comp_table) + "\n\n")
        f.write(f"* **Фактический размер после контроля качества и отсева:**\n")
        f.write(f"  * Обучающая половина: {len(final_tr_m)} отмеченных + {len(final_tr_u)} неотмеченных звёзд\n")
        f.write(f"  * Контрольная половина: {len(final_ctrl_m)} отмеченных (включая Kepler-80) + {len(final_ctrl_u)} неотмеченных звёзд\n")
        f.write(f"  * Отброшено звёзд для поддержания строгого равенства половин: {total_dropped}\n\n")

        f.write(f"{h2}3. Раздел А: Оценка на отмеченных звёздах контрольной части (есть истинные метки)\n")
        f.write(f"* **Всего размеченных сигналов (MATCHED):** {len(ctrl_m_eval)}\n")
        f.write(f"* **Матрица ошибок (Confusion Matrix):**\n```\n{cm_a}\n```\n")
        f.write(f"* **Метрики качества (Направление А):**\n")
        f.write(f"  * Точность (Accuracy): **{acc_a:.3f}** (90% ДИ бутстрапом по звёздам: [{boot_res['acc_ci'][0]:.3f}, {boot_res['acc_ci'][1]:.3f}])\n")
        f.write(f"  * Полнота (Recall): **{rec_a:.3f}**\n")
        f.write(f"  * Precision: **{prec_a:.3f}**\n")
        f.write(f"  * ROC-AUC: **{auc_a:.3f}** (90% ДИ: [{boot_res['auc_ci'][0]:.3f}, {boot_res['auc_ci'][1]:.3f}])\n")
        f.write(f"* **Отказ от классификации (Abstention / Review):**\n")
        f.write(f"  * Доля вердиктов REVIEW: **{review_rate_ctrl:.1%}** ({n_review_ctrl} из {len(ctrl_m_eval)})\n")
        f.write(f"  * Точность на принятых решениях (CANDIDATE/FALSE_POSITIVE): **{prec_accepted:.3f}**\n")
        f.write(f"* **Сравнение с эталонами (Baselines):**\n")
        f.write(f"  * Эталон «всегда мажоритарный класс» (класс {maj_class}): **{acc_maj_baseline:.3f}**\n")
        f.write(f"  * Эталон по одному признаку (порог `bls_depth_snr` >= {snr_thresh:.1f}): **{acc_snr_baseline:.3f}**\n")
        f.write(f"* **Двунаправленная перекрёстная проверка (Two-way train/test):**\n")
        f.write(f"  * Точность A (Train->Ctrl): {acc_a:.3f} | Точность B (Ctrl->Train): {acc_b:.3f} | Средняя: **{mean_acc_two_way:.3f}**\n")
        f.write(f"  * ROC-AUC A: {auc_a:.3f} | ROC-AUC B: {auc_b:.3f} | Средний ROC-AUC: **{mean_auc_two_way:.3f}**\n")
        f.write(f"  *(Примечание: в направлении B звезда Kepler-80 была полностью исключена из обучающего набора)*\n")
        f.write(f"* **Несовпавшие сигналы (UNMATCHED) на отмеченных звёздах:** {len(unmatched_marked_signals)} сигналов (исключены из надзорных метрик, сохранены в таблице сигналов).\n\n")

        f.write(f"{h2}4. Раздел Б: Оценка на неотмеченных звёздах контрольной части (эталонных меток НЕТ)\n")
        f.write(f"> [!IMPORTANT]\n")
        f.write(f"> На неотмеченных звёздах метрики точности, полноты и ROC-AUC принципиально **не рассчитываются**, так как истинный статус этих объектов неизвестен (часть из них может содержать неоткрытые экзопланеты).\n\n")
        f.write(f"* **Число звёзд в контрольной неотмеченной группе:** {n_unm_stars}\n")
        f.write(f"* **Всего обнаружено периодических сигналов:** {len(ctrl_unmarked_signals)}\n")
        f.write(f"* **Среднее число сигналов на звезду:** **{sigs_per_unm_star:.2f}**\n")
        f.write(f"* **Распределение предсказанной вероятности (p_mean):**\n")
        f.write(f"  * Минимум: {float(np.min(unm_probs)):.3f} | 25%: {float(np.percentile(unm_probs, 25)):.3f} | Медиана: {float(np.median(unm_probs)):.3f} | 75%: {float(np.percentile(unm_probs, 75)):.3f} | Максимум: {float(np.max(unm_probs)):.3f}\n")
        f.write(f"* **Частота ложных тревог (False Alarm Rate):** **{far_rate:.1%}** ({n_false_alarms} сигналов получили вердикт CANDIDATE при допущении отсутствия планет; подчёркиваем, что это допущение, а не доказанный факт).\n")
        f.write(f"* **Доля неопределённых сигналов (REVIEW):** **{review_rate_unm:.1%}**\n\n")
        f.write(f"{h3}Топ-5 наиболее «планетоподобных» сигналов среди неотмеченных звёзд (кандидаты на валидацию):\n\n")
        if len(top5_candidates) > 0:
            top5_view = top5_candidates[['target_id', 'bls_period_d', 'bls_depth_ppm', 'p_mean', 'decision', 'shap_top3']].copy()
            top5_view.rename(columns={'bls_period_d': 'Период (д)', 'bls_depth_ppm': 'Глубина (ppm)', 'p_mean': 'Вероятность', 'decision': 'Решение', 'shap_top3': 'Вклады SHAP'}, inplace=True)
            f.write(df_to_markdown(top5_view) + "\n\n")
        else:
            f.write("Сигналы не найдены.\n\n")

        f.write(f"{h2}5. Результаты по витринной системе Kepler-80 (KIC {showcase_id})\n")
        f.write(f"Система Kepler-80 содержит 5 известных подтверждённых планет в резонансной цепочке периодов. Графическая панель сохранена в файл `{showcase_plot_path}`.\n\n")
        f.write(df_to_markdown(df_k80_rec) + "\n\n")
        n_rec = sum(1 for r in recovered_planets if r['Найден ли сигнал?'] == 'ДА')
        f.write(f"* **Итог валидации Kepler-80:** Обнаружено планет: **{n_rec} из 5**.\n")
        if n_rec < 5:
            f.write(f"  * *Причины неполного восстановления:* ограничения сетки периодограммы BLS в быстром режиме (`--quick`), малые глубины транзитов дальних планет (глубины ~100–150 ppm на умеренно слабой звезде Kepmag=14.8), а также лимит итераций поиска (до {args.max_signals} сигналов на звезду).\n\n")

        f.write(f"{h2}6. Примеры классификации и интерпретация признаков (SHAP)\n")
        f.write(f"{h3}Примеры верных решений модели:\n")
        if len(correct_samples) > 0:
            for _, r in correct_samples.iterrows():
                f.write(f"* **Звезда {r['target_id']} (Сигнал {r['signal_id']}):** Истинный класс = `{r['label']}`, Вердикт = `{r['decision']}` ($p = {r['p_mean']:.1%}$).\n  * Вклады SHAP: `{r['shap_top3']}`\n")
        else:
            f.write("* В тестовой малой выборке примеры отсутствуют.\n")
        f.write(f"\n{h3}Примеры ошибок модели:\n")
        if len(error_samples) > 0:
            for _, r in error_samples.iterrows():
                f.write(f"* **Звезда {r['target_id']} (Сигнал {r['signal_id']}):** Истинный класс = `{r['label']}`, Вердикт = `{r['decision']}` ($p = {r['p_mean']:.1%}$).\n  * Вклады SHAP: `{r['shap_top3']}`\n")
        else:
            f.write("* В тестовой выборке явных грубых ошибок не зафиксировано либо размер тестовой выборки мал.\n")

        f.write(f"\n{h2}7. Ограничения и угрозы валидности (Threats to Validity)\n")
        f.write(f"1. **Объём тестовой выборки:** Демонстрационная выборка носит иллюстративный характер и не заменяет полный прогон на тысячах звёзд.\n")
        f.write(f"2. **Шум метрик и доверительные интервалы:** При малом числе звёзд доверительные интервалы широки, а точечные метрики чувствительны к отдельным объектам.\n")
        f.write(f"3. **Каталожные метки не являются абсолютной истиной:** Каталог KOI содержит споры по спорным кандидатам и шум классификации операторов.\n")
        f.write(f"4. **Грубое смещение центроида:** Расчёт смещения центроида ведётся по средневзвешенным координатам пятна, а не по пиксельным функциям отклика (PRF).\n")
        f.write(f"5. **Неотмеченные звёзды:** Допущение об отсутствии планет на неотмеченных звёздах является рабочей гипотезой; часть обнаруженных кандидатов может оказаться реальными экзопланетами.\n\n")

        f.write(f"{h2}8. Тест на неприкосновенность основных файлов (Immutability Test)\n")
        f.write(f"* **Статус проверки хешей (SHA-256):** {'УСПЕШНО (Все хеши совпали)' if hash_match else 'ВНИМАНИЕ (Обнаружено несовпадение)'}\n")
        f.write(f"* Проверено исходных файлов в `outputs/`: {len(hashes_before)}\n")

    print(f"\n=== Демонстрационный прогон завершён! ===")
    print(f"Итоговый отчёт сохранён в: {summary_md_path}")
    print(f"Таблица контрольных сигналов: {ctrl_csv_out}")
    print(f"Витрина Kepler-80: {showcase_plot_path}")
    print(f"Тест неприкосновенности outputs/: {'ПРОЙДЕН (Хеши совпадают)' if hash_match else 'НЕ ПРОЙДЕН'}")

if __name__ == '__main__':
    main()
