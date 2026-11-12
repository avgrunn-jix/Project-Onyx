import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEARCH_MIN_PERIOD_D = 0.5
SEARCH_MAX_PERIOD_D = 100.0
PERIOD_TOL = 0.01
PERIOD_BINS = [(0.5, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, 100.0)]
MULTIPLICITY_BINS = [('1 KOI у звезды', 1, 1), ('2–3 KOI у звезды', 2, 3), ('4 и больше KOI у звезды', 4, 10 ** 6)]
POSITIVE_LABELS = ('CONFIRMED', 'CANDIDATE')
STAR_FEATURE_COLS = [
    'star_teff', 'star_radius', 'star_logg', 'flux_std_ppm', 'flux_mad_ppm', 'flux_skew',
    'flux_kurtosis', 'cdpp_ppm', 'pointwise_scatter_ppm', 'autocorr_max', 'ls_peak_power',
]
GEOMETRY_COLS = ['transit_count', 'gap_fraction']
CENTROID_COLS = ['centroid_shift_col_sigma', 'centroid_shift_row_sigma', 'centroid_shift_total_sigma', 'centroid_shift_total_arcsec']
SECONDARY_COLS = ['secondary_depth_ppm', 'secondary_snr', 'secondary_z', 'secondary_phase', 'odd_even_depth_diff_ppm', 'odd_even_sigma', 'odd_even_z']
DIAGNOSTIC_GROUPS = [
    ('Без центроидных признаков', 'центроидные признаки', CENTROID_COLS),
    ('Без вторичного затмения и чёт-нечёт', 'признаки вторичного затмения и чёт-нечёт', SECONDARY_COLS),
]
BOTH_DIAGNOSTIC_NAME = 'Без центроидных признаков, вторичного затмения и чёт-нечёт'
SYMMETRIC_T_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)
BRANCH_LABELS_RU = {
    'main': 'основная (цель достигнута)',
    'fallback': 'запасная (максимум OOF-точности, цель не достигнута)',
    'default': 'пороги по умолчанию (ни одна тройка порогов не даёт двух принятых сигналов)',
    'not_trained': 'подбор не выполнялся (модель не обучена)',
}
CRITERION_LABELS_RU = {
    'point': 'point (точечная OOF-точность на принятых не ниже цели)',
    'wilson': 'wilson (нижняя граница 95% ДИ Уилсона OOF-точности на принятых не ниже цели)',
}


def wilson_interval(k, n, z=1.96):
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def format_share(k, n):
    if n <= 0:
        return '-'
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} = {k / n:.1%} [{lo:.1%}; {hi:.1%}]"


def normalize_disposition(label):
    return str(label).upper().replace(' ', '_')


def period_status(cat_period, signal_periods, tol=PERIOD_TOL):
    status = 'не найден'
    best = None
    for idx, sp in enumerate(signal_periods):
        if not np.isfinite(sp) or sp <= 0:
            continue
        ratio = sp / cat_period
        if abs(ratio - 1.0) < tol:
            return 'точно', idx
        if status == 'не найден' and (abs(ratio - 0.5) < tol or abs(ratio - 2.0) < tol or abs(ratio * 3.0 - 1.0) < tol or abs(ratio / 3.0 - 1.0) < tol):
            status = 'гармоника'
            best = idx
    return status, best


def build_koi_recovery(koi_df, signals_df, min_period=SEARCH_MIN_PERIOD_D, max_period=SEARCH_MAX_PERIOD_D, tol=PERIOD_TOL):
    rows = []
    sig_by_star = {}
    if signals_df is not None and len(signals_df) > 0:
        for tid, grp in signals_df.groupby(signals_df['target_id'].astype(str)):
            sig_by_star[tid] = grp.reset_index(drop=True)
    has_depth = 'koi_depth' in koi_df.columns
    n_koi = koi_df['target_id'].astype(str).value_counts().to_dict() if len(koi_df) > 0 else {}
    for r in koi_df.to_dict('records'):
        tid = str(r['target_id'])
        cat_p = float(r['catalog_period']) if pd.notna(r.get('catalog_period')) else np.nan
        disp = normalize_disposition(r['label'])
        in_range = bool(np.isfinite(cat_p) and min_period <= cat_p <= max_period)
        status, decision = 'вне диапазона', ''
        found_p = np.nan
        if in_range:
            grp = sig_by_star.get(tid)
            if grp is None or len(grp) == 0:
                status = 'не найден'
            else:
                status, idx = period_status(cat_p, grp['bls_period_d'].astype(float).tolist(), tol)
                if idx is not None:
                    found_p = float(grp.loc[idx, 'bls_period_d'])
                    if 'decision' in grp.columns:
                        decision = str(grp.loc[idx, 'decision'])
        depth = pd.to_numeric(pd.Series([r.get('koi_depth', np.nan)]), errors='coerce').iloc[0] if has_depth else np.nan
        rows.append({'target_id': tid, 'disposition': disp, 'catalog_period': cat_p, 'koi_depth': float(depth), 'n_koi_star': int(n_koi.get(tid, 0)),
                     'in_range': in_range, 'status': status, 'found_period': found_p, 'decision': decision})
    return pd.DataFrame(rows, columns=['target_id', 'disposition', 'catalog_period', 'koi_depth', 'n_koi_star', 'in_range', 'status', 'found_period', 'decision'])


def _status_row(name, sub):
    n = len(sub)
    k_exact = int((sub['status'] == 'точно').sum())
    k_harm = int((sub['status'] == 'гармоника').sum())
    k_none = int((sub['status'] == 'не найден').sum())
    return {
        'Группа': name,
        'KOI в диапазоне': n,
        'Точно': format_share(k_exact, n),
        'Гармоника': format_share(k_harm, n),
        'Не найден': format_share(k_none, n),
    }


def summarize_koi_recovery(rec):
    in_rng = rec[rec['in_range']]
    out_of_range = int((~rec['in_range']).sum())
    by_disp = []
    for disp in ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']:
        sub = in_rng[in_rng['disposition'] == disp]
        if len(sub) > 0:
            by_disp.append(_status_row(disp, sub))
    by_disp.append(_status_row('Все', in_rng)) if len(in_rng) > 0 else None
    by_bin = []
    for lo, hi in PERIOD_BINS:
        last = hi == PERIOD_BINS[-1][1]
        mask = (in_rng['catalog_period'] >= lo) & ((in_rng['catalog_period'] <= hi) if last else (in_rng['catalog_period'] < hi))
        sub = in_rng[mask]
        if len(sub) > 0:
            by_bin.append(_status_row(f"P {lo:g}–{hi:g} д", sub))
    by_depth = []
    depth_vals = in_rng['koi_depth'] if 'koi_depth' in in_rng.columns else pd.Series(dtype=float)
    depth_ok = in_rng[np.isfinite(pd.to_numeric(depth_vals, errors='coerce').astype(float))] if len(in_rng) > 0 else in_rng
    if len(depth_ok) >= 4:
        edges = np.unique(np.quantile(depth_ok['koi_depth'].astype(float), [0.0, 0.25, 0.5, 0.75, 1.0]))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            last = i == len(edges) - 2
            m = (depth_ok['koi_depth'] >= lo) & ((depth_ok['koi_depth'] <= hi) if last else (depth_ok['koi_depth'] < hi))
            sub = depth_ok[m]
            if len(sub) > 0:
                by_depth.append(_status_row(f"Q{i + 1}: {lo:.0f}–{hi:.0f} ppm", sub))
    by_mult = []
    for name, lo, hi in MULTIPLICITY_BINS:
        sub = in_rng[(in_rng['n_koi_star'] >= lo) & (in_rng['n_koi_star'] <= hi)] if 'n_koi_star' in in_rng.columns else in_rng.iloc[0:0]
        if len(sub) > 0:
            by_mult.append(_status_row(name, sub))
    pos = in_rng[in_rng['disposition'].isin(POSITIVE_LABELS)]
    n_pos = len(pos)
    k_e2e = int(((pos['status'] == 'точно') & (pos['decision'] == 'CANDIDATE')).sum())
    k_found = int((pos['status'] == 'точно').sum())
    lo, hi = wilson_interval(k_e2e, n_pos)
    return {
        'by_disposition': pd.DataFrame(by_disp),
        'by_period_bin': pd.DataFrame(by_bin),
        'by_depth': pd.DataFrame(by_depth),
        'by_multiplicity': pd.DataFrame(by_mult),
        'n_depth_known': int(len(depth_ok)),
        'n_total': int(len(rec)),
        'n_in_range': int(len(in_rng)),
        'n_out_of_range': out_of_range,
        'n_positive_in_range': int(n_pos),
        'n_positive_found_exact': k_found,
        'n_end_to_end': k_e2e,
        'end_to_end_rate': float(k_e2e / n_pos) if n_pos > 0 else 0.0,
        'end_to_end_ci': (lo, hi),
    }


def brier_score(y_true, p):
    y = np.asarray(y_true, dtype=float)
    pp = np.asarray(p, dtype=float)
    if len(y) == 0:
        return float('nan')
    return float(np.mean((pp - y) ** 2))


def reliability_table(y_true, p, n_bins=5):
    y = np.asarray(y_true, dtype=float)
    pp = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pp, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        rows.append({
            'Корзина p_mean': f"{edges[b]:.1f}–{edges[b + 1]:.1f}",
            'Средняя оценка': f"{pp[m].mean():.3f}" if n > 0 else '-',
            'Доля класса 1': f"{y[m].mean():.3f}" if n > 0 else '-',
            'Число сигналов': n,
        })
    return pd.DataFrame(rows)


def oriented_auc(y_true, score, direction):
    y = np.asarray(y_true)
    sc = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    if direction == '<':
        sc = -sc
    return float(roc_auc_score(y, sc))


def diagnostic_ablation_sets(feature_cols):
    configs = []
    skipped = []
    present_groups = []
    for name, title, cols in DIAGNOSTIC_GROUPS:
        present = [c for c in cols if c in feature_cols]
        present_groups.append(present)
        if not present:
            skipped.append(f"«{name}»: пропущена, {title} не входят в набор признаков модели")
            continue
        configs.append((name + ' (' + ', '.join(present) + ')', [c for c in feature_cols if c not in present]))
    if all(present_groups):
        both = [c for grp in present_groups for c in grp]
        configs.append((BOTH_DIAGNOSTIC_NAME, [c for c in feature_cols if c not in both]))
    else:
        skipped.append(f"«{BOTH_DIAGNOSTIC_NAME}»: пропущена, так как хотя бы одна из двух групп пуста")
    return configs, skipped


def ablation_feature_sets(feature_cols):
    no_star = [c for c in feature_cols if c not in STAR_FEATURE_COLS]
    no_geom = [c for c in feature_cols if c not in GEOMETRY_COLS]
    diag_configs, _ = diagnostic_ablation_sets(feature_cols)
    return [
        ('Без звёздных признаков (' + ', '.join(c for c in STAR_FEATURE_COLS if c in feature_cols) + ')', no_star),
        ('Без transit_count и gap_fraction', no_geom),
    ] + diag_configs


def ablation_skipped_notes(feature_cols):
    return diagnostic_ablation_sets(feature_cols)[1]


def abstention_masks(p_mean, p_std, t_high, t_low, s_max):
    pm = np.asarray(p_mean, dtype=float)
    ps = np.asarray(p_std, dtype=float)
    cand = (pm >= t_high) & (ps <= s_max)
    fp = (pm <= t_low) & (ps <= s_max)
    return cand, cand | fp


def abstention_counts(y_true, p_mean, p_std, t_high, t_low, s_max):
    y = np.asarray(y_true).astype(int)
    cand, accepted = abstention_masks(p_mean, p_std, t_high, t_low, s_max)
    n_acc = int(np.sum(accepted))
    n_ok = int(np.sum(cand[accepted].astype(int) == y[accepted])) if n_acc > 0 else 0
    return {'n': int(len(y)), 'n_accepted': n_acc, 'n_correct': n_ok}


def select_abstention_thresholds(y_true, oof_mean, oof_std, target_precision=0.90, criterion='point', default=(0.75, 0.25, 0.35)):
    if criterion not in ('point', 'wilson'):
        raise ValueError(f"unknown threshold criterion: {criterion}")
    y = np.asarray(y_true).astype(int)
    pm = np.asarray(oof_mean, dtype=float)
    ps = np.asarray(oof_std, dtype=float)
    grid_th = np.linspace(0.50, 0.90, 9)
    grid_tl = np.linspace(0.50, 0.10, 9)
    grid_sm = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0]
    best_cov = -1.0
    best_acc = 0.0
    best_thres = tuple(default)
    highest_acc = -1.0
    fallback_thres = None
    for s_m in grid_sm:
        for t_h in grid_th:
            for t_l in grid_tl:
                c_mask = (pm >= t_h) & (ps <= s_m)
                f_mask = (pm <= t_l) & (ps <= s_m)
                acc_mask = c_mask | f_mask
                n_acc = int(np.sum(acc_mask))
                if n_acc < 2:
                    continue
                n_ok = int(np.sum(np.where(c_mask[acc_mask], 1, 0) == y[acc_mask]))
                acc_val = float(n_ok / n_acc)
                cov_val = float(np.mean(acc_mask))
                if acc_val > highest_acc:
                    highest_acc = acc_val
                    fallback_thres = (float(t_h), float(t_l), float(s_m))
                if criterion == 'wilson':
                    passed = wilson_interval(n_ok, n_acc)[0] >= target_precision
                else:
                    passed = acc_val >= target_precision
                if passed and (cov_val > best_cov or (cov_val == best_cov and acc_val > best_acc)):
                    best_cov = cov_val
                    best_acc = acc_val
                    best_thres = (float(t_h), float(t_l), float(s_m))
    if best_cov >= 0:
        branch = 'main'
    elif fallback_thres is not None:
        branch = 'fallback'
        best_thres = fallback_thres
    else:
        branch = 'default'
    cnt = abstention_counts(y, pm, ps, *best_thres)
    n_acc, n_ok = cnt['n_accepted'], cnt['n_correct']
    return best_thres, {
        'trained': True,
        'criterion': criterion,
        'target_precision': float(target_precision),
        'branch': branch,
        'thresholds': best_thres,
        'n_oof': int(len(y)),
        'oof_n_accepted': n_acc,
        'oof_n_correct': n_ok,
        'oof_coverage': float(n_acc / len(y)) if len(y) > 0 else 0.0,
        'oof_precision': float(n_ok / n_acc) if n_acc > 0 else float('nan'),
        'oof_precision_ci': wilson_interval(n_ok, n_acc),
    }


def coverage_precision_table(samples, s_max_selected, t_grid=SYMMETRIC_T_GRID):
    s_values = [1.0]
    if s_max_selected is not None and np.isfinite(s_max_selected) and abs(float(s_max_selected) - 1.0) > 1e-12:
        s_values.append(float(s_max_selected))
    rows = []
    for name, y_true, p_mean, p_std in samples:
        for s_m in s_values:
            for t in t_grid:
                cnt = abstention_counts(y_true, p_mean, p_std, float(t), round(1.0 - float(t), 10), s_m)
                n, n_acc, n_ok = cnt['n'], cnt['n_accepted'], cnt['n_correct']
                rows.append({
                    'Выборка': name,
                    's_max': f"{s_m:.2f}",
                    't_high / t_low': f"{t:.2f} / {1.0 - t:.2f}",
                    'Сигналов': n,
                    'Принято': n_acc,
                    'Охват': f"{n_acc / n:.1%}" if n > 0 else '-',
                    'Точность на принятых': f"{n_ok / n_acc:.3f} ({n_ok}/{n_acc})" if n_acc > 0 else '-',
                    '95% ДИ Уилсона': ci_text(wilson_interval(n_ok, n_acc)) if n_acc > 0 else '-',
                })
    return pd.DataFrame(rows)


def threshold_selection_line(info):
    if not info or not info.get('trained'):
        return 'Подбор порогов не выполнялся: модель не обучена.'
    th = info['thresholds']
    n_acc = info['oof_n_accepted']
    if n_acc > 0:
        prec = f"{info['oof_precision']:.3f} ({info['oof_n_correct']}/{n_acc}, 95% ДИ Уилсона {ci_text(info['oof_precision_ci'])})"
    else:
        prec = 'не определена (принятых сигналов нет)'
    return (f"OOF-сигналов: {info['n_oof']}; при выбранных порогах (t_high = {th[0]:.2f}, t_low = {th[1]:.2f}, s_max = {th[2]:.2f}) "
            f"OOF-охват: {info['oof_coverage']:.1%} ({n_acc} из {info['n_oof']}); OOF-точность на принятых: {prec}; "
            f"цель: {info['target_precision']:.2f}; критерий: {CRITERION_LABELS_RU.get(info['criterion'], info['criterion'])}; "
            f"сработавшая ветка: {BRANCH_LABELS_RU.get(info['branch'], info['branch'])}.")


def ci_text(ci, digits=3):
    return f"[{ci[0]:.{digits}f}; {ci[1]:.{digits}f}]"


def inverted_star_rates(groups, controlled_ids, inverted_counts):
    rows = []
    controlled = set(str(s) for s in controlled_ids)
    for name, stars in groups:
        pool = [str(s) for s in stars if str(s) in controlled]
        n = len(pool)
        k = sum(1 for s in pool if int(inverted_counts.get(s, 0)) > 0)
        n_sig = int(sum(int(inverted_counts.get(s, 0)) for s in pool))
        rows.append({
            'Группа звёзд': name,
            'Звёзд с контролем': n,
            'Звёзд с INVERTED-сигналом': format_share(k, n),
            'INVERTED-сигналов всего': n_sig,
            'На звезду': f"{n_sig / n:.3f}" if n > 0 else '-',
        })
    return pd.DataFrame(rows)
