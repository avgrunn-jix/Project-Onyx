import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEARCH_MIN_PERIOD_D = 0.5
SEARCH_MAX_PERIOD_D = 100.0
PERIOD_TOL = 0.01
PERIOD_BINS = [(0.5, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, 100.0)]
POSITIVE_LABELS = ('CONFIRMED', 'CANDIDATE')
STAR_FEATURE_COLS = [
    'star_teff', 'star_radius', 'star_logg', 'flux_std_ppm', 'flux_mad_ppm', 'flux_skew',
    'flux_kurtosis', 'cdpp_ppm', 'pointwise_scatter_ppm', 'autocorr_max', 'ls_peak_power',
]
GEOMETRY_COLS = ['transit_count', 'gap_fraction']


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
        rows.append({'target_id': tid, 'disposition': disp, 'catalog_period': cat_p, 'in_range': in_range,
                     'status': status, 'found_period': found_p, 'decision': decision})
    return pd.DataFrame(rows, columns=['target_id', 'disposition', 'catalog_period', 'in_range', 'status', 'found_period', 'decision'])


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
    pos = in_rng[in_rng['disposition'].isin(POSITIVE_LABELS)]
    n_pos = len(pos)
    k_e2e = int(((pos['status'] == 'точно') & (pos['decision'] == 'CANDIDATE')).sum())
    k_found = int((pos['status'] == 'точно').sum())
    lo, hi = wilson_interval(k_e2e, n_pos)
    return {
        'by_disposition': pd.DataFrame(by_disp),
        'by_period_bin': pd.DataFrame(by_bin),
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


def ablation_feature_sets(feature_cols):
    no_star = [c for c in feature_cols if c not in STAR_FEATURE_COLS]
    no_geom = [c for c in feature_cols if c not in GEOMETRY_COLS]
    return [
        ('Без звёздных признаков (' + ', '.join(c for c in STAR_FEATURE_COLS if c in feature_cols) + ')', no_star),
        ('Без transit_count и gap_fraction', no_geom),
    ]


def ci_text(ci, digits=3):
    return f"[{ci[0]:.{digits}f}; {ci[1]:.{digits}f}]"
