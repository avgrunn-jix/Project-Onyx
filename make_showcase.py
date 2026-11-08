import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import onyx_console
onyx_console.setup_warnings()

from onyx_showcase import match_verdicts, render_showcase

STAR_ID = '4852528'
FLATTEN_DAYS = 3.0
MAX_SIGNALS = 4
FREQ_FACTOR = 5000
DUR_GRID = (0.05, 0.40, 4)


def load_extractor():
    spec = importlib.util.spec_from_file_location('onyx_extractor', str(ROOT / '1_data_extractor.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['onyx_extractor'] = mod
    spec.loader.exec_module(mod)
    return mod


def read_star_signals(features_file, star_id=STAR_ID, max_signals=MAX_SIGNALS):
    df = pd.read_csv(features_file, dtype={'target_id': str})
    df = df[df['target_id'].str.strip() == str(star_id)].copy()
    if 'signal_index' in df.columns:
        df = df.sort_values('signal_index')
    return df.head(max_signals).reset_index(drop=True)


def read_scored_signals(signals_file, star_id=STAR_ID):
    if not signals_file or not os.path.isfile(signals_file):
        return None
    df = pd.read_csv(signals_file, dtype={'target_id': str})
    if 'target_id' not in df.columns:
        return None
    return df[df['target_id'].str.strip() == str(star_id)].reset_index(drop=True)


def build_curves(extractor, star_id, cache_dir, clip_upward=False):
    lk = extractor.lk
    sr = lk.search_lightcurve(f'KIC {star_id}', mission='Kepler', author='Kepler', cadence='long')
    if len(sr) == 0:
        raise RuntimeError(f'No light curves found for KIC {star_id}')
    if 'target_name' in sr.table.colnames:
        digits = str(star_id).lstrip('0')
        mask = [digits in str(n).lstrip('0') for n in sr.table['target_name']]
        if any(mask):
            sr = sr[mask]
    lc_col = sr.download_all(download_dir=cache_dir)
    filtered = []
    for c_lc in lc_col:
        exp_val = getattr(c_lc, 'exptime', None)
        if exp_val is None and hasattr(c_lc, 'meta') and isinstance(c_lc.meta, dict):
            exp_val = c_lc.meta.get('EXPTIME')
        if exp_val is None or exp_val >= 1700.0:
            filtered.append(c_lc)
    if len(filtered) > 0:
        lc_col = lk.LightCurveCollection(filtered)
    n_files = len(lc_col)
    quarters = set()
    for c_lc in lc_col:
        q_val = getattr(c_lc, 'quarter', None)
        if q_val is None and hasattr(c_lc, 'meta') and isinstance(c_lc.meta, dict):
            q_val = c_lc.meta.get('QUARTER')
        if q_val is not None:
            quarters.add(q_val)
    n_quarters = len(quarters) if quarters else n_files

    extractor.detrend_centroids(lc_col, FLATTEN_DAYS)
    lc = lc_col.stitch(corrector_func=lambda x: x.normalize())
    lc = extractor.clip_outliers(lc.remove_nans(), upward_only=clip_upward)
    times = np.asarray(lc.time.value, dtype=float)
    baseline_d = float(np.nanmax(times) - np.nanmin(times))
    med_diff = np.nanmedian(np.diff(times))
    if med_diff > 0:
        wl = int(FLATTEN_DAYS / med_diff)
        if wl % 2 == 0:
            wl -= 1
        if wl < 3:
            wl = 3
    else:
        wl = 101
    if wl > len(lc):
        wl = len(lc) if len(lc) % 2 != 0 else len(lc) - 1
    lc_flat = lc.flatten(window_length=wl, polyorder=2, break_tolerance=5)
    max_p = min(100.0, baseline_d / 2.0)
    dur_grid = np.linspace(*DUR_GRID)
    bls = lc_flat.to_periodogram(method='bls', minimum_period=0.5, maximum_period=max_p, frequency_factor=FREQ_FACTOR, duration=dur_grid)
    power = np.asarray(bls.power.value, dtype=float)
    std = float(np.nanstd(power))
    sde_curve = (power - float(np.nanmean(power))) / (std if std > 0 else 1.0)
    return {
        'times_raw': times,
        'flux_raw': np.asarray(lc.flux.value, dtype=float),
        'times_flat': np.asarray(lc_flat.time.value, dtype=float),
        'flux_flat': np.asarray(lc_flat.flux.value, dtype=float),
        'bls_period': np.asarray(bls.period.value, dtype=float),
        'sde_curve': sde_curve,
        'n_files': n_files,
        'n_quarters': n_quarters,
        'baseline_days': baseline_d,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--signals-file', default='')
    parser.add_argument('--cache-dir', default=tempfile.gettempdir())
    parser.add_argument('--output', default=os.path.join('outputs', 'showcase_kepler80_wide.png'))
    parser.add_argument('--layout', choices=['tall', 'wide'], default='wide')
    parser.add_argument('--lang', choices=['ru', 'en'], default='ru')
    parser.add_argument('--no-resolve-aliases', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extractor = load_extractor()
    extractor.CLIP_UPWARD_ONLY = False
    extractor.RESOLVE_ALIASES = not args.no_resolve_aliases

    signals = read_star_signals(args.features_file)
    if len(signals) == 0:
        print(f'No rows for star {STAR_ID} in {args.features_file}', file=sys.stderr)
        return 1
    scored = read_scored_signals(args.signals_file)
    signals = match_verdicts(signals, scored)

    data = build_curves(extractor, STAR_ID, args.cache_dir)
    data['showcase_id'] = STAR_ID
    data['signals'] = signals
    render_showcase(data, args.output, layout=args.layout, lang=args.lang)
    print(f'Saved {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
