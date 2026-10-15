import argparse
import datetime
import logging
import os
import shutil
import sys
import tempfile
import time
import warnings

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import scipy.stats as st
import astropy.units as u
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import concurrent.futures

warnings.filterwarnings('ignore')
logging.getLogger('lightkurve').setLevel(logging.ERROR)
logging.getLogger('astroquery').setLevel(logging.ERROR)

def setup_logger():
    os.makedirs('logs', exist_ok=True)
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(os.path.join('logs', 'extractor_errors.log'), encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger = logging.getLogger('extractor')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def get_dir_size(path):
    total = 0
    if not os.path.exists(path): return total
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += get_dir_size(entry.path)
    except Exception:
        pass
    return total

def fetch_labels(mission):
    os.makedirs(os.path.join('data', 'labels'), exist_ok=True)
    if mission == 'Kepler':
        labels_table = NasaExoplanetArchive.query_criteria(table='cumulative', select='kepid,koi_disposition,koi_period,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec')
        df = labels_table.to_pandas()
        df.rename(columns={'kepid': 'target_id', 'koi_disposition': 'label', 'koi_period': 'catalog_period'}, inplace=True)
        df['target_id'] = df['target_id'].astype(str)
        df.to_csv(os.path.join('data', 'labels', 'kepler_labels.csv'), index=False, encoding='utf-8')
        return df
    else:
        labels_table = NasaExoplanetArchive.query_criteria(table='toi', select='tic_id,tfopwg_disp,pl_orbper')
        df = labels_table.to_pandas()
        df.rename(columns={'tic_id': 'target_id', 'tfopwg_disp': 'label', 'pl_orbper': 'catalog_period'}, inplace=True)
        df['target_id'] = df['target_id'].astype(str)
        df.to_csv(os.path.join('data', 'labels', 'tess_labels.csv'), index=False, encoding='utf-8')
        return df

def read_targets_file(filepath):
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath, 'rb') as f:
        raw = f.read()
    if not raw:
        return []
    encodings = ['utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'utf-8', 'latin-1']
    for enc in encodings:
        try:
            text = raw.decode(enc)
            lines = [l.strip().replace('\x00', '') for l in text.splitlines()]
            cleaned = [l for l in lines if l]
            if cleaned:
                return cleaned
        except Exception:
            continue
    return []

def bin_folded_curve(time_array, flux_array, bins=201, range_min=-0.5, range_max=0.5):
    bins_edges = np.linspace(range_min, range_max, bins + 1)
    binned_flux = np.zeros(bins)
    for i in range(bins):
        mask = (time_array >= bins_edges[i]) & (time_array < bins_edges[i+1])
        if np.any(mask):
            binned_flux[i] = np.nanmedian(flux_array[mask])
        else:
            binned_flux[i] = np.nan
    nan_mask = np.isnan(binned_flux)
    if np.any(nan_mask) and not np.all(nan_mask):
        valid_idx = np.where(~nan_mask)[0]
        binned_flux[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, binned_flux[valid_idx])
    elif np.all(nan_mask):
        binned_flux[:] = 1.0
    return binned_flux

def inject_transit(time, flux, period, epoch, rp_rs, duration):
    phase = ((time - epoch) % period) / period
    phase[phase > 0.5] -= 1.0
    time_from_mid = phase * period
    half_dur = duration / 2.0
    ingress = max(duration * 0.1, 0.001)
    depth = (rp_rs)**2
    model = np.ones_like(flux)
    full_mask = np.abs(time_from_mid) <= (half_dur - ingress)
    model[full_mask] = 1.0 - depth
    ing_mask = (np.abs(time_from_mid) > (half_dur - ingress)) & (np.abs(time_from_mid) <= half_dur)
    if np.any(ing_mask):
        dist = half_dur - np.abs(time_from_mid[ing_mask])
        model[ing_mask] = 1.0 - depth * (dist / ingress)
    return flux * model

def detrend_centroids(lc_col, flatten_days):
    for seg in lc_col:
        seg_times = np.asarray(seg.time.value, dtype=float)
        dt_seg = np.nanmedian(np.diff(seg_times)) if len(seg_times) > 1 else 0.0
        wl_seg = int(flatten_days / dt_seg) if dt_seg > 0 else 101
        if wl_seg % 2 == 0: wl_seg -= 1
        if wl_seg < 3: wl_seg = 3
        if wl_seg > len(seg): wl_seg = len(seg) if len(seg) % 2 != 0 else len(seg) - 1

        has_col = 'centroid_col' in seg.columns and not np.all(np.isnan(seg.centroid_col.value))
        if has_col and wl_seg >= 3:
            try:
                c_dummy = lk.LightCurve(time=seg.time, flux=seg.centroid_col.value)
                _, c_trend = c_dummy.flatten(window_length=wl_seg, polyorder=2, break_tolerance=5, return_trend=True)
                seg['centroid_col_detrended'] = u.Quantity(seg.centroid_col.value - c_trend.flux.value, u.pixel)
            except Exception:
                seg['centroid_col_detrended'] = u.Quantity(np.full(len(seg), np.nan), u.pixel)
        else:
            seg['centroid_col_detrended'] = u.Quantity(np.full(len(seg), np.nan), u.pixel)

        has_row = 'centroid_row' in seg.columns and not np.all(np.isnan(seg.centroid_row.value))
        if has_row and wl_seg >= 3:
            try:
                r_dummy = lk.LightCurve(time=seg.time, flux=seg.centroid_row.value)
                _, r_trend = r_dummy.flatten(window_length=wl_seg, polyorder=2, break_tolerance=5, return_trend=True)
                seg['centroid_row_detrended'] = u.Quantity(seg.centroid_row.value - r_trend.flux.value, u.pixel)
            except Exception:
                seg['centroid_row_detrended'] = u.Quantity(np.full(len(seg), np.nan), u.pixel)
        else:
            seg['centroid_row_detrended'] = u.Quantity(np.full(len(seg), np.nan), u.pixel)

def compute_centroid_shift(times, col_det, row_det, period, epoch, duration, mission):
    phase_c = ((times - epoch + period / 2.0) % period) - period / 2.0
    in_c_mask = np.abs(phase_c) <= (duration / 2.0)
    out_c_mask = (np.abs(phase_c) > (duration / 2.0)) & (np.abs(phase_c) <= (3.0 * duration))
    valid_c = ~np.isnan(col_det) & ~np.isnan(row_det)
    in_c_pts = np.where(in_c_mask & valid_c)[0]
    out_c_pts = np.where(out_c_mask & valid_c)[0]
    if len(in_c_pts) >= 3 and len(out_c_pts) >= 3:
        in_col_vals = col_det[in_c_pts]
        out_col_vals = col_det[out_c_pts]
        in_row_vals = row_det[in_c_pts]
        out_row_vals = row_det[out_c_pts]
        d_col = float(np.nanmean(in_col_vals) - np.nanmean(out_col_vals))
        d_row = float(np.nanmean(in_row_vals) - np.nanmean(out_row_vals))
        s_col = float(np.nanstd(out_col_vals, ddof=1))
        s_row = float(np.nanstd(out_row_vals, ddof=1))
        se_col = s_col / np.sqrt(len(in_c_pts)) if s_col > 0 else 0.0
        se_row = s_row / np.sqrt(len(in_c_pts)) if s_row > 0 else 0.0
        c_shift_col_sig = float(abs(d_col) / se_col) if se_col > 0 else 0.0
        c_shift_row_sig = float(abs(d_row) / se_row) if se_row > 0 else 0.0
        c_shift_tot_sig = float(np.sqrt(c_shift_col_sig**2 + c_shift_row_sig**2))
        px_scale = 3.98 if mission == 'Kepler' else 21.0
        c_shift_tot_arcsec = float(np.sqrt(d_col**2 + d_row**2) * px_scale)
        return c_shift_col_sig, c_shift_row_sig, c_shift_tot_sig, c_shift_tot_arcsec
    return np.nan, np.nan, np.nan, np.nan

def run_injection(tid, mission, injections_per_star, min_free_gb, flatten_days, quick_mode=False, cache_dir=None, star_timeout=1800.0, max_sectors=0):
    free_bytes = shutil.disk_usage(os.path.abspath('.')).free
    if free_bytes < min_free_gb * 1024**3: return None, tid, 'Disk space below threshold'
    start_t = time.time()
    deadline = start_t + star_timeout
    base_cache = cache_dir if cache_dir else tempfile.gettempdir()
    tmp_dir = os.path.join(base_cache, f'onyx_inj_{tid}_{os.getpid()}_{int(time.time()*1000)}')
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        author = 'Kepler' if mission == 'Kepler' else 'SPOC'
        search_target = f"TIC {tid}" if mission == 'TESS' and not str(tid).startswith('TIC') else tid
        for attempt in range(3):
            if time.time() > deadline:
                return None, tid, f'TimeoutError: Exceeded {star_timeout}s'
            try:
                search_result = lk.search_lightcurve(search_target, author=author, cadence='long')
                break
            except Exception:
                time.sleep(2 ** attempt)
        else: return None, tid, 'Download failed'
        if len(search_result) == 0: return None, tid, 'No lightcurves found'
        if max_sectors > 0 and len(search_result) > max_sectors:
            search_result = search_result[:max_sectors]
        if time.time() > deadline:
            return None, tid, f'TimeoutError: Exceeded {star_timeout}s'
        lc_col = search_result.download_all(download_dir=tmp_dir)
        if lc_col is None or len(lc_col) == 0: return None, tid, 'No lightcurves found'
        detrend_centroids(lc_col, flatten_days)
        lc = lc_col.stitch(corrector_func=lambda x: x.normalize())
        lc = lc.remove_nans().remove_outliers(sigma=5)
        if len(lc) < 500: return None, tid, 'Not enough points'
        times = np.asarray(lc.time.value, dtype=float)
        flux_val = np.asarray(lc.flux.value, dtype=float)
        col_det = np.asarray(lc['centroid_col_detrended'].value, dtype=float) if 'centroid_col_detrended' in lc.columns else np.full(len(lc), np.nan)
        row_det = np.asarray(lc['centroid_row_detrended'].value, dtype=float) if 'centroid_row_detrended' in lc.columns else np.full(len(lc), np.nan)
        baseline = float(np.nanmax(times) - np.nanmin(times))
        star_radius = float(lc.meta.get('RADIUS', 1.0)) if lc.meta.get('RADIUS') is not None and not np.isnan(lc.meta.get('RADIUS')) else 1.0
        results = []
        features = []
        freq_factor = 5000 if quick_mode else 500
        dur_grid = np.linspace(0.05, 0.40, 4) if quick_mode else np.linspace(0.04, 0.45, 10)
        for inj_idx in range(injections_per_star):
            if time.time() > deadline:
                break
            inj_period = float(np.random.uniform(1.0, 30.0))
            inj_rad_earth = float(np.random.uniform(0.8, 15.0))
            rp_rs = (inj_rad_earth * 0.009168) / star_radius
            duration = float(np.random.uniform(0.05, 0.3))
            epoch = float(np.random.uniform(times[0], times[0] + inj_period))
            b = float(np.random.uniform(0.0, 0.8))
            inj_flux = inject_transit(times, flux_val, inj_period, epoch, rp_rs, duration)
            inj_lc = lk.LightCurve(time=lc.time, flux=inj_flux)
            med_diff = np.nanmedian(np.diff(times))
            wl = int(flatten_days / med_diff) if med_diff > 0 else 101
            if wl % 2 == 0: wl -= 1
            if wl < 3: wl = 3
            if wl > len(inj_lc): wl = len(inj_lc) if len(inj_lc) % 2 != 0 else len(inj_lc) - 1
            inj_flat = inj_lc.flatten(window_length=wl, polyorder=2, break_tolerance=5)
            bls = inj_flat.to_periodogram(method='bls', minimum_period=0.5, maximum_period=min(100.0, baseline/2.0), frequency_factor=freq_factor, duration=dur_grid)
            if len(bls.power) == 0: continue
            rec_period = float(bls.period_at_max_power.value)
            rec_epoch = float(bls.transit_time_at_max_power.value)
            p_vals = np.array(bls.power.value)
            sde = float((np.nanmax(p_vals) - np.nanmean(p_vals)) / np.nanstd(p_vals))
            snr = float(np.nanmax(bls.snr))
            recovered = 0
            alias_type = 'none'
            ratio = rec_period / inj_period
            def epoch_match(e1, e2, p, dur):
                diff = abs((e1 - e2) % p)
                diff = min(diff, p - diff)
                return diff < (dur / 2.0)
            if abs(ratio - 1.0) < 0.01 and epoch_match(rec_epoch, epoch, inj_period, duration):
                recovered = 1
            elif (abs(ratio - 0.5) < 0.01 or abs(ratio - 2.0) < 0.01 or abs(ratio - 3.0) < 0.01) and epoch_match(rec_epoch, epoch, inj_period, duration):
                alias_type = 'alias'
            c_col_sig, c_row_sig, c_tot_sig, c_tot_arcsec = compute_centroid_shift(times, col_det, row_det, inj_period, epoch, duration, mission)
            res = {
                'target_id': tid, 'period_d': inj_period, 'radius_earth': inj_rad_earth,
                'depth_ppm': (rp_rs**2)*1e6, 'duration_d': duration, 'impact_b': b,
                'recovered': recovered, 'alias_type': alias_type,
                'bls_sde': sde, 'bls_depth_snr': snr, 'star_radius': star_radius,
                'centroid_shift_col_sigma': c_col_sig,
                'centroid_shift_row_sigma': c_row_sig,
                'centroid_shift_total_sigma': c_tot_sig,
                'centroid_shift_total_arcsec': c_tot_arcsec
            }
            results.append(res)
            if recovered:
                rem_timeout = max(10.0, deadline - time.time())
                feat, _, _, _, _, _ = process_single_star(tid, mission, [{'catalog_period': inj_period, 'label': 'INJECTED'}], True, 0, flatten_days, 1, quick_mode=quick_mode, cache_dir=cache_dir, star_timeout=rem_timeout, max_sectors=max_sectors)
                if feat and len(feat) > 0:
                    feat[0]['centroid_shift_col_sigma'] = c_col_sig
                    feat[0]['centroid_shift_row_sigma'] = c_row_sig
                    feat[0]['centroid_shift_total_sigma'] = c_tot_sig
                    feat[0]['centroid_shift_total_arcsec'] = c_tot_arcsec
                    features.append(feat[0])
        return results, features, tid, None
    except Exception as e:
        return None, None, tid, str(type(e).__name__) + ' - ' + str(e)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

def process_single_star(tid, mission, row_dicts, keep_raw, min_free_gb, flatten_days, max_signals, quick_mode=False, cache_dir=None, star_timeout=1800.0, max_sectors=0):
    free_bytes = shutil.disk_usage(os.path.abspath('.')).free
    if free_bytes < min_free_gb * 1024**3:
        return None, None, tid, 'Disk space below threshold', 0, None
    start_t = time.time()
    deadline = start_t + star_timeout
    base_cache = cache_dir if cache_dir else tempfile.gettempdir()
    tmp_dir = os.path.join(base_cache, f'onyx_raw_{tid}_{os.getpid()}_{int(time.time()*1000)}')
    os.makedirs(tmp_dir, exist_ok=True)
    dl_size = 0
    try:
        t_dl_start = time.time()
        author = 'Kepler' if mission == 'Kepler' else 'SPOC'
        search_target = f"TIC {tid}" if mission == 'TESS' and not str(tid).startswith('TIC') else tid
        for attempt in range(3):
            if time.time() > deadline:
                return None, None, tid, f'TimeoutError: Exceeded {star_timeout}s', dl_size, None
            try:
                search_result = lk.search_lightcurve(search_target, author=author, cadence='long')
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            return None, None, tid, 'Download failed', dl_size, None
        if len(search_result) == 0:
            return None, None, tid, 'No lightcurves found', dl_size, None
        if max_sectors > 0 and len(search_result) > max_sectors:
            search_result = search_result[:max_sectors]
        if time.time() > deadline:
            return None, None, tid, f'TimeoutError: Exceeded {star_timeout}s', dl_size, None
        lc_col = search_result.download_all(download_dir=tmp_dir)
        dl_size = get_dir_size(tmp_dir)
        t_dl_end = time.time()
        timing_dl = t_dl_end - t_dl_start

        if lc_col is None or len(lc_col) == 0:
            return None, None, tid, 'No lightcurves found', dl_size, None

        t_stitch_start = time.time()
        detrend_centroids(lc_col, flatten_days)
        lc = lc_col.stitch(corrector_func=lambda x: x.normalize())
        lc = lc.remove_nans().remove_outliers(sigma=5)
        if len(lc) < 500:
            return None, None, tid, 'Not enough valid points', dl_size, None
        times = np.asarray(lc.time.value, dtype=float)
        baseline_d = float(np.nanmax(times) - np.nanmin(times))
        if baseline_d <= 0:
            return None, None, tid, 'Invalid baseline', dl_size, None
        med_diff = np.nanmedian(np.diff(times))
        if med_diff > 0:
            wl = int(flatten_days / med_diff)
            if wl % 2 == 0: wl -= 1
            if wl < 3: wl = 3
        else:
            wl = 101
        if wl > len(lc): wl = len(lc) if len(lc) % 2 != 0 else len(lc) - 1
        if wl < 3: return None, None, tid, 'Time series too short for flatten', dl_size, None
        if time.time() > deadline:
            return None, None, tid, f'TimeoutError: Exceeded {star_timeout}s', dl_size, None
        lc_flat = lc.flatten(window_length=wl, polyorder=2, break_tolerance=5)
        t_stitch_end = time.time()
        timing_stitch = t_stitch_end - t_stitch_start

        flux_flat = np.asarray(lc_flat.flux.value, dtype=float)
        flux_std_ppm = float(np.nanstd(flux_flat) * 1e6)
        flux_mad_ppm = float(np.nanmedian(np.abs(flux_flat - np.nanmedian(flux_flat))) * 1e6)
        flux_skew = float(st.skew(flux_flat, nan_policy='omit'))
        flux_kurtosis = float(st.kurtosis(flux_flat, nan_policy='omit'))
        cdpp_ppm = float(lc_flat.estimate_cdpp().value)
        pointwise_scatter_ppm = float(np.nanstd(np.diff(flux_flat)) * 1e6 / np.sqrt(2))
        acorr = np.correlate(flux_flat - np.nanmean(flux_flat), flux_flat - np.nanmean(flux_flat), mode='full')
        acorr = acorr[len(acorr)//2:]
        acorr_norm = acorr / acorr[0] if acorr[0] > 0 else acorr
        autocorr_max = float(np.nanmax(acorr_norm[1:min(100, len(acorr_norm))])) if len(acorr_norm) > 1 else 0.0
        max_p = min(100.0, baseline_d / 2.0)
        ls = lc_flat.to_periodogram(method='lombscargle', minimum_period=0.5, maximum_period=max_p, oversample_factor=10)
        ls_peak_power = float(np.nanmax(ls.power.value))
        gap_fraction = float(1.0 - (len(lc) / (baseline_d / med_diff))) if med_diff > 0 else 0.0
        star_teff = float(lc.meta.get('TEFF', np.nan)) if lc.meta.get('TEFF') is not None else np.nan
        star_radius = float(lc.meta.get('RADIUS', np.nan)) if lc.meta.get('RADIUS') is not None else np.nan
        star_logg = float(lc.meta.get('LOGG', np.nan)) if lc.meta.get('LOGG') is not None else np.nan
        lc_search = lc_flat.copy()
        features_list = []
        curves_dict = {}
        freq_factor = 5000 if quick_mode else 500
        dur_grid = np.linspace(0.05, 0.40, 4) if quick_mode else np.linspace(0.04, 0.45, 10)
        timing_bls = []
        timing_bls_details = []
        timing_feats = []

        for sig_idx in range(max_signals):
            if time.time() > deadline:
                break
            if max_p <= 0.5 or len(lc_search) < 100:
                break
            t_bls_start = time.time()
            bls = lc_search.to_periodogram(method='bls', minimum_period=0.5, maximum_period=max_p, frequency_factor=freq_factor, duration=dur_grid)
            t_bls_end = time.time()
            bls_iter_time = t_bls_end - t_bls_start
            timing_bls.append(bls_iter_time)
            if len(bls.power) == 0 or np.isnan(bls.power).all():
                timing_bls_details.append({
                    'iter': sig_idx + 1,
                    'time': bls_iter_time,
                    'n_freqs': len(bls.frequency) if hasattr(bls, 'frequency') else 0,
                    'period': 0.0,
                    'sde': 0.0,
                    'snr': 0.0
                })
                break
            power_values = np.array(bls.power.value)
            bls_power_peak = float(np.nanmax(power_values))
            bls_sde = float((bls_power_peak - np.nanmean(power_values)) / np.nanstd(power_values))
            bls_depth_snr = float(np.nanmax(bls.snr))
            bls_period_d = float(bls.period_at_max_power.value)
            timing_bls_details.append({
                'iter': sig_idx + 1,
                'time': bls_iter_time,
                'n_freqs': len(bls.frequency),
                'period': bls_period_d,
                'sde': bls_sde,
                'snr': bls_depth_snr
            })
            if bls_sde < 7.0 or bls_depth_snr < 7.1:
                break
            bls_period_d = float(bls.period_at_max_power.value)
            bls_epoch = float(bls.transit_time_at_max_power.value)
            bls_duration_d = float(bls.duration_at_max_power.value)
            bls_depth_ppm = float(bls.depth_at_max_power.value * 1e6)
            transit_count = int(baseline_d / bls_period_d)
            duty_cycle = float(bls_duration_d / bls_period_d)

            t_feat_start = time.time()
            folded = lc_search.fold(period=bls_period_d, epoch_time=bls_epoch, normalize_phase=False)
            phase_times = np.asarray(folded.time.value, dtype=float)
            fold_flux = np.asarray(folded.flux.value, dtype=float)
            norm_phase = phase_times / bls_period_d
            norm_phase = (norm_phase + 0.5) % 1.0 - 0.5
            binned_phase = bin_folded_curve(norm_phase, fold_flux)
            sig_id = f'{tid}_{sig_idx}'
            zoom_mask = np.abs(phase_times) <= (bls_duration_d * 2.0)
            if np.any(zoom_mask):
                zoom_phase = phase_times[zoom_mask] / (bls_duration_d * 2.0)
                zoom_phase = (zoom_phase + 0.5) % 1.0 - 0.5
                binned_zoom = bin_folded_curve(zoom_phase, fold_flux[zoom_mask])
            else:
                binned_zoom = np.ones(201)
            curves_dict[sig_id] = np.stack([binned_phase, binned_zoom])
            match_status = 'UNMATCHED'
            match_label = 'UNMATCHED'
            match_flags = {'koi_fpflag_nt': 0, 'koi_fpflag_ss': 0, 'koi_fpflag_co': 0, 'koi_fpflag_ec': 0}
            for r in row_dicts:
                cat_p = r.get('catalog_period', np.nan)
                if pd.isna(cat_p):
                    continue
                ratio = bls_period_d / float(cat_p)
                if abs(ratio - 1.0) < 0.01:
                    match_status = 'MATCHED'
                    match_label = r.get('label', 'UNMATCHED')
                    if mission == 'Kepler':
                        match_flags = {
                            'koi_fpflag_nt': r.get('koi_fpflag_nt', 0),
                            'koi_fpflag_ss': r.get('koi_fpflag_ss', 0),
                            'koi_fpflag_co': r.get('koi_fpflag_co', 0),
                            'koi_fpflag_ec': r.get('koi_fpflag_ec', 0)
                        }
                    break
                elif abs(ratio - 0.5) < 0.01 or abs(ratio - 2.0) < 0.01:
                    match_status = 'HARMONIC'
                    match_label = 'HARMONIC'
            folded_double = lc_search.fold(period=bls_period_d * 2, epoch_time=bls_epoch, normalize_phase=False)
            odd_mask = np.abs(folded_double.time.value) < (bls_duration_d / 2)
            even_mask = np.abs(folded_double.time.value - bls_period_d) < (bls_duration_d / 2)
            flux_odd = folded_double.flux.value[odd_mask]
            flux_even = folded_double.flux.value[even_mask]
            med_odd = np.nanmedian(flux_odd) if len(flux_odd) > 0 else 1.0
            med_even = np.nanmedian(flux_even) if len(flux_even) > 0 else 1.0
            odd_even_depth_diff_ppm = abs(med_odd - med_even) * 1e6
            combined_std = np.nanstd(folded_double.flux.value)
            odd_even_sigma = float(abs(med_odd - med_even) / combined_std) if combined_std > 0 else 0.0
            sec_mask = (np.abs(folded.time.value - bls_period_d/2) < (bls_duration_d / 2))
            flux_sec = folded.flux.value[sec_mask]
            sec_depth = 1.0 - np.nanmedian(flux_sec) if len(flux_sec) > 0 else 0.0
            secondary_depth_ppm = float(sec_depth * 1e6)
            secondary_snr = float(sec_depth / combined_std) if combined_std > 0 else 0.0
            idx_p = np.argmin(np.abs(bls.period.value - bls_period_d))
            idx_half = np.argmin(np.abs(bls.period.value - (bls_period_d / 2)))
            idx_double = np.argmin(np.abs(bls.period.value - (bls_period_d * 2)))
            power_p = power_values[idx_p]
            harmonic_power_ratio = float((power_values[idx_half] + power_values[idx_double]) / power_p) if power_p > 0 else 0.0
            in_trans_mask = np.abs(folded.time.value) < (bls_duration_d / 2)
            flux_in_trans = folded.flux.value[in_trans_mask]
            cent_mask = np.abs(folded.time.value) < (bls_duration_d / 4)
            flux_cent = folded.flux.value[cent_mask]
            depth_full = 1.0 - np.nanmean(flux_in_trans) if len(flux_in_trans) > 0 else 1e-9
            depth_cent = 1.0 - np.nanmean(flux_cent) if len(flux_cent) > 0 else 1e-9
            shape_v_u_ratio = float(depth_cent / depth_full)
            in_transit_scatter_ppm = float(np.nanstd(flux_in_trans) * 1e6) if len(flux_in_trans) > 0 else 0.0
            times_search = np.asarray(lc_search.time.value, dtype=float)
            flux_search = np.asarray(lc_search.flux.value, dtype=float)
            transit_depths = []
            for i in range(transit_count):
                mid = bls_epoch + i * bls_period_d
                mask = np.abs(times_search - mid) < (bls_duration_d / 2)
                if np.any(mask):
                    transit_depths.append(1.0 - np.nanmean(flux_search[mask]))
            depth_scatter_ppm = float(np.nanstd(transit_depths) * 1e6) if len(transit_depths) > 0 else 0.0

            col_det = np.asarray(lc_search['centroid_col_detrended'].value, dtype=float) if 'centroid_col_detrended' in lc_search.columns else np.full(len(lc_search), np.nan)
            row_det = np.asarray(lc_search['centroid_row_detrended'].value, dtype=float) if 'centroid_row_detrended' in lc_search.columns else np.full(len(lc_search), np.nan)
            c_shift_col_sig, c_shift_row_sig, c_shift_tot_sig, c_shift_tot_arcsec = compute_centroid_shift(times_search, col_det, row_det, bls_period_d, bls_epoch, bls_duration_d, mission)

            feat = {
                'signal_id': sig_id, 'target_id': tid, 'signal_index': sig_idx, 
                'mission': mission, 'label': match_label, 'match_status': match_status,
                'bls_period_d': bls_period_d, 'bls_epoch': bls_epoch, 'bls_duration_d': bls_duration_d,
                'bls_depth_ppm': bls_depth_ppm, 'bls_depth_snr': bls_depth_snr,
                'bls_power_peak': bls_power_peak, 'bls_sde': bls_sde, 'transit_count': transit_count,
                'duty_cycle': duty_cycle, 'odd_even_depth_diff_ppm': odd_even_depth_diff_ppm,
                'odd_even_sigma': odd_even_sigma, 'secondary_depth_ppm': secondary_depth_ppm,
                'secondary_snr': secondary_snr, 'harmonic_power_ratio': harmonic_power_ratio,
                'shape_v_u_ratio': shape_v_u_ratio, 'in_transit_scatter_ppm': in_transit_scatter_ppm,
                'depth_scatter_ppm': depth_scatter_ppm, 'flux_std_ppm': flux_std_ppm,
                'flux_mad_ppm': flux_mad_ppm, 'flux_skew': flux_skew, 'flux_kurtosis': flux_kurtosis,
                'cdpp_ppm': cdpp_ppm, 'pointwise_scatter_ppm': pointwise_scatter_ppm,
                'autocorr_max': autocorr_max, 'ls_peak_power': ls_peak_power, 'gap_fraction': gap_fraction,
                'baseline_d': baseline_d, 'star_teff': star_teff, 'star_radius': star_radius,
                'star_logg': star_logg,
                'centroid_shift_col_sigma': c_shift_col_sig,
                'centroid_shift_row_sigma': c_shift_row_sig,
                'centroid_shift_total_sigma': c_shift_tot_sig,
                'centroid_shift_total_arcsec': c_shift_tot_arcsec
            }
            if mission == 'Kepler':
                feat.update(match_flags)
            features_list.append(feat)
            t_feat_end = time.time()
            timing_feats.append(t_feat_end - t_feat_start)

            transit_mask = lc_search.create_transit_mask(period=bls_period_d, transit_time=bls_epoch, duration=bls_duration_d * 1.5)
            lc_search = lc_search[~transit_mask]

        for f in features_list:
            f['n_signals_star'] = len(features_list)
        timing_total = time.time() - start_t
        timings = {
            'dl': timing_dl,
            'stitch': timing_stitch,
            'bls': sum(timing_bls),
            'bls_list': timing_bls,
            'bls_details': timing_bls_details,
            'feats': sum(timing_feats),
            'total': timing_total,
            'n_quarters': len(lc_col) if 'lc_col' in locals() and lc_col is not None else 0,
            'baseline_d': baseline_d if 'baseline_d' in locals() else 0.0
        }
        return features_list, curves_dict, tid, None, dl_size, timings
    except Exception as e:
        return None, None, tid, str(type(e).__name__) + ' - ' + str(e), dl_size, None
    finally:
        if not keep_raw and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

def append_to_csv_flush(filepath, df_rows):
    if df_rows is None or len(df_rows) == 0:
        return
    hdr = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
    with open(filepath, 'a', encoding='utf-8', newline='') as f:
        df_rows.to_csv(f, header=hdr, index=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

def log_failure(failed_path, tid_str, err_msg, attempts_dict):
    att = attempts_dict.get(tid_str, 0) + 1
    attempts_dict[tid_str] = att
    df_err = pd.DataFrame([{'target_id': tid_str, 'error': str(err_msg), 'attempts': att}])
    append_to_csv_flush(failed_path, df_err)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission', default='Kepler', choices=['Kepler', 'TESS'])
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--output', default='extracted_features.csv')
    parser.add_argument('--targets-file', default='')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--keep-raw', action='store_true')
    parser.add_argument('--min-free-gb', type=float, default=5.0)
    parser.add_argument('--purge-lk-cache', action='store_true')
    parser.add_argument('--flatten-days', type=float, default=3.0)
    parser.add_argument('--max-signals', type=int, default=4)
    parser.add_argument('--inject', action='store_true')
    parser.add_argument('--injection-stars', type=int, default=100)
    parser.add_argument('--injections-per-star', type=int, default=12)
    parser.add_argument('--injection-output', default='outputs/injection_results.csv')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--star-timeout', type=float, default=1800.0)
    parser.add_argument('--cache-dir', default=tempfile.gettempdir())
    parser.add_argument('--max-sectors', type=int, default=0, help='[DEBUG ONLY] Limit the number of sectors/quarters downloaded. Warning: alters baseline, features, and detection results; not for production training.')
    parser.add_argument('--smoke-test', action='store_true', help='[TEST ONLY] Run extractor in isolated smoke test mode, writing outputs to outputs/_smoke_test/')
    args = parser.parse_args()
    if args.smoke_test and args.min_free_gb == 5.0:
        args.min_free_gb = 1.0
    logger = setup_logger()

    logical_cores = os.cpu_count() or 1
    cpu_msg = f"Detected CPU cores: {logical_cores} logical. Default workers: {args.workers}. Note: on systems with Hyper-Threading (SMT), the optimal number of worker processes for CPU-bound BLS computation is typically equal to the number of physical cores."
    print(cpu_msg, flush=True)
    logger.info(cpu_msg)

    base_outputs = os.path.join('outputs', '_smoke_test') if args.smoke_test else 'outputs'
    if args.smoke_test:
        if args.output == 'extracted_features.csv':
            args.output = os.path.join(base_outputs, 'extracted_features.csv')
        elif not '_smoke_test' in args.output:
            args.output = os.path.join(base_outputs, os.path.basename(args.output))
        if args.injection_output == 'outputs/injection_results.csv':
            args.injection_output = os.path.join(base_outputs, 'injection_results.csv')
        elif not '_smoke_test' in args.injection_output:
            args.injection_output = os.path.join(base_outputs, os.path.basename(args.injection_output))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs(base_outputs, exist_ok=True)
    os.makedirs(os.path.join(base_outputs, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(base_outputs, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_outputs, 'explanations'), exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.quick:
        logger.warning('WARNING: Quick mode enabled. Using coarse BLS grid (frequency_factor=5000, 4 duration steps). Outputs will have _quick suffix.')
        print('WARNING: Quick mode enabled. Using coarse BLS grid (frequency_factor=5000, 4 duration steps). Outputs will have _quick suffix.', flush=True)
        if not '_quick' in args.output:
            b_o, e_o = os.path.splitext(args.output)
            args.output = f"{b_o}_quick{e_o if e_o else '.csv'}"

    if args.max_sectors > 0:
        sec_warn = f"WARNING: [DEBUG MODE] --max-sectors={args.max_sectors} is active. This alters the observation baseline, features, and detection results. Output files MUST NOT be used for primary model training!"
        logger.warning(sec_warn)
        print(sec_warn, flush=True)
        if not f'_sec{args.max_sectors}' in args.output and not '_sec' in args.output:
            b_o, e_o = os.path.splitext(args.output)
            args.output = f"{b_o}_sec{args.max_sectors}{e_o if e_o else '.csv'}"

    base_out, ext_out = os.path.splitext(args.output)
    failed_csv = f"{base_out}_failed{ext_out if ext_out else '.csv'}"
    diag_csv = f"{base_out}_diagnostics{ext_out if ext_out else '.csv'}"
    resume = not args.no_resume

    if resume and os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        try:
            with open(args.output, 'r', encoding='utf-8') as f_chk:
                hdr_line = f_chk.readline().strip()
            exist_cols = [c.strip() for c in hdr_line.split(',') if c.strip()]
            required_cols = ['centroid_shift_col_sigma', 'centroid_shift_row_sigma', 'centroid_shift_total_sigma', 'centroid_shift_total_arcsec']
            missing_in_hdr = [c for c in required_cols if c not in exist_cols]
            if missing_in_hdr:
                err_msg = f"Schema mismatch in existing {args.output}! Existing file is missing current schema columns: {', '.join(missing_in_hdr)}. Use --no-resume to recompute with the new schema, or specify a different --output file."
                logger.error(err_msg)
                print(f"ERROR: {err_msg}", flush=True)
                sys.exit(1)
        except SystemExit:
            raise
        except Exception:
            pass

    if not resume:
        if os.path.exists(args.output):
            os.remove(args.output)
        if os.path.exists(failed_csv):
            os.remove(failed_csv)
        if os.path.exists(diag_csv):
            os.remove(diag_csv)
        npz_del = os.path.join(os.path.dirname(args.output) or base_outputs, 'folded_curves.npz')
        if os.path.exists(npz_del):
            os.remove(npz_del)

    def purge_cache():
        cache_dir = os.path.expanduser('~/.lightkurve/cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
    if args.purge_lk_cache:
        purge_cache()

    free_start = shutil.disk_usage(os.path.abspath('.')).free
    with open(os.path.join('logs', 'disk_usage.log'), 'a', encoding='utf-8') as f:
        f.write(f'Start: {free_start / 1024**3:.2f} GB free\n')

    labels_df = fetch_labels(args.mission)

    if args.inject:
        all_tids = labels_df['target_id'].unique().tolist()
        unmarked_stars = []
        if args.targets_file and os.path.exists(args.targets_file):
            tids_from_file = read_targets_file(args.targets_file)
            unmarked_stars = [tid for tid in tids_from_file if tid not in all_tids]
        else:
            logger.error("Please provide --targets-file with a list of target IDs for injection (which will be filtered to exclude KOIs).")
            return
        np.random.shuffle(unmarked_stars)
        unmarked_stars = unmarked_stars[:args.injection_stars]

        inj_out = args.injection_output
        if args.quick and not '_quick' in inj_out:
            b_i, e_i = os.path.splitext(inj_out)
            inj_out = f"{b_i}_quick{e_i if e_i else '.csv'}"

        if args.max_sectors > 0:
            sec_warn = f"WARNING: [DEBUG MODE] --max-sectors={args.max_sectors} is active in injection mode. This alters observation baselines. Outputs are for testing only!"
            logger.warning(sec_warn)
            print(sec_warn, flush=True)
            if not f'_sec{args.max_sectors}' in inj_out and not '_sec' in inj_out:
                b_i, e_i = os.path.splitext(inj_out)
                inj_out = f"{b_i}_sec{args.max_sectors}{e_i if e_i else '.csv'}"

        os.makedirs(os.path.dirname(inj_out) or '.', exist_ok=True)
        inj_feat_name = 'injection_features_quick.csv' if args.quick else 'injection_features.csv'
        inj_feat_out = os.path.join(base_outputs, inj_feat_name)

        if not resume:
            if os.path.exists(inj_out):
                os.remove(inj_out)
            if os.path.exists(inj_feat_out):
                os.remove(inj_feat_out)

        completed_inj_stars = set()
        if resume and os.path.exists(inj_out) and os.path.getsize(inj_out) > 0:
            try:
                df_prev_inj = pd.read_csv(inj_out, usecols=['target_id'], dtype=str)
                counts = df_prev_inj['target_id'].value_counts()
                for c_tid, count_val in counts.items():
                    if count_val >= args.injections_per_star:
                        completed_inj_stars.add(str(c_tid))
            except Exception:
                pass

        tasks = []
        for tid in unmarked_stars:
            tid_str = str(tid)
            if tid_str in completed_inj_stars:
                continue
            tasks.append((tid_str, args.mission, args.injections_per_star, args.min_free_gb, args.flatten_days, args.quick, args.cache_dir, args.star_timeout, args.max_sectors))

        total_inj = len(tasks)
        done_inj = 0
        durations_inj = []
        print(f"Injection tasks: {total_inj} stars to process (already completed: {len(completed_inj_stars)})", flush=True)
        logger.info(f"Injection tasks: {total_inj} stars to process (already completed: {len(completed_inj_stars)})")

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            task_iter = iter(tasks)
            active_futures = {}
            for _ in range(min(args.workers * 2, total_inj)):
                try:
                    t = next(task_iter)
                    f = executor.submit(run_injection, *t)
                    active_futures[f] = (t[0], time.time())
                except StopIteration:
                    break

            while active_futures:
                done, _ = concurrent.futures.wait(active_futures.keys(), timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED)
                now = time.time()
                for f in done:
                    tid_str, start_t = active_futures.pop(f)
                    elapsed = now - start_t
                    durations_inj.append(elapsed)
                    done_inj += 1
                    try:
                        res, feat, r_tid, err = f.result()
                    except Exception as exc:
                        err = f"{type(exc).__name__}: {str(exc)}"
                        res, feat = None, None

                    if err:
                        logger.error(f"Injection error on {tid_str}: {err}")
                    else:
                        if res:
                            append_to_csv_flush(inj_out, pd.DataFrame(res))
                        if feat:
                            append_to_csv_flush(inj_feat_out, pd.DataFrame(feat))

                    avg_t = sum(durations_inj) / len(durations_inj)
                    rem_stars = total_inj - done_inj
                    eta_sec = int((rem_stars * avg_t) / max(1, args.workers))
                    eta_str = str(datetime.timedelta(seconds=eta_sec))
                    p_msg = f"[{done_inj}/{total_inj}] Injection on {tid_str} ({elapsed:.1f}s) | Avg: {avg_t:.1f}s/star | ETA: {eta_str}"
                    print(p_msg, flush=True)
                    logger.info(p_msg)
                    sys.stdout.flush()

                    try:
                        next_t = next(task_iter)
                        next_f = executor.submit(run_injection, *next_t)
                        active_futures[next_f] = (next_t[0], time.time())
                    except StopIteration:
                        pass

                for f, (tid_str, start_t) in list(active_futures.items()):
                    if now - start_t > args.star_timeout:
                        active_futures.pop(f)
                        f.cancel()
                        done_inj += 1
                        durations_inj.append(args.star_timeout)
                        logger.error(f"Injection TIMEOUT on {tid_str}")
                        try:
                            next_t = next(task_iter)
                            next_f = executor.submit(run_injection, *next_t)
                            active_futures[next_f] = (next_t[0], time.time())
                        except StopIteration:
                            pass

        if os.path.exists(inj_out) and os.path.getsize(inj_out) > 0:
            df_res = pd.read_csv(inj_out)
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(10, 8))
            df_res['log_p'] = np.log10(df_res['period_d'])
            df_res['log_r'] = np.log10(df_res['radius_earth'])
            p_bins = np.linspace(np.log10(0.8), np.log10(100), 10)
            r_bins = np.linspace(np.log10(0.5), np.log10(15), 10)
            rec_grid = np.zeros((9, 9))
            tot_grid = np.zeros((9, 9))
            for i in range(9):
                for j in range(9):
                    mask = (df_res['log_p'] >= p_bins[i]) & (df_res['log_p'] < p_bins[i+1]) & (df_res['log_r'] >= r_bins[j]) & (df_res['log_r'] < r_bins[j+1])
                    tot = np.sum(mask)
                    rec = np.sum(df_res[mask]['recovered'])
                    tot_grid[j, i] = tot
                    rec_grid[j, i] = rec / tot if tot >= 5 else np.nan
            sns.heatmap(rec_grid, cmap='viridis', xticklabels=np.round(10**p_bins[:-1], 1), yticklabels=np.round(10**r_bins[:-1], 1))
            plt.xlabel('Period (days)')
            plt.ylabel('Radius (Earth)')
            title_str = 'Completeness Heatmap (Quick Mode)' if args.quick else 'Completeness Heatmap'
            plt.title(title_str)
            heatmap_name = 'injection_heatmap_quick.png' if args.quick else 'injection_heatmap.png'
            heatmap_out = os.path.join(base_outputs, 'figures', heatmap_name)
            plt.savefig(heatmap_out)
            plt.close()
        return

    target_ids = []
    if args.targets_file and os.path.exists(args.targets_file):
        target_ids = read_targets_file(args.targets_file)
    else:
        labels_df = labels_df.dropna(subset=['label'])
        target_ids = sorted(labels_df['target_id'].dropna().astype(str).unique().tolist())
    np.random.seed(42)
    np.random.shuffle(target_ids)
    target_ids = target_ids[:args.limit]

    processed_ids = set()
    if resume:
        if os.path.exists(diag_csv) and os.path.getsize(diag_csv) > 0:
            try:
                df_diag = pd.read_csv(diag_csv, usecols=['target_id', 'status'], dtype=str)
                processed_ids.update(df_diag[df_diag['status'] == 'OK']['target_id'].dropna().astype(str))
            except Exception:
                pass
        elif os.path.exists(args.output) and os.path.getsize(args.output) > 0:
            try:
                df_exist = pd.read_csv(args.output, usecols=['target_id'], dtype=str)
                processed_ids.update(df_exist['target_id'].dropna().astype(str))
            except Exception:
                pass

    failed_attempts = {}
    if resume and os.path.exists(failed_csv) and os.path.getsize(failed_csv) > 0:
        try:
            df_failed = pd.read_csv(failed_csv, dtype={'target_id': str, 'attempts': int})
            for _, r in df_failed.iterrows():
                tid_str = str(r['target_id'])
                failed_attempts[tid_str] = max(failed_attempts.get(tid_str, 0), int(r.get('attempts', 1)))
        except Exception:
            pass

    tasks = []
    for tid in target_ids:
        tid_str = str(tid)
        if tid_str in processed_ids:
            continue
        if tid_str in failed_attempts and failed_attempts[tid_str] >= 2:
            continue
        rows = labels_df[labels_df['target_id'] == tid_str].to_dict('records')
        tasks.append((tid_str, args.mission, rows, args.keep_raw, args.min_free_gb, args.flatten_days, args.max_signals, args.quick, args.cache_dir, args.star_timeout, args.max_sectors))

    total_tasks = len(tasks)
    completed_stars = 0
    success_count = 0
    skip_count = 0
    durations = []
    stage_timings = []
    match_stats = {'MATCHED': 0, 'HARMONIC': 0, 'UNMATCHED': 0}
    npz_path = os.path.join(os.path.dirname(args.output) or base_outputs, 'folded_curves.npz')

    logger.info(f"Targets: {len(target_ids)} total | Already processed: {len(processed_ids)} | Failed (permanent): {sum(1 for v in failed_attempts.values() if v >= 2)} | To process: {total_tasks}")
    print(f"Targets: {len(target_ids)} total | Already processed: {len(processed_ids)} | Failed (permanent): {sum(1 for v in failed_attempts.values() if v >= 2)} | To process: {total_tasks}", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        task_iter = iter(tasks)
        active_futures = {}
        for _ in range(min(args.workers * 2, total_tasks)):
            try:
                t = next(task_iter)
                f = executor.submit(process_single_star, *t)
                active_futures[f] = (t[0], time.time())
            except StopIteration:
                break

        while active_futures:
            done, _ = concurrent.futures.wait(active_futures.keys(), timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED)
            now = time.time()
            for f in done:
                tid_str, start_t = active_futures.pop(f)
                elapsed = now - start_t
                durations.append(elapsed)
                completed_stars += 1
                try:
                    feats, curves, r_tid, err, dl_size, timings = f.result()
                except Exception as exc:
                    err = f"{type(exc).__name__}: {str(exc)}"
                    feats, curves, dl_size, timings = None, None, 0, None

                if err:
                    logger.error(f"Error on {tid_str}: {err}")
                    log_failure(failed_csv, tid_str, err, failed_attempts)
                    skip_count += 1
                    diag_row = {
                        'target_id': tid_str,
                        'n_quarters': 0,
                        'baseline_d': 0.0,
                        'signals_found': 0,
                        'matched_count': 0,
                        'harmonic_count': 0,
                        'unmatched_count': 0,
                        'status': f'FAILED: {err}',
                        'max_sde': 0.0,
                        'max_snr': 0.0,
                        'time_dl_s': 0.0,
                        'time_stitch_s': 0.0,
                        'time_bls_s': 0.0,
                        'time_feats_s': 0.0,
                        'time_total_s': round(elapsed, 2)
                    }
                    append_to_csv_flush(diag_csv, pd.DataFrame([diag_row]))
                else:
                    success_count += 1
                    n_q = timings.get('n_quarters', 0) if timings else 0
                    b_d = timings.get('baseline_d', 0.0) if timings else 0.0
                    n_sig = len(feats) if feats else 0
                    m_cnt = sum(1 for item in feats if item.get('match_status') == 'MATCHED') if feats else 0
                    h_cnt = sum(1 for item in feats if item.get('match_status') == 'HARMONIC') if feats else 0
                    u_cnt = sum(1 for item in feats if item.get('match_status') == 'UNMATCHED') if feats else 0
                    b_details = timings.get('bls_details', []) if timings else []
                    m_sde = max([d.get('sde', 0.0) for d in b_details]) if b_details else 0.0
                    m_snr = max([d.get('snr', 0.0) for d in b_details]) if b_details else 0.0
                    diag_row = {
                        'target_id': tid_str,
                        'n_quarters': n_q,
                        'baseline_d': round(b_d, 2),
                        'signals_found': n_sig,
                        'matched_count': m_cnt,
                        'harmonic_count': h_cnt,
                        'unmatched_count': u_cnt,
                        'status': 'OK',
                        'max_sde': round(m_sde, 2),
                        'max_snr': round(m_snr, 2),
                        'time_dl_s': round(timings['dl'], 2) if timings else 0.0,
                        'time_stitch_s': round(timings['stitch'], 2) if timings else 0.0,
                        'time_bls_s': round(timings['bls'], 2) if timings else 0.0,
                        'time_feats_s': round(timings['feats'], 2) if timings else 0.0,
                        'time_total_s': round(timings['total'], 2) if timings else round(elapsed, 2)
                    }
                    append_to_csv_flush(diag_csv, pd.DataFrame([diag_row]))
                    if timings:
                        stage_timings.append(timings)
                        logger.info(f"Star {tid_str} breakdown: total={timings['total']:.1f}s | dl={timings['dl']:.1f}s | stitch={timings['stitch']:.1f}s | bls={timings['bls']:.1f}s | feats={timings['feats']:.1f}s")
                        print(f"Star {tid_str} breakdown: total={timings['total']:.1f}s | dl={timings['dl']:.1f}s | stitch={timings['stitch']:.1f}s | bls={timings['bls']:.1f}s | feats={timings['feats']:.1f}s", flush=True)
                        if 'bls_details' in timings and timings['bls_details']:
                            for b_d in timings['bls_details']:
                                detail_msg = f"  BLS iter {b_d['iter']}: {b_d['time']:.2f}s (grid: {b_d['n_freqs']} freqs) -> P={b_d.get('period', 0):.4f}d, SDE={b_d.get('sde', 0):.1f}, SNR={b_d.get('snr', 0):.1f}"
                                logger.info(detail_msg)
                                print(detail_msg, flush=True)
                    if feats:
                        append_to_csv_flush(args.output, pd.DataFrame(feats))
                        for item in feats:
                            m_st = item.get('match_status', 'UNMATCHED')
                            match_stats[m_st] = match_stats.get(m_st, 0) + 1
                    if curves:
                        if os.path.exists(npz_path):
                            with np.load(npz_path) as data:
                                existing = dict(data)
                            existing.update(curves)
                            np.savez_compressed(npz_path, **existing)
                        else:
                            np.savez_compressed(npz_path, **curves)

                avg_time = sum(durations) / len(durations)
                remaining_stars = total_tasks - completed_stars
                est_rem_sec = int((remaining_stars * avg_time) / max(1, args.workers))
                eta_str = str(datetime.timedelta(seconds=est_rem_sec))
                progress_msg = f"[{completed_stars}/{total_tasks}] Star {tid_str} ({elapsed:.1f}s) | Avg: {avg_time:.1f}s/star | ETA: {eta_str}"
                print(progress_msg, flush=True)
                logger.info(progress_msg)
                sys.stdout.flush()

                try:
                    next_t = next(task_iter)
                    next_f = executor.submit(process_single_star, *next_t)
                    active_futures[next_f] = (next_t[0], time.time())
                except StopIteration:
                    pass

            for f, (tid_str, start_t) in list(active_futures.items()):
                if now - start_t > args.star_timeout:
                    active_futures.pop(f)
                    f.cancel()
                    err = f"TimeoutError: Exceeded {args.star_timeout}s"
                    logger.error(f"Error on {tid_str}: {err}")
                    log_failure(failed_csv, tid_str, err, failed_attempts)
                    skip_count += 1
                    diag_row = {
                        'target_id': tid_str,
                        'n_quarters': 0,
                        'baseline_d': 0.0,
                        'signals_found': 0,
                        'matched_count': 0,
                        'harmonic_count': 0,
                        'unmatched_count': 0,
                        'status': f'FAILED: {err}',
                        'max_sde': 0.0,
                        'max_snr': 0.0,
                        'time_dl_s': 0.0,
                        'time_stitch_s': 0.0,
                        'time_bls_s': 0.0,
                        'time_feats_s': 0.0,
                        'time_total_s': round(args.star_timeout, 2)
                    }
                    append_to_csv_flush(diag_csv, pd.DataFrame([diag_row]))
                    completed_stars += 1
                    durations.append(args.star_timeout)
                    avg_time = sum(durations) / len(durations)
                    remaining_stars = total_tasks - completed_stars
                    est_rem_sec = int((remaining_stars * avg_time) / max(1, args.workers))
                    eta_str = str(datetime.timedelta(seconds=est_rem_sec))
                    progress_msg = f"[{completed_stars}/{total_tasks}] Star {tid_str} TIMEOUT ({args.star_timeout:.1f}s) | Avg: {avg_time:.1f}s/star | ETA: {eta_str}"
                    print(progress_msg, flush=True)
                    logger.info(progress_msg)
                    sys.stdout.flush()
                    try:
                        next_t = next(task_iter)
                        next_f = executor.submit(process_single_star, *next_t)
                        active_futures[next_f] = (next_t[0], time.time())
                    except StopIteration:
                        pass

    logger.info(f'Total processed: {success_count}')
    logger.info(f'Total skipped: {skip_count}')
    logger.info(f'Match stats: {match_stats}')
    if stage_timings:
        avg_dl = float(np.mean([t['dl'] for t in stage_timings]))
        avg_stitch = float(np.mean([t['stitch'] for t in stage_timings]))
        avg_bls = float(np.mean([t['bls'] for t in stage_timings]))
        avg_feats = float(np.mean([t['feats'] for t in stage_timings]))
        avg_tot = float(np.mean([t['total'] for t in stage_timings]))
        avg_report = f"Stage averages per star:\n  Download: {avg_dl:.1f}s\n  Stitch & detrend: {avg_stitch:.1f}s\n  BLS periodogram: {avg_bls:.1f}s\n  Feature computation: {avg_feats:.1f}s\n  Total per star: {avg_tot:.1f}s"
        print(avg_report, flush=True)
        logger.info(avg_report)

    if args.purge_lk_cache:
        purge_cache()

if __name__ == '__main__':
    main()
