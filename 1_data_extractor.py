import argparse
import datetime
import logging
import os
import shutil
import sys
import tempfile
import time
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
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

def run_injection(tid, mission, injections_per_star, min_free_gb, flatten_days, quick_mode=False, cache_dir=None, star_timeout=1800.0):
    free_bytes = shutil.disk_usage(os.path.abspath('.')).free
    if free_bytes < min_free_gb * 1024**3: return None, tid, 'Disk space below threshold'
    start_t = time.time()
    deadline = start_t + star_timeout
    base_cache = cache_dir if cache_dir else tempfile.gettempdir()
    tmp_dir = os.path.join(base_cache, f'onyx_inj_{tid}_{os.getpid()}_{int(time.time()*1000)}')
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        author = 'Kepler' if mission == 'Kepler' else 'SPOC'
        for attempt in range(3):
            if time.time() > deadline:
                return None, tid, f'TimeoutError: Exceeded {star_timeout}s'
            try:
                search_result = lk.search_lightcurve(tid, author=author, cadence='long')
                break
            except Exception:
                time.sleep(2 ** attempt)
        else: return None, tid, 'Download failed'
        if len(search_result) == 0: return None, tid, 'No lightcurves found'
        if time.time() > deadline:
            return None, tid, f'TimeoutError: Exceeded {star_timeout}s'
        lc_col = search_result.download_all(download_dir=tmp_dir)
        if lc_col is None or len(lc_col) == 0: return None, tid, 'No lightcurves found'
        lc = lc_col.stitch(corrector_func=lambda x: x.normalize())
        lc = lc.remove_nans().remove_outliers(sigma=5)
        if len(lc) < 500: return None, tid, 'Not enough points'
        times = np.asarray(lc.time.value, dtype=float)
        flux_val = np.asarray(lc.flux.value, dtype=float)
        baseline_d = float(np.nanmax(times) - np.nanmin(times))
        star_radius = float(lc.meta.get('RADIUS', 1.0))
        if np.isnan(star_radius): star_radius = 1.0
        star_logg = float(lc.meta.get('LOGG', 4.438))
        if np.isnan(star_logg): star_logg = 4.438
        rho_star_sun = (star_radius**-3) * (10**(star_logg - 4.438)) * (star_radius**2) 
        freq_factor = 5000 if quick_mode else 500
        dur_grid = np.linspace(0.05, 0.40, 4) if quick_mode else np.linspace(0.04, 0.45, 10)
        results = []
        features = []
        for i in range(injections_per_star):
            if time.time() > deadline:
                break
            inj_period = np.exp(np.random.uniform(np.log(0.8), np.log(min(100.0, baseline_d/2.0))))
            inj_rad_earth = np.exp(np.random.uniform(np.log(0.5), np.log(15.0)))
            rp_rs = inj_rad_earth * 0.009168 / star_radius
            b = np.random.uniform(0, 0.8)
            duration = (13.0 / 24.0) * ((inj_period/365.0)**(1/3)) * (rho_star_sun**(-1/3)) * np.sqrt(1 - b**2)
            epoch = np.nanmin(times) + np.random.uniform(0, inj_period)
            inj_flux = inject_transit(times, flux_val, inj_period, epoch, rp_rs, duration)
            lc_inj = lc.copy()
            lc_inj.flux = inj_flux * lc_inj.flux.unit
            med_diff = np.nanmedian(np.diff(times))
            wl = int(flatten_days / med_diff) if med_diff > 0 else 101
            if wl % 2 == 0: wl -= 1
            if wl < 3: wl = 3
            if wl > len(lc_inj): wl = len(lc_inj) if len(lc_inj) % 2 != 0 else len(lc_inj) - 1
            lc_flat = lc_inj.flatten(window_length=wl, polyorder=2, break_tolerance=5)
            bls = lc_flat.to_periodogram(method='bls', minimum_period=0.5, maximum_period=min(100.0, baseline_d/2.0), frequency_factor=freq_factor, duration=dur_grid)
            rec_period = float(bls.period_at_max_power.value)
            rec_epoch = float(bls.transit_time_at_max_power.value)
            sde = float((np.nanmax(bls.power.value) - np.nanmean(bls.power.value)) / np.nanstd(bls.power.value))
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
            res = {
                'target_id': tid, 'period_d': inj_period, 'radius_earth': inj_rad_earth,
                'depth_ppm': (rp_rs**2)*1e6, 'duration_d': duration, 'impact_b': b,
                'recovered': recovered, 'alias_type': alias_type,
                'bls_sde': sde, 'bls_depth_snr': snr, 'star_radius': star_radius
            }
            results.append(res)
            if recovered:
                rem_timeout = max(10.0, deadline - time.time())
                feat, _, _, _, _ = process_single_star(tid, mission, [{'catalog_period': inj_period, 'label': 'INJECTED'}], True, 0, flatten_days, 1, quick_mode=quick_mode, cache_dir=cache_dir, star_timeout=rem_timeout)
                if feat and len(feat) > 0:
                    features.append(feat[0])
        return results, features, tid, None
    except Exception as e:
        return None, None, tid, str(type(e).__name__) + ' - ' + str(e)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

def process_single_star(tid, mission, row_dicts, keep_raw, min_free_gb, flatten_days, max_signals, quick_mode=False, cache_dir=None, star_timeout=1800.0):
    free_bytes = shutil.disk_usage(os.path.abspath('.')).free
    if free_bytes < min_free_gb * 1024**3:
        return None, None, tid, 'Disk space below threshold', 0
    start_t = time.time()
    deadline = start_t + star_timeout
    base_cache = cache_dir if cache_dir else tempfile.gettempdir()
    tmp_dir = os.path.join(base_cache, f'onyx_raw_{tid}_{os.getpid()}_{int(time.time()*1000)}')
    os.makedirs(tmp_dir, exist_ok=True)
    dl_size = 0
    try:
        author = 'Kepler' if mission == 'Kepler' else 'SPOC'
        for attempt in range(3):
            if time.time() > deadline:
                return None, None, tid, f'TimeoutError: Exceeded {star_timeout}s', dl_size
            try:
                search_result = lk.search_lightcurve(tid, author=author, cadence='long')
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            return None, None, tid, 'Download failed', dl_size
        if len(search_result) == 0:
            return None, None, tid, 'No lightcurves found', dl_size
        if time.time() > deadline:
            return None, None, tid, f'TimeoutError: Exceeded {star_timeout}s', dl_size
        lc_col = search_result.download_all(download_dir=tmp_dir)
        dl_size = get_dir_size(tmp_dir)
        if lc_col is None or len(lc_col) == 0:
            return None, None, tid, 'No lightcurves found', dl_size
        lc = lc_col.stitch(corrector_func=lambda x: x.normalize())
        lc = lc.remove_nans().remove_outliers(sigma=5)
        if len(lc) < 500:
            return None, None, tid, 'Not enough valid points', dl_size
        times = np.asarray(lc.time.value, dtype=float)
        flux_val = np.asarray(lc.flux.value, dtype=float)
        baseline_d = float(np.nanmax(times) - np.nanmin(times))
        if baseline_d <= 0:
            return None, None, tid, 'Invalid baseline', dl_size
        med_diff = np.nanmedian(np.diff(times))
        if med_diff > 0:
            wl = int(flatten_days / med_diff)
            if wl % 2 == 0: wl -= 1
            if wl < 3: wl = 3
        else:
            wl = 101
        if wl > len(lc): wl = len(lc) if len(lc) % 2 != 0 else len(lc) - 1
        if wl < 3: return None, None, tid, 'Time series too short for flatten', dl_size
        if time.time() > deadline:
            return None, None, tid, f'TimeoutError: Exceeded {star_timeout}s', dl_size
        lc_flat = lc.flatten(window_length=wl, polyorder=2, break_tolerance=5)
        times_flat = np.asarray(lc_flat.time.value, dtype=float)
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
        for sig_idx in range(max_signals):
            if time.time() > deadline:
                break
            if max_p <= 0.5 or len(lc_search) < 100:
                break
            bls = lc_search.to_periodogram(method='bls', minimum_period=0.5, maximum_period=max_p, frequency_factor=freq_factor, duration=dur_grid)
            if len(bls.power) == 0 or np.isnan(bls.power).all():
                break
            power_values = np.array(bls.power.value)
            bls_power_peak = float(np.nanmax(power_values))
            bls_sde = float((bls_power_peak - np.nanmean(power_values)) / np.nanstd(power_values))
            bls_depth_snr = float(np.nanmax(bls.snr))
            if bls_sde < 7.0 or bls_depth_snr < 7.1:
                break
            bls_period_d = float(bls.period_at_max_power.value)
            bls_epoch = float(bls.transit_time_at_max_power.value)
            bls_duration_d = float(bls.duration_at_max_power.value)
            bls_depth_fraction = float(bls.depth_at_max_power.value)
            bls_depth_ppm = bls_depth_fraction * 1e6
            transit_count = max(1, int(baseline_d / bls_period_d))
            duty_cycle = bls_duration_d / bls_period_d
            folded = lc_search.fold(period=bls_period_d, epoch_time=bls_epoch, normalize_phase=False)
            folded_time_val = np.asarray(folded.time.value, dtype=float)
            folded_flux_val = np.asarray(folded.flux.value, dtype=float)
            phase_time = folded_time_val / bls_period_d
            dur_time = folded_time_val / bls_duration_d
            phase_curve = bin_folded_curve(phase_time, folded_flux_val, 201, -0.5, 0.5)
            dur_curve = bin_folded_curve(dur_time, folded_flux_val, 201, -3.0, 3.0)
            sig_id = f"{tid}_{sig_idx}"
            curves_dict[sig_id] = np.vstack([phase_curve, dur_curve])
            match_status = 'UNMATCHED'
            match_label = 'UNMATCHED'
            match_flags = {'koi_fpflag_nt': 0, 'koi_fpflag_ss': 0, 'koi_fpflag_co': 0, 'koi_fpflag_ec': 0}
            for row in row_dicts:
                cat_p = row.get('catalog_period', np.nan)
                if pd.isna(cat_p): continue
                ratio = bls_period_d / cat_p
                if abs(ratio - 1.0) < 0.01:
                    match_status = 'MATCHED'
                    match_label = row.get('label', 'UNMATCHED')
                    if mission == 'Kepler':
                        match_flags['koi_fpflag_nt'] = row.get('koi_fpflag_nt', 0)
                        match_flags['koi_fpflag_ss'] = row.get('koi_fpflag_ss', 0)
                        match_flags['koi_fpflag_co'] = row.get('koi_fpflag_co', 0)
                        match_flags['koi_fpflag_ec'] = row.get('koi_fpflag_ec', 0)
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
                'star_logg': star_logg
            }
            if mission == 'Kepler':
                feat.update(match_flags)
            features_list.append(feat)
            transit_mask = lc_search.create_transit_mask(period=bls_period_d, transit_time=bls_epoch, duration=bls_duration_d * 1.5)
            lc_search = lc_search[~transit_mask]
        for f in features_list:
            f['n_signals_star'] = len(features_list)
        return features_list, curves_dict, tid, None, dl_size
    except Exception as e:
        return None, None, tid, str(type(e).__name__) + ' - ' + str(e), dl_size
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
    append_to_csv_flush(failed_path, df_fail) if 'df_fail' in locals() else append_to_csv_flush(failed_path, df_err)

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
    args = parser.parse_args()
    logger = setup_logger()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)
    os.makedirs(os.path.join('outputs', 'models'), exist_ok=True)
    os.makedirs(os.path.join('outputs', 'explanations'), exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.quick:
        logger.warning('WARNING: Quick mode enabled. Using coarse BLS grid (frequency_factor=5000, 4 duration steps). Outputs will have _quick suffix.')
        print('WARNING: Quick mode enabled. Using coarse BLS grid (frequency_factor=5000, 4 duration steps). Outputs will have _quick suffix.', flush=True)
        if args.output == 'extracted_features.csv':
            args.output = 'extracted_features_quick.csv'

    base_out, ext_out = os.path.splitext(args.output)
    failed_csv = f"{base_out}_failed{ext_out if ext_out else '.csv'}"
    resume = not args.no_resume

    if not resume:
        if os.path.exists(args.output):
            os.remove(args.output)
        if os.path.exists(failed_csv):
            os.remove(failed_csv)
        npz_del = os.path.join(os.path.dirname(args.output) or 'outputs', 'folded_curves.npz')
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
        quiet_stars = []
        if args.targets_file and os.path.exists(args.targets_file):
            tids_from_file = read_targets_file(args.targets_file)
            quiet_stars = [tid for tid in tids_from_file if tid not in all_tids]
        else:
            logger.error("Please provide --targets-file with a list of target IDs for injection (which will be filtered to exclude KOIs).")
            return
        np.random.shuffle(quiet_stars)
        quiet_stars = quiet_stars[:args.injection_stars]

        inj_out = args.injection_output
        if args.quick and inj_out == 'outputs/injection_results.csv':
            inj_out = 'outputs/injection_results_quick.csv'
        elif args.quick and not inj_out.endswith('_quick.csv'):
            inj_out = inj_out.replace('.csv', '_quick.csv')

        inj_feat_out = 'outputs/injection_features_quick.csv' if args.quick else 'outputs/injection_features.csv'

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
        for tid in quiet_stars:
            if str(tid) not in completed_inj_stars:
                tasks.append((tid, args.mission, args.injections_per_star, args.min_free_gb, args.flatten_days, args.quick, args.cache_dir, args.star_timeout))

        total_inj = len(tasks)
        done_inj = 0
        durations_inj = []
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
                        res, feats, r_tid, err = f.result()
                    except Exception as exc:
                        err = f"{type(exc).__name__}: {str(exc)}"
                        res, feats = None, None

                    if err:
                        logger.error(f"Injection error on {tid_str}: {err}")
                    else:
                        if res:
                            append_to_csv_flush(inj_out, pd.DataFrame(res))
                        if feats:
                            append_to_csv_flush(inj_feat_out, pd.DataFrame(feats))

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
            heatmap_out = 'outputs/figures/injection_heatmap_quick.png' if args.quick else 'outputs/figures/injection_heatmap.png'
            plt.savefig(heatmap_out)
            plt.close()
        return

    target_ids = []
    if args.targets_file and os.path.exists(args.targets_file):
        target_ids = read_targets_file(args.targets_file)
    else:
        labels_df = labels_df.dropna(subset=['label'])
        target_ids = labels_df['target_id'].unique().tolist()
    np.random.seed(42)
    np.random.shuffle(target_ids)
    target_ids = target_ids[:args.limit]

    processed_ids = set()
    if resume and os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        try:
            df_exist = pd.read_csv(args.output, usecols=['target_id'], dtype=str)
            processed_ids = set(df_exist['target_id'].dropna().astype(str))
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
        tasks.append((tid_str, args.mission, rows, args.keep_raw, args.min_free_gb, args.flatten_days, args.max_signals, args.quick, args.cache_dir, args.star_timeout))

    total_tasks = len(tasks)
    completed_stars = 0
    success_count = 0
    skip_count = 0
    durations = []
    match_stats = {'MATCHED': 0, 'HARMONIC': 0, 'UNMATCHED': 0}
    npz_path = os.path.join(os.path.dirname(args.output) or 'outputs', 'folded_curves.npz')

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
                    feats, curves, r_tid, err, dl_size = f.result()
                except Exception as exc:
                    err = f"{type(exc).__name__}: {str(exc)}"
                    feats, curves, dl_size = None, None, 0

                if err:
                    logger.error(f"Error on {tid_str}: {err}")
                    log_failure(failed_csv, tid_str, err, failed_attempts)
                    skip_count += 1
                else:
                    success_count += 1
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
    if args.purge_lk_cache:
        purge_cache()

if __name__ == '__main__':
    main()
