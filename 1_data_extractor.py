import argparse
import logging
import os
import shutil
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
    file_handler = logging.FileHandler('logs/extractor_errors.log')
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
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
    os.makedirs('data/labels', exist_ok=True)
    if mission == 'Kepler':
        labels_table = NasaExoplanetArchive.query_criteria(table='cumulative', select='kepid,koi_disposition,koi_period,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec')
        df = labels_table.to_pandas()
        df.rename(columns={'kepid': 'target_id', 'koi_disposition': 'label', 'koi_period': 'catalog_period'}, inplace=True)
        df['target_id'] = df['target_id'].astype(str)
        df.to_csv('data/labels/kepler_labels.csv', index=False)
        return df
    else:
        labels_table = NasaExoplanetArchive.query_criteria(table='toi', select='tic_id,tfopwg_disp,pl_orbper')
        df = labels_table.to_pandas()
        df.rename(columns={'tic_id': 'target_id', 'tfopwg_disp': 'label', 'pl_orbper': 'catalog_period'}, inplace=True)
        df['target_id'] = df['target_id'].astype(str)
        df.to_csv('data/labels/tess_labels.csv', index=False)
        return df

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

def run_injection(tid, mission, injections_per_star, min_free_gb, flatten_days):
    free_bytes = shutil.disk_usage(os.path.abspath('.')).free
    if free_bytes < min_free_gb * 1024**3: return None, tid, 'Disk space'
    tmp_dir = f'data/raw/tmp_inj_{tid}_{os.getpid()}'
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        author = 'Kepler' if mission == 'Kepler' else 'SPOC'
        for attempt in range(3):
            try:
                search_result = lk.search_lightcurve(tid, author=author, cadence='long')
                break
            except Exception:
                time.sleep(2 ** attempt)
        else: return None, tid, 'Download failed'
        if len(search_result) == 0: return None, tid, 'No lightcurves found'
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
        results = []
        features = []
        for i in range(injections_per_star):
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
            bls = lc_flat.to_periodogram(method='bls', minimum_period=0.5, maximum_period=min(100.0, baseline_d/2.0), frequency_factor=500)
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
                feat, _, _, _, _ = process_single_star(tid, mission, [{'catalog_period': inj_period, 'label': 'INJECTED'}], True, 0, flatten_days, 1)
                if feat and len(feat) > 0:
                    features.append(feat[0])
        return results, features, tid, None
    except Exception as e:
        return None, None, tid, str(type(e).__name__) + ' - ' + str(e)
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

def process_single_star(tid, mission, row_dicts, keep_raw, min_free_gb, flatten_days, max_signals):
    free_bytes = shutil.disk_usage(os.path.abspath('.')).free
    if free_bytes < min_free_gb * 1024**3:
        return None, None, tid, 'Disk space below threshold', 0

    tmp_dir = f'data/raw/tmp_{tid}_{os.getpid()}'
    os.makedirs(tmp_dir, exist_ok=True)
    dl_size = 0
    try:
        author = 'Kepler' if mission == 'Kepler' else 'SPOC'
        for attempt in range(3):
            try:
                search_result = lk.search_lightcurve(tid, author=author, cadence='long')
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            return None, None, tid, 'Download failed', dl_size
        if len(search_result) == 0:
            return None, None, tid, 'No lightcurves found', dl_size
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
        for sig_idx in range(max_signals):
            if max_p <= 0.5 or len(lc_search) < 100:
                break
            duration_grid = np.linspace(0.04, 0.45, 10)
            bls = lc_search.to_periodogram(method='bls', minimum_period=0.5, maximum_period=max_p, frequency_factor=500, duration=duration_grid)
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
            sec_mask = np.abs(folded.time.value) < (bls_period_d / 2)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission', default='Kepler', choices=['Kepler', 'TESS'])
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--output', default='extracted_features.csv')
    parser.add_argument('--targets-file', default='')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--keep-raw', action='store_true')
    parser.add_argument('--min-free-gb', type=float, default=5.0)
    parser.add_argument('--purge-lk-cache', action='store_true')
    parser.add_argument('--flatten-days', type=float, default=3.0)
    parser.add_argument('--max-signals', type=int, default=4)
    parser.add_argument('--inject', action='store_true')
    parser.add_argument('--injection-stars', type=int, default=100)
    parser.add_argument('--injections-per-star', type=int, default=12)
    parser.add_argument('--injection-output', default='outputs/injection_results.csv')
    args = parser.parse_args()
    logger = setup_logger()
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    def purge_cache():
        cache_dir = os.path.expanduser('~/.lightkurve/cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
    if args.purge_lk_cache:
        purge_cache()
    for item in os.listdir(args.raw_dir):
        if item.startswith('tmp_'):
            shutil.rmtree(os.path.join(args.raw_dir, item), ignore_errors=True)

    free_start = shutil.disk_usage(os.path.abspath('.')).free
    with open('logs/disk_usage.log', 'a') as f:
        f.write(f'Start: {free_start / 1024**3:.2f} GB free\n')

    if args.inject:
        os.makedirs('outputs', exist_ok=True)
        all_tids = labels_df['target_id'].unique().tolist()
        quiet_stars = []
        if args.targets_file and os.path.exists(args.targets_file):
            with open(args.targets_file, 'r') as f:
                quiet_stars = [line.strip() for line in f if line.strip() not in all_tids]
        else:
            logger.error("Please provide --targets-file with a list of target IDs for injection (which will be filtered to exclude KOIs).")
            return
        np.random.shuffle(quiet_stars)
        quiet_stars = quiet_stars[:args.injection_stars]
        tasks = [(tid, args.mission, args.injections_per_star, args.min_free_gb, args.flatten_days) for tid in quiet_stars]
        all_results = []
        all_features = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_injection, *t): t for t in tasks}
            for future in concurrent.futures.as_completed(futures):
                res, feats, tid, err = future.result()
                if err:
                    logger.error(f"Injection error on {tid}: {err}")
                else:
                    all_results.extend(res)
                    all_features.extend(feats)
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(args.injection_output, index=False)
        if all_features:
            df_feat = pd.DataFrame(all_features)
            df_feat.to_csv('outputs/injection_features.csv', index=False)
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
        plt.title('Completeness Heatmap')
        plt.savefig('outputs/figures/injection_heatmap.png')
        plt.close()
        return

    labels_df = fetch_labels(args.mission)
    target_ids = []
    if args.targets_file and os.path.exists(args.targets_file):
        with open(args.targets_file, 'r') as f:
            target_ids = [line.strip() for line in f if line.strip()]
    else:
        labels_df = labels_df.dropna(subset=['label'])
        target_ids = labels_df['target_id'].unique().tolist()
    np.random.seed(42)
    np.random.shuffle(target_ids)
    target_ids = target_ids[:args.limit]
    processed_ids = set()
    if os.path.exists(args.output):
        df_exist = pd.read_csv(args.output)
        processed_ids = set(df_exist['target_id'].astype(str))
    tasks = []
    for tid in target_ids:
        if str(tid) not in processed_ids:
            rows = labels_df[labels_df['target_id'] == str(tid)].to_dict('records')
            tasks.append((str(tid), args.mission, rows, args.keep_raw, args.min_free_gb, args.flatten_days, args.max_signals))
    success_count = 0
    skip_count = 0
    total_dl_size = 0
    processed_since_log = 0
    all_features = []
    all_curves = {}
    match_stats = {'MATCHED': 0, 'HARMONIC': 0, 'UNMATCHED': 0}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_star, *t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            feats, curves, tid, err, dl_size = future.result()
            total_dl_size += dl_size
            processed_since_log += 1
            if err:
                logger.error(f'Error on {tid}: {err}')
                if err == 'Disk space below threshold':
                    logger.error('Disk space critical. Stopping early.')
                    for f in futures: f.cancel()
                    break
                skip_count += 1
            else:
                success_count += 1
                if feats:
                    all_features.extend(feats)
                    all_curves.update(curves)
                    for f in feats:
                        match_stats[f['match_status']] = match_stats.get(f['match_status'], 0) + 1
            if processed_since_log >= 25:
                free_now = shutil.disk_usage(os.path.abspath('.')).free
                deleted_size = total_dl_size if not args.keep_raw else 0
                with open('logs/disk_usage.log', 'a') as f:
                    f.write(f'Processed 25: DL {total_dl_size / 1024**2:.1f}MB, Del {deleted_size / 1024**2:.1f}MB, Free {free_now / 1024**3:.2f}GB\n')
                total_dl_size = 0
                processed_since_log = 0
            if len(all_features) >= 10:
                df_out = pd.DataFrame(all_features)
                hdr = not os.path.exists(args.output)
                df_out.to_csv(args.output, mode='a', header=hdr, index=False)
                npz_path = 'outputs/folded_curves.npz'
                if os.path.exists(npz_path):
                    with np.load(npz_path) as data:
                        existing = dict(data)
                    existing.update(all_curves)
                    np.savez_compressed(npz_path, **existing)
                else:
                    np.savez_compressed(npz_path, **all_curves)
                all_features = []
                all_curves = {}
    if len(all_features) > 0:
        df_out = pd.DataFrame(all_features)
        hdr = not os.path.exists(args.output)
        df_out.to_csv(args.output, mode='a', header=hdr, index=False)
        npz_path = 'outputs/folded_curves.npz'
        if os.path.exists(npz_path):
            with np.load(npz_path) as data:
                existing = dict(data)
            existing.update(all_curves)
            np.savez_compressed(npz_path, **existing)
        else:
            np.savez_compressed(npz_path, **all_curves)
    logger.info(f'Total processed: {success_count}')
    logger.info(f'Total skipped: {skip_count}')
    logger.info(f'Match stats: {match_stats}')
    if args.purge_lk_cache:
        purge_cache()

if __name__ == '__main__':
    main()
