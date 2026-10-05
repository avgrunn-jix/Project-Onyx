import argparse
import logging
import os
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

def fetch_labels(mission):
    os.makedirs('data/labels', exist_ok=True)
    if mission == 'Kepler':
        labels_table = NasaExoplanetArchive.query_criteria(table='cumulative', select='kepid,koi_disposition')
        df = labels_table.to_pandas()
        df.rename(columns={'kepid': 'target_id', 'koi_disposition': 'label'}, inplace=True)
        df['target_id'] = df['target_id'].astype(str)
        df.to_csv('data/labels/kepler_labels.csv', index=False)
        return df
    else:
        labels_table = NasaExoplanetArchive.query_criteria(table='toi', select='tic_id,tfopwg_disp')
        df = labels_table.to_pandas()
        df.rename(columns={'tic_id': 'target_id', 'tfopwg_disp': 'label'}, inplace=True)
        df['target_id'] = df['target_id'].astype(str)
        df.to_csv('data/labels/tess_labels.csv', index=False)
        return df

def process_single_star(tid, mission, label):
    try:
        author = 'Kepler' if mission == 'Kepler' else 'SPOC'
        for attempt in range(3):
            try:
                search_result = lk.search_lightcurve(tid, author=author, cadence='long')
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            return None, tid, 'Download failed'
            
        if len(search_result) == 0:
            return None, tid, 'No lightcurves found'
        
        lc_col = search_result.download_all()
        if lc_col is None or len(lc_col) == 0:
            return None, tid, 'No lightcurves found'
            
        lc = lc_col.stitch(corrector_func=lambda x: x.normalize())
        lc = lc.remove_nans().remove_outliers(sigma=5)
        if len(lc) < 500:
            return None, tid, 'Not enough valid points'
        wl = 101 if len(lc) > 101 else len(lc)
        if wl % 2 == 0:
            wl -= 1
        if wl < 3:
            return None, tid, 'Time series too short for flatten'
        lc = lc.flatten(window_length=wl, polyorder=2, break_tolerance=5)
        baseline_d = float((lc.time.max() - lc.time.min()).value)
        if baseline_d <= 0:
            return None, tid, 'Invalid baseline'
            
        duration_grid = np.linspace(0.04, 0.45, 10)
        bls = lc.to_periodogram(method='bls', minimum_period=0.5, maximum_period=min(100.0, baseline_d / 2.0), frequency_factor=500, duration=duration_grid)
        if len(bls.power) == 0 or np.isnan(bls.power).all():
            return None, tid, 'BLS failed'
            
        bls_period_d = float(bls.period_at_max_power.value)
        bls_epoch = float(bls.transit_time_at_max_power.value)
        bls_duration_d = float(bls.duration_at_max_power.value)
        bls_depth_fraction = float(bls.depth_at_max_power.value)
        bls_depth_ppm = bls_depth_fraction * 1e6
        power_values = np.array(bls.power.value)
        bls_power_peak = float(np.nanmax(power_values))
        bls_sde = float((bls_power_peak - np.nanmean(power_values)) / np.nanstd(power_values))
        transit_count = max(1, int(baseline_d / bls_period_d))
        duty_cycle = bls_duration_d / bls_period_d
        folded = lc.fold(period=bls_period_d, epoch_time=bls_epoch)
        folded_double = lc.fold(period=bls_period_d * 2, epoch_time=bls_epoch)
        odd_mask = np.abs(folded_double.time.value) < (bls_duration_d / 2)
        even_mask = np.abs(folded_double.time.value - bls_period_d) < (bls_duration_d / 2)
        flux_odd = folded_double.flux.value[odd_mask]
        flux_even = folded_double.flux.value[even_mask]
        med_odd = np.nanmedian(flux_odd) if len(flux_odd) > 0 else 1.0
        med_even = np.nanmedian(flux_even) if len(flux_even) > 0 else 1.0
        odd_even_depth_diff_ppm = abs(med_odd - med_even) * 1e6
        combined_std = np.nanstd(folded_double.flux.value)
        odd_even_sigma = float(abs(med_odd - med_even) / combined_std) if combined_std > 0 else 0.0
        sec_mask = np.abs(folded.time.value - 0.5) < (duty_cycle / 2)
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
        times = np.asarray(lc.time.value, dtype=float)
        flux_val = np.asarray(lc.flux.value, dtype=float)
        transit_depths = []
        for i in range(transit_count):
            mid = bls_epoch + i * bls_period_d
            mask = np.abs(times - mid) < (bls_duration_d / 2)
            if np.any(mask):
                transit_depths.append(1.0 - np.nanmean(flux_val[mask]))
        depth_scatter_ppm = float(np.nanstd(transit_depths) * 1e6) if len(transit_depths) > 0 else 0.0
        flux_std_ppm = float(np.nanstd(flux_val) * 1e6)
        flux_mad_ppm = float(np.nanmedian(np.abs(flux_val - np.nanmedian(flux_val))) * 1e6)
        flux_skew = float(st.skew(flux_val, nan_policy='omit'))
        flux_kurtosis = float(st.kurtosis(flux_val, nan_policy='omit'))
        cdpp_ppm = float(lc.estimate_cdpp().value)
        pointwise_scatter_ppm = float(np.nanstd(np.diff(flux_val)) * 1e6 / np.sqrt(2))
        acorr = np.correlate(flux_val - np.nanmean(flux_val), flux_val - np.nanmean(flux_val), mode='full')
        acorr = acorr[len(acorr)//2:]
        acorr_norm = acorr / acorr[0] if acorr[0] > 0 else acorr
        autocorr_max = float(np.nanmax(acorr_norm[1:min(100, len(acorr_norm))])) if len(acorr_norm) > 1 else 0.0
        ls = lc.to_periodogram(method='lombscargle', minimum_period=0.5, maximum_period=min(100.0, baseline_d / 2.0), oversample_factor=10)
        ls_peak_power = float(np.nanmax(ls.power.value))
        time_diffs = np.diff(times)
        med_diff = np.nanmedian(time_diffs)
        gap_fraction = float(1.0 - (len(lc) / (baseline_d / med_diff))) if med_diff > 0 else 0.0
        star_teff = float(lc.meta.get('TEFF', np.nan))
        star_radius = float(lc.meta.get('RADIUS', np.nan))
        star_logg = float(lc.meta.get('LOGG', np.nan))
        
        feat = {
            'target_id': tid, 'mission': mission, 'label': label,
            'bls_period_d': bls_period_d, 'bls_epoch': bls_epoch, 'bls_duration_d': bls_duration_d,
            'bls_depth_ppm': bls_depth_ppm, 'bls_depth_snr': float(np.nanmax(bls.snr)),
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
        return feat, tid, None
    except Exception as e:
        return None, tid, str(type(e).__name__) + ' - ' + str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission', default='Kepler', choices=['Kepler', 'TESS'])
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--output', default='extracted_features.csv')
    parser.add_argument('--targets-file', default='')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    logger = setup_logger()
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
            row = labels_df[labels_df['target_id'] == str(tid)]
            lbl = row.iloc[0]['label'] if len(row) > 0 else ''
            tasks.append((str(tid), args.mission, lbl))
            
    success_count = 0
    skip_count = 0
    features_list = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_star, t[0], t[1], t[2]): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            feat, tid, err = future.result()
            if err:
                logger.error('Error on ' + str(tid) + ': ' + err)
                skip_count += 1
            else:
                features_list.append(feat)
                success_count += 1
                if len(features_list) >= 10:
                    df_out = pd.DataFrame(features_list)
                    hdr = not os.path.exists(args.output)
                    df_out.to_csv(args.output, mode='a', header=hdr, index=False)
                    features_list = []
                    
    if len(features_list) > 0:
        df_out = pd.DataFrame(features_list)
        hdr = not os.path.exists(args.output)
        df_out.to_csv(args.output, mode='a', header=hdr, index=False)
        
    logger.info('Total processed: ' + str(success_count))
    logger.info('Total skipped: ' + str(skip_count))

if __name__ == '__main__':
    main()
