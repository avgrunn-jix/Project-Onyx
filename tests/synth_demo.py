import os

import numpy as np
import pandas as pd

SHOWCASE = '4852528'

FEATURE_COLUMNS = [
    'bls_period_d', 'bls_epoch', 'bls_duration_d', 'bls_depth_ppm', 'bls_depth_snr',
    'bls_power_peak', 'bls_sde', 'transit_count', 'duty_cycle',
    'odd_even_depth_diff_ppm', 'odd_even_sigma', 'secondary_depth_ppm',
    'secondary_snr', 'harmonic_power_ratio', 'shape_v_u_ratio',
    'in_transit_scatter_ppm', 'depth_scatter_ppm', 'flux_std_ppm',
    'flux_mad_ppm', 'flux_skew', 'flux_kurtosis', 'cdpp_ppm',
    'pointwise_scatter_ppm', 'autocorr_max', 'ls_peak_power', 'gap_fraction',
    'baseline_d', 'star_teff', 'star_radius', 'star_logg',
    'centroid_shift_col_sigma', 'centroid_shift_row_sigma',
    'centroid_shift_total_sigma', 'centroid_shift_total_arcsec',
]


def build_workspace(root, n_per_group=16, seed=3, with_version=True, version='2|quick=1|flatten_days=3|max_signals=4', suffix='_quick'):
    rng = np.random.RandomState(seed)
    root = str(root)
    labels_dir = os.path.join(root, 'data', 'labels')
    out_dir = os.path.join(root, 'outputs', 'demo_ws')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    eb_ids = [str(1000000 + i * 17) for i in range(12)]
    pd.DataFrame({'kepid': eb_ids}).to_csv(os.path.join(labels_dir, 'kepler_eb_catalog.csv'), index=False)

    dispositions = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
    groups = {}
    next_id = [2000000]

    def new_ids(n):
        out = []
        for _ in range(n):
            next_id[0] += 1237
            out.append(str(next_id[0]))
        return out

    groups['tr_m'] = new_ids(n_per_group)
    groups['ctrl_m'] = [SHOWCASE] + new_ids(n_per_group - 1)
    groups['tr_u'] = new_ids(n_per_group)
    groups['ctrl_u'] = new_ids(n_per_group)

    label_rows = []
    feat_rows = []
    koi_periods = {}
    marked_all = groups['tr_m'] + groups['ctrl_m']
    for i, sid in enumerate(marked_all):
        disp = 'CONFIRMED' if sid == SHOWCASE else dispositions[i % 3]
        n_koi = 2 if i % 3 == 0 else 1
        periods = [round(2.0 + 1.7 * i % 40 + 3.1 * k, 4) for k in range(n_koi)]
        koi_periods[sid] = periods
        for p in periods:
            label_rows.append({'target_id': sid, 'label': disp, 'catalog_period': p,
                               'koi_fpflag_nt': 0, 'koi_fpflag_ss': 0, 'koi_fpflag_co': 0, 'koi_fpflag_ec': 0})

    def base_features(cls, snr):
        row = {c: float(rng.lognormal(0.0, 0.5)) for c in FEATURE_COLUMNS}
        row['odd_even_sigma'] = float(abs(rng.normal(1.5 - 1.2 * cls, 0.4)))
        row['secondary_snr'] = float(abs(rng.normal(1.2 - 0.9 * cls, 0.4)))
        row['shape_v_u_ratio'] = float(abs(rng.normal(1.0 - 0.4 * cls, 0.2)))
        row['bls_depth_snr'] = snr
        row['bls_sde'] = snr * 1.1
        row['bls_depth_ppm'] = float(rng.lognormal(6.0, 0.4))
        row['transit_count'] = float(rng.randint(5, 200))
        row['gap_fraction'] = float(rng.uniform(0.0, 0.2))
        row['baseline_d'] = 1400.0
        return row

    star_class = {}
    for i, sid in enumerate(marked_all):
        disp = 'CONFIRMED' if sid == SHOWCASE else dispositions[i % 3]
        cls = 1 if disp in ('CONFIRMED', 'CANDIDATE') else 0
        star_class[sid] = (disp, cls)
        star_snr = float(rng.uniform(8, 40))
        sig_idx = 0
        for k, p in enumerate(koi_periods[sid]):
            miss = (i % 7 == 3 and k == 1)
            harmonic = (i % 5 == 4 and k == 0)
            if miss:
                continue
            row = base_features(cls, star_snr + float(rng.normal(0, 2)))
            if harmonic:
                row['bls_period_d'] = p * 2.0
                status, label = 'HARMONIC', 'HARMONIC'
            else:
                row['bls_period_d'] = p * (1.0 + 0.001)
                status, label = 'MATCHED', disp
            row.update({'signal_id': f'{sid}_{sig_idx}', 'target_id': sid, 'signal_index': sig_idx, 'mission': 'Kepler',
                        'label': label, 'match_status': status})
            feat_rows.append(row)
            sig_idx += 1
        row = base_features(0, float(rng.uniform(7.2, 12)))
        row['bls_period_d'] = float(rng.uniform(0.7, 60.0))
        row.update({'signal_id': f'{sid}_{sig_idx}', 'target_id': sid, 'signal_index': sig_idx, 'mission': 'Kepler',
                    'label': 'UNMATCHED', 'match_status': 'UNMATCHED'})
        feat_rows.append(row)
    for sid in groups['tr_u'] + groups['ctrl_u']:
        for k in range(1 + (int(sid) % 2)):
            row = base_features(0, float(rng.uniform(7.2, 14)))
            row['bls_period_d'] = float(rng.uniform(0.7, 60.0))
            row.update({'signal_id': f'{sid}_{k}', 'target_id': sid, 'signal_index': k, 'mission': 'Kepler',
                        'label': 'UNMATCHED', 'match_status': 'UNMATCHED'})
            feat_rows.append(row)

    df = pd.DataFrame(feat_rows)
    df['koi_fpflag_nt'] = 0
    df['koi_fpflag_ss'] = 0
    df['koi_fpflag_co'] = 0
    df['koi_fpflag_ec'] = 0
    df['n_signals_star'] = df.groupby('target_id')['signal_id'].transform('count')
    if with_version:
        df['pipeline_version'] = version
    df.to_csv(os.path.join(out_dir, f'features_demo{suffix}.csv'), index=False)
    pd.DataFrame(label_rows).to_csv(os.path.join(labels_dir, 'kepler_labels.csv'), index=False)

    split_rows = []
    for key, st_type, split in (('tr_m', 'MARKED', 'TRAIN'), ('ctrl_m', 'MARKED', 'CONTROL'), ('tr_u', 'UNMARKED', 'TRAIN'), ('ctrl_u', 'UNMARKED', 'CONTROL')):
        for sid in groups[key]:
            split_rows.append({'target_id': sid, 'star_type': st_type, 'split': split,
                               'primary_label': star_class[sid][0].replace(' ', '_') if sid in star_class else 'NONE'})
    pd.DataFrame(split_rows).to_csv(os.path.join(out_dir, f'demo_stars_split{suffix}.csv'), index=False)
    return {'root': root, 'out_dir': out_dir, 'groups': groups, 'features': df, 'koi_periods': koi_periods}
