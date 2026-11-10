import pandas as pd
import onyx_features


REAL_HEADER = (
    "signal_id,target_id,signal_index,mission,label,match_status,bls_period_d,"
    "bls_epoch,bls_duration_d,bls_depth_ppm,bls_depth_snr,bls_power_peak,bls_sde,"
    "transit_count,duty_cycle,odd_even_depth_diff_ppm,odd_even_sigma,"
    "secondary_depth_ppm,secondary_snr,harmonic_power_ratio,shape_v_u_ratio,"
    "in_transit_scatter_ppm,depth_scatter_ppm,flux_std_ppm,flux_mad_ppm,flux_skew,"
    "flux_kurtosis,cdpp_ppm,pointwise_scatter_ppm,autocorr_max,ls_peak_power,"
    "gap_fraction,baseline_d,star_teff,star_radius,star_logg,"
    "centroid_shift_col_sigma,centroid_shift_row_sigma,centroid_shift_total_sigma,"
    "centroid_shift_total_arcsec,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,"
    "koi_fpflag_ec,n_signals_star"
).split(',')

EXTRACTOR_COLUMNS = REAL_HEADER + ['target_bin']

LEGACY_ML_FEATURES = [
    'bls_period_d', 'bls_duration_d', 'bls_depth_ppm', 'bls_depth_snr',
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

LEGACY_DEMO_FEATURES = [
    'bls_period_d', 'bls_duration_d', 'bls_depth_ppm', 'bls_depth_snr',
    'bls_power_peak', 'bls_sde', 'transit_count', 'duty_cycle',
    'odd_even_depth_diff_ppm', 'odd_even_sigma', 'secondary_depth_ppm',
    'secondary_snr', 'harmonic_power_ratio', 'shape_v_u_ratio',
    'in_transit_scatter_ppm', 'depth_scatter_ppm', 'flux_std_ppm',
    'flux_mad_ppm', 'flux_skew', 'flux_kurtosis', 'cdpp_ppm',
    'pointwise_scatter_ppm', 'autocorr_max', 'ls_peak_power', 'gap_fraction',
    'star_teff', 'star_radius', 'star_logg',
    'centroid_shift_col_sigma', 'centroid_shift_row_sigma',
    'centroid_shift_total_sigma', 'centroid_shift_total_arcsec',
]

LEGACY_EXCLUDE_COLS = [
    'signal_id', 'target_id', 'signal_index', 'mission', 'label',
    'match_status', 'target_bin', 'koi_fpflag_nt', 'koi_fpflag_ss',
    'koi_fpflag_co', 'koi_fpflag_ec', 'p_candidate', 'n_signals_star',
]

LEGACY_COVERAGE_COLS = [
    'baseline_d', 'n_files', 'n_quarters', 'quarters', 'st_quarters',
    'dataspan', 'cadence_days', 'mean_cadence_min',
]


def test_equivalence_ml_pipeline_schema():
    df = pd.DataFrame(columns=EXTRACTOR_COLUMNS)
    actual = onyx_features.compute_feature_columns(df, exclude_coverage=False)
    assert len(REAL_HEADER) == 45
    assert len(actual) == 33
    assert actual == LEGACY_ML_FEATURES
    assert 'baseline_d' in actual


def test_equivalence_demo_pipeline_schema():
    cols = EXTRACTOR_COLUMNS + ['n_files', 'n_quarters']
    df = pd.DataFrame(columns=cols)
    actual = onyx_features.compute_feature_columns(df, exclude_coverage=True)
    assert len(actual) == 32
    assert actual == LEGACY_DEMO_FEATURES
    for cov in LEGACY_COVERAGE_COLS:
        assert cov not in actual


def test_bls_epoch_and_pipeline_version_are_not_features():
    df = pd.DataFrame(columns=EXTRACTOR_COLUMNS + ['pipeline_version'])
    for cov_flag in (False, True):
        feats = onyx_features.compute_feature_columns(df, exclude_coverage=cov_flag)
        assert 'bls_epoch' not in feats
        assert 'pipeline_version' not in feats


def test_equivalence_transfer_intersection():
    train_cols = EXTRACTOR_COLUMNS
    test_cols = [c for c in EXTRACTOR_COLUMNS if c not in ('star_logg', 'koi_fpflag_nt')]
    df_train = pd.DataFrame(columns=train_cols)
    df_test = pd.DataFrame(columns=test_cols)
    expected = [c for c in LEGACY_ML_FEATURES if c != 'star_logg']
    actual = onyx_features.compute_feature_columns(df_train, exclude_coverage=False, other_cols=df_test.columns)
    assert actual == expected
    assert 'star_logg' not in actual


def test_no_koi_columns_ever_included():
    extra_koi = [
        'koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_period',
        'koi_time0bk', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
        'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad',
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
        'KOI_CUSTOM_FLAG', 'kepoi_name', 'p_candidate',
    ]
    df = pd.DataFrame(columns=EXTRACTOR_COLUMNS + extra_koi)
    for cov_flag in (False, True):
        feats = onyx_features.compute_feature_columns(df, exclude_coverage=cov_flag)
        assert len(feats) > 0
        for col in feats:
            assert not col.lower().startswith('koi_')
            assert col.lower() != 'kepoi_name'
            assert col not in LEGACY_EXCLUDE_COLS


def test_pipelines_import_shared_helper(ml_pipeline, demo):
    assert ml_pipeline.compute_feature_columns is onyx_features.compute_feature_columns
    assert demo.compute_feature_columns is onyx_features.compute_feature_columns
