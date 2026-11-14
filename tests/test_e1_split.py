import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

import e1_split
import e2_compare
import exp_common as ec
from onyx_features import compute_feature_columns

HEADER = ['signal_id', 'target_id', 'signal_index', 'pipeline_version', 'mission', 'label', 'match_status', 'bls_period_d', 'bls_epoch', 'bls_duration_d', 'bls_depth_ppm', 'bls_depth_snr', 'bls_power_peak', 'bls_sde', 'bls_sde_raw', 'transit_count', 'n_transits_observed', 'single_event_fraction', 'dur_at_grid_edge', 'duty_cycle', 'odd_even_depth_diff_ppm', 'odd_even_sigma', 'odd_even_z', 'secondary_depth_ppm', 'secondary_snr', 'secondary_z', 'secondary_phase', 'implied_radius_rjup', 'duration_ratio', 'eb_period_match', 'harmonic_power_ratio', 'shape_v_u_ratio', 'in_transit_scatter_ppm', 'depth_scatter_ppm', 'flux_std_ppm', 'flux_mad_ppm', 'flux_skew', 'flux_kurtosis', 'cdpp_ppm', 'pointwise_scatter_ppm', 'autocorr_max', 'ls_peak_power', 'gap_fraction', 'baseline_d', 'star_teff', 'star_radius', 'star_logg', 'centroid_shift_col_sigma', 'centroid_shift_row_sigma', 'centroid_shift_total_sigma', 'centroid_shift_total_arcsec', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'n_signals_star']
FORBIDDEN = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'baseline_d', 'n_quarters', 'n_files', 'signal_id', 'target_id', 'signal_index', 'pipeline_version', 'mission', 'label', 'match_status', 'n_signals_star', 'bls_epoch', 'bls_sde_raw', 'target_bin']
REAL_OUT = ec.OUT_DIR


def build_run(root, n_marked=80, n_unmarked=30, n_failed=4, seed=0):
    rng = np.random.RandomState(seed)
    fd = Path(root) / ec.FEATURES_SUBDIR
    fd.mkdir(parents=True)
    rows, diag, failed = [], [], []
    stars = [str(1000 + i) for i in range(n_marked)] + [ec.SHOWCASE_ID]
    for k, tid in enumerate(stars):
        diag.append({'target_id': tid, 'тип': 'Отмеченная (KOI)'})
        if tid == ec.SHOWCASE_ID:
            plan = [('MATCHED', 'CONFIRMED')] * 4
        elif k % 10 == 0:
            plan = [('MATCHED', 'CONFIRMED'), ('MATCHED', 'FALSE POSITIVE')]
        elif k % 7 == 0:
            plan = [('MATCHED', 'CANDIDATE')]
        elif k % 11 == 0:
            plan = [('UNMATCHED', 'CONFIRMED')]
        elif k % 2 == 0:
            plan = [('MATCHED', 'CONFIRMED'), ('HARMONIC', 'CONFIRMED'), ('MATCHED', 'CANDIDATE')]
        else:
            plan = [('MATCHED', 'FALSE POSITIVE')] * (1 + k % 3)
        for j, (st, lab) in enumerate(plan):
            rows.append((tid, j, st, lab))
    for i in range(n_unmarked):
        tid = str(5000 + i)
        diag.append({'target_id': tid, 'тип': 'Неотмеченная (Control)'})
        if i < n_failed:
            failed.append({'target_id': tid, 'star_type': 'UNMARKED', 'split': 'TRAIN', 'stage': 'features', 'error': 'Not enough valid points', 'attempts': 2})
            continue
        rows.append((tid, 0, 'UNMATCHED', 'UNMARKED'))
    recs = []
    for tid, j, st, lab in rows:
        r = {c: float(rng.rand()) for c in HEADER}
        r.update({'signal_id': f"{tid}_{j}", 'target_id': tid, 'signal_index': j, 'pipeline_version': 'x', 'mission': 'Kepler', 'label': lab, 'match_status': st})
        recs.append(r)
    pd.DataFrame(recs, columns=HEADER).to_csv(fd / 'features_demo.csv', index=False)
    pd.DataFrame(recs[:3], columns=HEADER).to_csv(fd / 'features_demo_inverted.csv', index=False)
    pd.DataFrame(failed, columns=['target_id', 'star_type', 'split', 'stage', 'error', 'attempts']).to_csv(fd / 'features_demo_failed.csv', index=False)
    pd.DataFrame(diag).to_csv(fd / 'diagnostics_stars.csv', index=False)
    pd.DataFrame({'target_id': [d['target_id'] for d in diag]}).to_csv(fd / 'demo_stars_split.csv', index=False)
    (fd / 'summary.md').write_text('x', encoding='utf-8')
    (Path(root) / 'run_commit.txt').write_text('abc\n', encoding='utf-8')
    (Path(root) / 'versions.txt').write_text('numpy==0\n', encoding='utf-8')
    return Path(root)


@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp('e1run')
    run = build_run(root / 'run')
    out = root / 'out'
    stars, registry, lab = e1_split.run(run, out, seed=42, identity_check=False)
    return {'run': run, 'out': out, 'stars': stars, 'registry': registry, 'lab': lab}


def test_parts_do_not_overlap_and_star_is_whole(synth):
    stars = synth['stars']
    lab = stars[stars['роль'] == e1_split.ROLE_LABELED]
    assert lab['target_id'].is_unique
    assert set(lab['часть_полная']) <= set(ec.PARTS)
    sig = synth['lab']
    assert (sig.groupby('target_id')['part'].nunique() == 1).all()
    test = set(pd.read_csv(synth['out'] / e1_split.TEST_IDS_FILE, dtype={'target_id': str})['target_id'])
    calib = set(pd.read_csv(synth['out'] / e1_split.CALIB_IDS_FILE, dtype={'target_id': str})['target_id'])
    pub = pd.read_csv(synth['out'] / ec.SPLIT_STARS_FILE, dtype={'target_id': str})
    train = set(pub.loc[pub['часть'] == ec.PART_TRAIN, 'target_id'])
    val = set(pub.loc[pub['часть'] == ec.PART_VAL, 'target_id'])
    sets = [train, val, calib, test]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (sets[i] & sets[j])


def test_fractions_per_stratum(synth):
    stars = synth['stars']
    lab = stars[stars['роль'] == e1_split.ROLE_LABELED]
    for stratum, grp in lab.groupby('слой'):
        counts = [int((grp['часть_полная'] == p).sum()) for p in ec.PARTS]
        assert counts == e1_split.allocate_counts(len(grp), e1_split.FRACTIONS)
        for c, fr in zip(counts, e1_split.FRACTIONS):
            assert abs(c - fr * len(grp)) < 1.0 + 1e-9


def test_allocate_counts_examples():
    assert e1_split.allocate_counts(316, (0.5, 0.15, 0.15, 0.2)) == [158, 48, 47, 63]
    assert e1_split.allocate_counts(3, (0.5, 0.15, 0.15, 0.2)) == [2, 0, 0, 1]
    assert sum(e1_split.allocate_counts(207, (0.5, 0.15, 0.15, 0.2))) == 207


def test_split_reproducible_by_seed(synth, tmp_path):
    e1_split.run(synth['run'], tmp_path / 'again', seed=42, identity_check=False)
    a = (synth['out'] / ec.SPLIT_STARS_FILE).read_bytes()
    assert a == (tmp_path / 'again' / ec.SPLIT_STARS_FILE).read_bytes()
    e1_split.run(synth['run'], tmp_path / 'other', seed=7, identity_check=False)
    assert a != (tmp_path / 'other' / ec.SPLIT_STARS_FILE).read_bytes()


def test_public_split_hides_calibration_and_test(synth):
    pub = pd.read_csv(synth['out'] / ec.SPLIT_STARS_FILE, dtype={'target_id': str})
    assert ec.PART_TEST not in set(pub['часть'].dropna())
    assert ec.PART_CALIB not in set(pub['часть'].dropna())
    reg = pd.read_csv(synth['out'] / 'e1_signal_registry.csv', dtype={'target_id': str})
    assert ec.PART_TEST not in set(reg['часть'].dropna())
    test = pd.read_csv(synth['out'] / e1_split.TEST_IDS_FILE, dtype={'target_id': str})
    assert len(test) > 0


def test_registry_exclusions(synth):
    reg = synth['registry']
    k80 = reg[reg['target_id'] == ec.SHOWCASE_ID]
    assert len(k80) == 4 and (k80['причина_исключения'] == 'витринная система').all() and (k80['в_разметке'] == 0).all()
    assert (reg.loc[reg['label'] == 'CANDIDATE', 'в_разметке'] == 0).all()
    assert set(reg.loc[reg['в_разметке'] == 1, 'label']) <= {'CONFIRMED', 'FALSE POSITIVE'}
    assert (reg.loc[reg['в_разметке'] == 1, 'match_status'] == 'MATCHED').all()
    assert set(reg['причина_исключения'].fillna('')) <= {'', 'CANDIDATE', 'HARMONIC', 'UNMATCHED', 'витринная система'}
    noted = reg[reg['примечание'] != '']
    assert (noted['причина_исключения'] == 'CANDIDATE').all() and (noted['часть'] == ec.PART_TRAIN).all()
    stars = synth['stars']
    assert stars.loc[stars['target_id'] == ec.SHOWCASE_ID, 'часть_полная'].iat[0] == ''
    assert (stars.loc[stars['роль'] == e1_split.ROLE_FAILED, 'часть_полная'] == '').all()


def test_hash_part_is_deterministic():
    a = [e1_split.hash_part(str(t), 42, e1_split.FRACTIONS) for t in range(2000)]
    b = [e1_split.hash_part(str(t), 42, e1_split.FRACTIONS) for t in range(2000)]
    assert a == b
    share = np.mean([x == ec.PART_TRAIN for x in a])
    assert 0.45 < share < 0.55


def check_e2_dataset(out, run):
    test = set(pd.read_csv(Path(out) / e1_split.TEST_IDS_FILE, dtype={'target_id': str})['target_id'])
    calib = set(pd.read_csv(Path(out) / e1_split.CALIB_IDS_FILE, dtype={'target_id': str})['target_id'])
    df, cols = e2_compare.load_labeled_part(out, run, ec.PART_TRAIN)
    val, _ = e2_compare.load_labeled_part(out, run, ec.PART_VAL)
    ids = set(df['target_id'])
    assert len(df) > 0
    assert not (ids & test) and not (ids & calib)
    assert not (set(val['target_id']) & (test | calib | ids))
    assert ec.SHOWCASE_ID not in ids
    assert set(df['label']) <= {'CONFIRMED', 'FALSE POSITIVE'}
    return df, cols


def test_e2_dataset_has_no_calibration_or_test_stars(synth):
    check_e2_dataset(synth['out'], synth['run'])
    with pytest.raises(ValueError):
        e2_compare.load_labeled_part(synth['out'], synth['run'], ec.PART_TEST)


def test_e2_code_does_not_open_hidden_files():
    for name in ['e2_compare.py', 'exp_common.py']:
        src = (ROOT / 'experiments' / name).read_text(encoding='utf-8')
        assert 'split_test_ids' not in src and 'split_calib_ids' not in src
        assert 'TEST_IDS_FILE' not in src and 'CALIB_IDS_FILE' not in src
        assert 'e1_signal_registry' not in src


@pytest.mark.skipif(not (REAL_OUT / ec.SPLIT_STARS_FILE).exists() or not (ec.features_dir(ec.RUN_DIR) / 'features_demo.csv').exists(), reason='2000-star run or its split is not present')
def test_e2_dataset_real_run():
    df, cols = check_e2_dataset(REAL_OUT, ec.RUN_DIR)
    assert len(cols) == 38


def test_feature_columns_match_main_model():
    cols = ec.model_feature_columns(HEADER + ['n_files', 'n_quarters'])
    assert cols == compute_feature_columns(HEADER, exclude_coverage=True)
    assert len(cols) == 38
    for c in FORBIDDEN:
        assert c not in cols
    assert not any(c.startswith('koi_') for c in cols)
    src = (ROOT / '3_demo_pipeline.py').read_text(encoding='utf-8')
    assert 'feature_cols = compute_feature_columns(df_features, exclude_coverage=True)' in src


def test_e1_feature_list_file(synth):
    import json
    data = json.loads((synth['out'] / ec.FEATURE_LIST_FILE).read_text(encoding='utf-8'))
    assert len(data['features']) == 38
    assert not set(data['features']) & set(FORBIDDEN)
