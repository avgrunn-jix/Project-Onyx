import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_compare():
    spec = importlib.util.spec_from_file_location('onyx_compare_bench', str(ROOT / 'bench' / 'compare_bench.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_run(run_dir, feats, diag):
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feats).to_csv(run_dir / 'extracted_features.csv', index=False, encoding='utf-8')
    pd.DataFrame(diag).to_csv(run_dir / 'extracted_features_diagnostics.csv', index=False, encoding='utf-8')


def test_report_prints_depth_and_observed_transits_per_period(tmp_path):
    cb = load_compare()
    new_dir = tmp_path / 'new'
    old_dir = tmp_path / 'old'
    write_run(new_dir, [
        {'target_id': '5780885', 'bls_period_d': 94.1761, 'bls_depth_ppm': 101.71, 'n_transits_observed': 13},
        {'target_id': '5780885', 'bls_period_d': 4.88543, 'bls_depth_ppm': 6398.63, 'n_transits_observed': 276},
    ], [{'target_id': '5780885', 'status': 'OK', 'time_total_s': 10.0, 'alias_factors': ''}])
    write_run(old_dir, [
        {'target_id': '5780885', 'bls_period_d': 2.44274, 'bls_depth_ppm': 41.16},
    ], [{'target_id': '5780885', 'status': 'OK', 'time_total_s': 12.0, 'alias_factors': ''}])
    report = cb.build_report(str(new_dir), str(old_dir))
    lines = report.splitlines()
    assert '  периоды new: 4.8854, 94.1761' in lines
    assert '  периоды old: 2.4427' in lines
    assert '  глубина и число транзитов new: 4.8854: bls_depth_ppm=6398.6, n_transits_observed=276; 94.1761: bls_depth_ppm=101.7, n_transits_observed=13' in lines
    assert '  глубина и число транзитов old: 2.4427: bls_depth_ppm=41.2, n_transits_observed=н/д' in lines
    assert lines.index('  периоды old: 2.4427') + 1 == lines.index('  глубина и число транзитов new: 4.8854: bls_depth_ppm=6398.6, n_transits_observed=276; 94.1761: bls_depth_ppm=101.7, n_transits_observed=13')
    for old_line in ('Звезда 5780885', '  статус: new=OK | old=OK', '  сигналов: new=2 | old=1', '  alias_factors (new): -', '=== Звезды, где в новом режиме сигналов меньше, чем в старом ===', '=== Звезды с непустыми alias_factors ==='):
        assert old_line in lines


def test_report_handles_star_without_signals(tmp_path):
    cb = load_compare()
    new_dir = tmp_path / 'new'
    old_dir = tmp_path / 'old'
    write_run(new_dir, [{'target_id': '1', 'bls_period_d': 3.0, 'bls_depth_ppm': 500.0, 'n_transits_observed': 30}], [{'target_id': '1', 'status': 'OK'}, {'target_id': '2', 'status': 'OK'}])
    old_dir.mkdir()
    report = cb.build_report(str(new_dir), str(old_dir))
    assert '  глубина и число транзитов new: 3.0000: bls_depth_ppm=500.0, n_transits_observed=30' in report
    assert report.count('  глубина и число транзитов old: -') == 2
    assert report.count('  глубина и число транзитов new: -') == 1


def test_report_prints_restore_fields_only_when_present(tmp_path):
    cb = load_compare()
    new_dir = tmp_path / 'new'
    old_dir = tmp_path / 'old'
    feats = [{'target_id': '5780885', 'bls_period_d': 4.88543, 'bls_depth_ppm': 6398.63, 'n_transits_observed': 276}]
    write_run(new_dir, feats, [
        {'target_id': '5780885', 'status': 'OK', 'time_total_s': 10.0, 'alias_factors': '', 'restore_pool_points': 2790, 'restored_points': '2789;0;0'},
        {'target_id': '11442793', 'status': 'OK', 'time_total_s': 11.0, 'alias_factors': '', 'restore_pool_points': 0, 'restored_points': ''},
        {'target_id': '6541920', 'status': 'OK', 'time_total_s': 11.0, 'alias_factors': '', 'restore_pool_points': 35, 'restored_points': '12'},
    ])
    write_run(old_dir, feats, [{'target_id': '5780885', 'status': 'OK', 'time_total_s': 12.0, 'alias_factors': ''}])
    lines = cb.build_report(str(new_dir), str(old_dir)).splitlines()
    assert '  restore_pool_points: new=2790 | old=н/д' in lines
    assert '  restored_points: new=2789;0;0 | old=н/д' in lines
    assert '  restore_pool_points: new=0 | old=н/д' in lines
    assert '  restored_points: new=- | old=н/д' in lines
    assert '  restore_pool_points: new=35 | old=н/д' in lines
    assert '  restored_points: new=12 | old=н/д' in lines
    i = lines.index('  restore_pool_points: new=2790 | old=н/д')
    assert lines[i - 1].startswith('  глубина и число транзитов old: ')
    assert lines[i + 1] == '  restored_points: new=2789;0;0 | old=н/д'
    assert lines[i + 2] == '  alias_factors (new): -'
    swapped = cb.build_report(str(old_dir), str(new_dir)).splitlines()
    assert '  restore_pool_points: new=н/д | old=2790' in swapped
    assert '  restored_points: new=н/д | old=2789;0;0' in swapped
    both_old = cb.build_report(str(old_dir), str(old_dir))
    assert 'restore_pool_points' not in both_old
    assert 'restored_points' not in both_old
