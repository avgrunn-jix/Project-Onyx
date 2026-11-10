import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

import onyx_features
from synth_demo import SHOWCASE, build_workspace


def _hash_tree(root):
    out = {}
    for base, _, files in os.walk(root):
        for f in files:
            p = os.path.join(base, f)
            with open(p, 'rb') as fh:
                out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def test_failure_in_one_group_does_not_shrink_other_groups(demo):
    tr_m = [f'm{i}' for i in range(10)]
    ctrl_m = [SHOWCASE] + [f'c{i}' for i in range(9)]
    tr_u = [f'u{i}' for i in range(10)]
    ctrl_u = [f'v{i}' for i in range(10)]
    processed = set(tr_m + ctrl_m + tr_u + ctrl_u) - {'v1', 'v2', 'v3', 'v4'}
    fin = demo.finalize_groups(tr_m, ctrl_m, tr_u, ctrl_u, processed, SHOWCASE)
    assert len(fin['tr_m']) == 10
    assert len(fin['ctrl_m']) == 10
    assert len(fin['tr_u']) == 10
    assert len(fin['ctrl_u']) == 6
    assert fin['dropped'] == []
    assert SHOWCASE in fin['ctrl_m']


def test_balance_halves_restores_trimming_and_keeps_showcase(demo):
    tr_m = [f'm{i}' for i in range(10)]
    ctrl_m = [SHOWCASE] + [f'c{i}' for i in range(9)]
    tr_u = [f'u{i}' for i in range(10)]
    ctrl_u = [f'v{i}' for i in range(10)]
    processed = set(tr_m + ctrl_m + tr_u + ctrl_u) - {'v1', 'v2', 'v3', 'v4'}
    fin = demo.finalize_groups(tr_m, ctrl_m, tr_u, ctrl_u, processed, SHOWCASE, balance=True)
    assert len(fin['tr_m']) == len(fin['ctrl_m']) == len(fin['tr_u']) == len(fin['ctrl_u']) == 6
    assert fin['ctrl_m'][0] == SHOWCASE
    assert len(fin['dropped']) == 12


def test_split_stars_unbalanced_by_default(demo):
    marked = ['m0', SHOWCASE] + [f'm{i}' for i in range(1, 9)]
    unmarked = [f'u{i}' for i in range(7)]
    tr_m, ctrl_m, tr_u, ctrl_u = demo.split_stars(marked, unmarked, {}, SHOWCASE, 42)
    assert len(tr_m) == 5 and len(ctrl_m) == 5
    assert len(tr_u) == 4 and len(ctrl_u) == 3
    assert SHOWCASE in ctrl_m
    tr_m, ctrl_m, tr_u, ctrl_u = demo.split_stars(marked, unmarked, {}, SHOWCASE, 42, balance=True)
    assert len(tr_m) == len(ctrl_m) == len(tr_u) == len(ctrl_u) == 3


def test_processing_order_is_deterministic_by_seed(demo):
    stars = [str(i) for i in range(50)]
    a = demo.shuffled_order(stars, 42)
    b = demo.shuffled_order(stars, 42)
    c = demo.shuffled_order(stars, 43)
    assert a == b
    assert a != stars
    assert sorted(a) == sorted(stars)
    assert a != c
    assert stars == [str(i) for i in range(50)]


def test_failed_table_keeps_error_text_and_group(extractor, tmp_path):
    path = str(tmp_path / 'f_failed.csv')
    attempts = {}
    extractor.log_failure(path, '111', 'DownloadError: Download failed: boom', attempts, stage='download')
    extractor.log_failure(path, '222', 'Not enough valid points: files=3, raw=10, finite=9, clipped=8, min=500', attempts, stage='features')
    df = pd.read_csv(path, dtype={'target_id': str})
    df.loc[df.target_id == '111', ['star_type', 'split']] = ['MARKED', 'CONTROL']
    df.to_csv(path, index=False)
    extractor.update_failed_table(path, {'222'}, {'111': ('DownloadError: Download failed: still boom', 'download', 2)})
    out = pd.read_csv(path, dtype={'target_id': str})
    assert list(out['target_id']) == ['111']
    row = out.iloc[0]
    assert row['error'] == 'DownloadError: Download failed: still boom'
    assert row['stage'] == 'download'
    assert int(row['attempts']) == 2
    assert row['star_type'] == 'MARKED' and row['split'] == 'CONTROL'
    assert 'Persistent failure' not in open(path, encoding='utf-8').read()


def test_not_enough_points_message_format(extractor):
    msg = extractor.not_enough_points_message(17, 65432, 312, 310, 500)
    assert msg == 'Not enough valid points: files=17, raw=65432, finite=312, clipped=310, min=500'
    assert msg.startswith('Not enough valid points')


def test_annotate_failed_table_sets_star_type_and_split(demo, tmp_path):
    path = str(tmp_path / 'x_failed.csv')
    pd.DataFrame([
        {'target_id': '1', 'star_type': 'UNKNOWN', 'split': 'UNKNOWN', 'stage': 's', 'error': 'e1', 'attempts': 1},
        {'target_id': '2', 'star_type': 'UNKNOWN', 'split': 'UNKNOWN', 'stage': 's', 'error': 'e2', 'attempts': 1},
    ]).to_csv(path, index=False)
    gm = demo.plan_group_map(['1'], [], [], ['2'])
    assert demo.annotate_failed_table(path, gm) == 2
    df = pd.read_csv(path, dtype={'target_id': str})
    assert df.loc[df.target_id == '1', 'star_type'].iloc[0] == 'MARKED'
    assert df.loc[df.target_id == '1', 'split'].iloc[0] == 'TRAIN'
    assert df.loc[df.target_id == '2', 'star_type'].iloc[0] == 'UNMARKED'
    assert df.loc[df.target_id == '2', 'split'].iloc[0] == 'CONTROL'
    assert list(df['error']) == ['e1', 'e2']


def test_failed_stars_not_done_until_three_attempts(demo, tmp_path):
    feats = str(tmp_path / 'f.csv')
    diag = str(tmp_path / 'f_diag.csv')
    failed = str(tmp_path / 'f_failed.csv')
    pd.DataFrame({'target_id': ['10']}).to_csv(feats, index=False)
    pd.DataFrame([
        {'target_id': '21', 'attempts': 1},
        {'target_id': '22', 'attempts': 2},
        {'target_id': '23', 'attempts': 3},
    ]).to_csv(failed, index=False)
    done = demo.stars_done_for_run(feats, diag, failed)
    assert done == {'10', '23'}


def test_target_name_filter_is_exact_numeric(extractor):
    assert extractor.target_name_matches('4852528', 'kplr004852528')
    assert extractor.target_name_matches('4852528', 'KIC 4852528')
    assert not extractor.target_name_matches('485252', 'kplr004852528')
    assert not extractor.target_name_matches('4852528', 'kplr048525281')
    assert not extractor.target_name_matches('123', 'KIC 1234')


def test_eb_catalog_negative_control_is_kepler_80(demo, tmp_path):
    ids = [str(1000 + i) for i in range(6)]
    ok = tmp_path / 'ok.csv'
    pd.DataFrame({'kepid': ids}).to_csv(ok, index=False)
    demo.verify_eb_catalog(str(ok))
    bad = tmp_path / 'bad.csv'
    pd.DataFrame({'kepid': ids + ['4852528']}).to_csv(bad, index=False)
    with pytest.raises(RuntimeError):
        demo.verify_eb_catalog(str(bad))


def test_wait_for_disk_waits_then_succeeds_or_fails(extractor):
    state = {'t': 0.0, 'sleeps': []}

    def clock():
        return state['t']

    def sleep(s):
        state['sleeps'].append(s)
        state['t'] += s

    free_values = iter([0, 0, 10 * 1024 ** 3])
    ok = extractor.wait_for_disk(1.0, 600.0, 30.0, free_fn=lambda: next(free_values), sleep_fn=sleep, clock_fn=clock)
    assert ok is True
    assert state['sleeps'] == [30.0, 30.0]
    state['t'] = 0.0
    state['sleeps'] = []
    ok = extractor.wait_for_disk(1.0, 600.0, 30.0, free_fn=lambda: 0, sleep_fn=sleep, clock_fn=clock)
    assert ok is False
    assert sum(state['sleeps']) == 600.0
    assert all(s <= 30.0 for s in state['sleeps'])


def test_emergency_limit_triggers_from_worker_start_not_submit(extractor):
    active = {'f_running': ('1', 0.0), 'f_queued': ('2', 0.0), 'f_fresh': ('3', 0.0)}
    starts = {'1': 100.0, '3': 5000.0}
    expired = extractor.find_emergency_expired(active, starts, now=5000.0 + 10, star_timeout=1000.0)
    assert [e[1] for e in expired] == ['1']
    assert expired[0][2] == 100.0
    none = extractor.find_emergency_expired(active, starts, now=3100.0, star_timeout=1000.0)
    assert none == []


def test_version_blocks_resume_but_not_allow_legacy_or_report_only(tmp_path):
    expected = onyx_features.build_pipeline_version(True, 3.0, 4)
    p_old = tmp_path / 'old.csv'
    pd.DataFrame({'target_id': ['1'], 'x': [1]}).to_csv(p_old, index=False)
    p_other = tmp_path / 'other.csv'
    pd.DataFrame({'target_id': ['1'], 'pipeline_version': ['1|quick=0']}).to_csv(p_other, index=False)
    p_same = tmp_path / 'same.csv'
    pd.DataFrame({'target_id': ['1'], 'pipeline_version': [expected]}).to_csv(p_same, index=False)

    with pytest.raises(onyx_features.PipelineVersionError):
        onyx_features.check_pipeline_version(str(p_old), expected)
    with pytest.raises(onyx_features.PipelineVersionError):
        onyx_features.check_pipeline_version(str(p_other), expected)
    assert onyx_features.check_pipeline_version(str(p_same), expected) is None
    assert 'allow-legacy' in onyx_features.check_pipeline_version(str(p_old), expected, allow_legacy=True)
    assert onyx_features.check_pipeline_version(str(p_other), expected, report_only=True)
    assert onyx_features.check_pipeline_version(str(p_old), expected, report_only=True)
    assert onyx_features.check_pipeline_version(str(tmp_path / 'missing.csv'), expected) is None


def test_pipeline_version_depends_on_parameters():
    a = onyx_features.build_pipeline_version(True, 3.0, 4)
    assert a != onyx_features.build_pipeline_version(False, 3.0, 4)
    assert a != onyx_features.build_pipeline_version(True, 2.0, 4)
    assert a != onyx_features.build_pipeline_version(True, 3.0, 3)
    assert a.startswith(onyx_features.PIPELINE_VERSION)
    assert 'pipeline_version' in onyx_features.EXCLUDE_COLS


def _run_extractor_main(extractor, monkeypatch, tmp_path, extra):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / 'feat_quick.csv'
    pd.DataFrame({'target_id': ['1'], 'centroid_shift_col_sigma': [0.0], 'centroid_shift_row_sigma': [0.0],
                  'centroid_shift_total_sigma': [0.0], 'centroid_shift_total_arcsec': [0.0]}).to_csv(out, index=False)
    tf = tmp_path / 't.txt'
    tf.write_text('1\n', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['x', '--output', str(out), '--targets-file', str(tf), '--workers', '1',
                                     '--cache-dir', str(tmp_path / 'c'), '--min-free-gb', '0', '--quick'] + extra)

    class Reached(Exception):
        pass

    def stop(*a, **k):
        raise Reached()

    monkeypatch.setattr(extractor, 'fetch_labels', stop)
    return Reached


def test_extractor_resume_stops_on_version_mismatch(extractor, monkeypatch, tmp_path):
    _run_extractor_main(extractor, monkeypatch, tmp_path, [])
    with pytest.raises(SystemExit) as exc:
        extractor.main()
    assert exc.value.code == 1


def test_extractor_allow_legacy_continues_past_version_check(extractor, monkeypatch, tmp_path):
    reached = _run_extractor_main(extractor, monkeypatch, tmp_path, ['--allow-legacy'])
    with pytest.raises(reached):
        extractor.main()


def test_report_only_does_no_extraction_or_download(demo, monkeypatch, tmp_path):
    ws = build_workspace(tmp_path, with_version=False)
    out_dir = ws['out_dir']
    monkeypatch.chdir(tmp_path)
    before = _hash_tree(out_dir)

    def forbidden(*a, **k):
        raise AssertionError('network or subprocess call in report-only mode')

    import urllib.request
    import subprocess
    monkeypatch.setattr(demo, 'run_child', forbidden)
    monkeypatch.setattr(demo, 'generate_showcase_figure', forbidden)
    monkeypatch.setattr(demo, 'fetch_stellar_params', forbidden)
    monkeypatch.setattr(demo, 'get_unmarked_pool', forbidden)
    monkeypatch.setattr(demo, 'match_unmarked_stars', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(sys, 'argv', ['x', '--quick', '--report-only', '--output-dir', out_dir, '--showcase-star', SHOWCASE, '--cache-dir', str(tmp_path / 'c')])
    demo.main()

    after = _hash_tree(out_dir)
    for k, v in before.items():
        assert after[k] == v
    new_files = set(after) - set(before)
    assert 'summary_quick_report.md' in new_files
    assert all('_report' in f or f.startswith('figures') for f in new_files)
    text = open(os.path.join(out_dir, 'summary_quick_report.md'), encoding='utf-8').read()
    assert '--report-only' in text
    assert 'pipeline_version' in text
