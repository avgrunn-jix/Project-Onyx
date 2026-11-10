import os
import sys

import numpy as np
import pandas as pd

import onyx_report
from synth_demo import SHOWCASE, build_workspace


def _inputs():
    rows = []
    sig = []
    for i in range(8):
        tid = f's{i}'
        rows.append({'target_id': tid, 'label': 'CONFIRMED', 'catalog_period': 2.0 + i, 'koi_depth': 100.0 * (i + 1)})
        if i % 2 == 0:
            sig.append({'target_id': tid, 'bls_period_d': 2.0 + i, 'decision': 'CANDIDATE'})
    for k in range(3):
        rows.append({'target_id': 'm3', 'label': 'CANDIDATE', 'catalog_period': 5.0 + 7 * k, 'koi_depth': 50.0})
    for k in range(5):
        rows.append({'target_id': 'm5', 'label': 'FALSE POSITIVE', 'catalog_period': 1.5 + 9 * k, 'koi_depth': np.nan})
    rows.append({'target_id': 'm5', 'label': 'FALSE POSITIVE', 'catalog_period': 300.0, 'koi_depth': 10.0})
    sig.append({'target_id': 'm3', 'bls_period_d': 5.0, 'decision': 'REVIEW'})
    return pd.DataFrame(rows), pd.DataFrame(sig)


def test_recovery_has_depth_and_multiplicity_columns():
    koi, sig = _inputs()
    rec = onyx_report.build_koi_recovery(koi, sig)
    assert {'koi_depth', 'n_koi_star'} <= set(rec.columns)
    assert set(rec.loc[rec['target_id'] == 'm5', 'n_koi_star']) == {6}
    assert set(rec.loc[rec['target_id'] == 'm3', 'n_koi_star']) == {3}
    assert set(rec.loc[rec['target_id'] == 's0', 'n_koi_star']) == {1}


def test_depth_quartiles_and_multiplicity_bins():
    koi, sig = _inputs()
    summ = onyx_report.summarize_koi_recovery(onyx_report.build_koi_recovery(koi, sig))
    by_depth = summ['by_depth']
    assert summ['n_depth_known'] == 11
    assert len(by_depth) == 4
    assert int(by_depth['KOI в диапазоне'].sum()) == 11
    assert by_depth['Группа'].iloc[0].startswith('Q1: 50')
    by_mult = summ['by_multiplicity'].set_index('Группа')
    assert by_mult.loc['1 KOI у звезды', 'KOI в диапазоне'] == 8
    assert by_mult.loc['1 KOI у звезды', 'Точно'].startswith('4/8 = 50.0%')
    assert by_mult.loc['2–3 KOI у звезды', 'KOI в диапазоне'] == 3
    assert by_mult.loc['2–3 KOI у звезды', 'Точно'].startswith('1/3')
    assert by_mult.loc['4 и больше KOI у звезды', 'KOI в диапазоне'] == 5
    assert int(summ['by_multiplicity']['KOI в диапазоне'].sum()) == summ['n_in_range']


def test_depth_breakdown_absent_without_column():
    koi, sig = _inputs()
    summ = onyx_report.summarize_koi_recovery(onyx_report.build_koi_recovery(koi.drop(columns=['koi_depth']), sig))
    assert len(summ['by_depth']) == 0
    assert summ['n_depth_known'] == 0
    assert len(summ['by_multiplicity']) == 3


def test_catalog_depth_never_becomes_a_feature():
    import onyx_features
    assert onyx_features.compute_feature_columns(['koi_depth', 'n_koi_star', 'bls_depth_ppm']) == ['n_koi_star', 'bls_depth_ppm']


def test_demo_summary_shows_new_breakdowns(demo, monkeypatch, tmp_path):
    ws = build_workspace(tmp_path)
    labels_path = os.path.join(ws['root'], 'data', 'labels', 'kepler_labels.csv')
    lab = pd.read_csv(labels_path, dtype={'target_id': str})
    lab['koi_depth'] = np.linspace(80.0, 5000.0, len(lab))
    lab.to_csv(labels_path, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--quick', '--report-only', '--output-dir', ws['out_dir'], '--showcase-star', SHOWCASE, '--cache-dir', str(tmp_path / 'c')])
    demo.main()
    text = open(os.path.join(ws['out_dir'], 'summary_quick_report.md'), encoding='utf-8').read()
    assert 'По глубине транзита (koi_depth' in text
    assert '| Q1: ' in text and '| Q4: ' in text
    assert 'По числу KOI у звезды' in text
    assert '| 1 KOI у звезды |' in text
    assert 'Разбивка не построена' not in text
