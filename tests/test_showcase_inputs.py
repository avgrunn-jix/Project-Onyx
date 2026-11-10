import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import onyx_showcase
from conftest import load_module
from test_showcase import all_texts, make_data

CYR = re.compile('[Ѐ-ӿ]')
REAL_HEADER = ['target_id', 'star_type', 'signal_id', 'period_d', 'depth_ppm', 'catalog_label', 'p_mean', 'p_std',
               'decision', 'possible_artifact', 'is_kepler_eb', 'shap_top3']


def real_table(rows):
    out = []
    for i, (tid, period, p_mean, decision) in enumerate(rows):
        out.append({'target_id': tid, 'star_type': 'MARKED', 'signal_id': f'{tid}_{i}', 'period_d': period,
                    'depth_ppm': 1000.0, 'catalog_label': 'CONFIRMED', 'p_mean': p_mean, 'p_std': 0.05,
                    'decision': decision, 'possible_artifact': False, 'is_kepler_eb': False, 'shap_top3': 'a;b;c'})
    return pd.DataFrame(out, columns=REAL_HEADER)


@pytest.fixture(scope='module')
def ms():
    return load_module('make_showcase.py', 'onyx_make_showcase')


def test_reads_real_header_with_period_d(ms, tmp_path):
    f = tmp_path / 's.csv'
    real_table([('4852528', 9.5216, 0.84, 'REVIEW'), ('99', 9.5216, 0.1, 'CANDIDATE'), ('4852528', 7.05, 0.4, 'CANDIDATE')]).to_csv(f, index=False)
    scored = ms.read_scored_signals(str(f))
    assert list(scored['target_id'].unique()) == ['4852528']
    sig = pd.DataFrame([dict(bls_period_d=9.5215), dict(bls_period_d=3.0)])
    out = onyx_showcase.match_verdicts(sig, scored)
    assert out.loc[0, 'p_mean'] == pytest.approx(0.84)
    assert out.loc[0, 'decision'] == 'REVIEW'
    assert pd.isnull(out.loc[1, 'p_mean'])
    assert pd.isnull(out.loc[1, 'decision'])


def test_bls_period_d_fallback(ms, tmp_path):
    f = tmp_path / 's.csv'
    df = real_table([('4852528', 9.5216, 0.84, 'CANDIDATE')]).rename(columns={'period_d': 'bls_period_d'})
    df.to_csv(f, index=False)
    scored = ms.read_scored_signals(str(f))
    out = onyx_showcase.match_verdicts(pd.DataFrame([dict(bls_period_d=9.5215)]), scored)
    assert out.loc[0, 'decision'] == 'CANDIDATE'


def test_missing_required_columns_raises_with_found_columns(ms, tmp_path):
    f = tmp_path / 's.csv'
    real_table([('4852528', 9.5, 0.8, 'REVIEW')]).drop(columns=['p_mean']).to_csv(f, index=False)
    with pytest.raises(ValueError) as err:
        ms.read_scored_signals(str(f))
    msg = str(err.value)
    assert 'p_mean' in msg
    assert 'period_d' in msg and 'shap_top3' in msg


def test_missing_period_column_raises(ms, tmp_path):
    f = tmp_path / 's.csv'
    real_table([('4852528', 9.5, 0.8, 'REVIEW')]).drop(columns=['period_d']).to_csv(f, index=False)
    with pytest.raises(ValueError):
        ms.read_scored_signals(str(f))


@pytest.mark.parametrize('value,expected', [
    ('CANDIDATE', 'CANDIDATE'),
    ('false_positive', 'FALSE_POSITIVE'),
    ('Review', 'REVIEW'),
    ('кандидат', 'CANDIDATE'),
    ('Кандидат', 'CANDIDATE'),
    ('ложное срабатывание', 'FALSE_POSITIVE'),
    ('ЛОЖНОЕ СРАБАТЫВАНИЕ', 'FALSE_POSITIVE'),
    ('на проверку', 'REVIEW'),
    ('ОЦЕНКА_НЕ_ПРОВОДИЛАСЬ', None),
    ('whatever', None),
    (float('nan'), None),
    (None, None),
])
def test_normalize_verdict(value, expected):
    assert onyx_showcase.normalize_verdict(value) == expected


def test_nan_p_mean_gives_na_and_unknown_verdict_gives_na(tmp_path):
    scored = real_table([('1', 9.5215, float('nan'), 'ОЦЕНКА_НЕ_ПРОВОДИЛАСЬ')])
    data = make_data(1)
    data['signals'] = onyx_showcase.match_verdicts(data['signals'].drop(columns=['p_mean', 'decision']), scored)
    for lang in ('ru', 'en'):
        fig = onyx_showcase.render_showcase(data, str(tmp_path / f'{lang}.png'), layout='wide', lang=lang, close=False)
        titles = [ax.get_title() for ax in fig.axes]
        plt.close(fig)
        assert any('p = n/a, n/a' in t for t in titles)


@pytest.mark.parametrize('layout', ['wide', 'tall'])
def test_english_figure_has_no_cyrillic_with_russian_verdicts(tmp_path, layout):
    scored = real_table([
        ('1', 9.5215, 0.9, 'кандидат'),
        ('1', 7.0535, 0.1, 'ложное срабатывание'),
        ('1', 4.6454, 0.5, 'на проверку'),
        ('1', 3.0721, 0.5, 'ОЦЕНКА_НЕ_ПРОВОДИЛАСЬ'),
    ])
    data = make_data()
    data['signals'] = onyx_showcase.match_verdicts(data['signals'].drop(columns=['p_mean', 'decision']), scored)
    fig = onyx_showcase.render_showcase(data, str(tmp_path / 'e.png'), layout=layout, lang='en', close=False)
    texts = all_texts(fig)
    plt.close(fig)
    assert not [t for t in texts if CYR.search(t)]
    assert any('CANDIDATE' in t for t in texts)
    assert any('FALSE_POSITIVE' in t for t in texts)


def test_raw_russian_decision_in_english_figure_is_cleaned(tmp_path):
    data = make_data()
    data['signals'].loc[0, 'decision'] = 'кандидат'
    data['signals'].loc[1, 'decision'] = 'ОЦЕНКА_НЕ_ПРОВОДИЛАСЬ'
    fig = onyx_showcase.render_showcase(data, str(tmp_path / 'e.png'), layout='wide', lang='en', close=False)
    texts = all_texts(fig)
    plt.close(fig)
    assert not [t for t in texts if CYR.search(t)]


def test_lang_both_computes_once_and_writes_two_files(ms, tmp_path, monkeypatch):
    feats = tmp_path / 'f.csv'
    pd.DataFrame({'target_id': ['4852528'] * 4, 'signal_index': [0, 1, 2, 3],
                  'bls_period_d': [9.5215, 7.0535, 4.6454, 3.0721], 'bls_sde': [12.0] * 4,
                  'bls_epoch': [105.0, 103.0, 104.0, 101.5], 'bls_duration_d': [0.12, 0.1, 0.09, 0.08],
                  'bls_depth_ppm': [1351.0, 1000.0, 700.0, 500.0]}).to_csv(feats, index=False)
    sigs = tmp_path / 'c.csv'
    real_table([('4852528', 9.5215, 0.84, 'REVIEW')]).to_csv(sigs, index=False)
    base = make_data()
    calls = {'n': 0}

    def fake_build(extractor, star_id, cache_dir, clip_upward=False):
        calls['n'] += 1
        return {k: base[k] for k in ('times_raw', 'flux_raw', 'times_flat', 'flux_flat', 'bls_period', 'sde_curve',
                                     'n_files', 'n_quarters', 'baseline_days')}

    monkeypatch.setattr(ms, 'build_curves', fake_build)
    monkeypatch.setattr(ms, 'load_extractor', lambda: type('E', (), {})())
    out_dir = tmp_path / 'new' / 'figs'
    rc = ms.main(['--features-file', str(feats), '--signals-file', str(sigs), '--layout', 'wide',
                  '--lang', 'both', '--output-dir', str(out_dir)])
    assert rc == 0
    assert calls['n'] == 1
    assert sorted(os.listdir(out_dir)) == ['showcase_kepler80_wide_en.png', 'showcase_kepler80_wide_ru.png']


def test_single_output_and_bad_signals_file(ms, tmp_path, monkeypatch):
    feats = tmp_path / 'f.csv'
    base = make_data()
    sig = base['signals'].copy()
    sig['target_id'] = '4852528'
    sig.to_csv(feats, index=False)
    bad = tmp_path / 'bad.csv'
    pd.DataFrame({'target_id': ['4852528'], 'x': [1]}).to_csv(bad, index=False)
    monkeypatch.setattr(ms, 'build_curves', lambda *a, **k: {k2: base[k2] for k2 in ('times_raw', 'flux_raw', 'times_flat', 'flux_flat', 'bls_period', 'sde_curve', 'n_files', 'n_quarters', 'baseline_days')})
    monkeypatch.setattr(ms, 'load_extractor', lambda: type('E', (), {})())
    assert ms.main(['--features-file', str(feats), '--signals-file', str(bad), '--output', str(tmp_path / 'x.png')]) == 2
    assert ms.main(['--features-file', str(feats), '--output', str(tmp_path / 'one.png'), '--lang', 'en']) == 0
    assert (tmp_path / 'one.png').is_file()
    with pytest.raises(SystemExit):
        ms.parse_args(['--features-file', 'f', '--lang', 'both', '--output', 'x.png'])
