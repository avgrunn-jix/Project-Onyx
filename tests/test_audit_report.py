import os
import sys

import numpy as np
import pandas as pd
import pytest

import onyx_report
from synth_demo import SHOWCASE, build_workspace


def test_wilson_interval_known_values():
    lo, hi = onyx_report.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = onyx_report.wilson_interval(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-3)
    assert hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = onyx_report.wilson_interval(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert lo == pytest.approx(0.7225, abs=1e-3)
    assert onyx_report.wilson_interval(0, 0) == (0.0, 0.0)
    for k in range(0, 21):
        lo, hi = onyx_report.wilson_interval(k, 20)
        assert 0.0 <= lo <= k / 20 <= hi <= 1.0


def _recovery_inputs():
    koi = pd.DataFrame([
        {'target_id': '1', 'label': 'CONFIRMED', 'catalog_period': 2.0},
        {'target_id': '1', 'label': 'CONFIRMED', 'catalog_period': 5.0},
        {'target_id': '2', 'label': 'CANDIDATE', 'catalog_period': 12.0},
        {'target_id': '3', 'label': 'FALSE POSITIVE', 'catalog_period': 40.0},
        {'target_id': '3', 'label': 'FALSE POSITIVE', 'catalog_period': 150.0},
        {'target_id': '4', 'label': 'CONFIRMED', 'catalog_period': 0.3},
        {'target_id': '5', 'label': 'CANDIDATE', 'catalog_period': 8.0},
    ])
    sig = pd.DataFrame([
        {'target_id': '1', 'bls_period_d': 2.005, 'decision': 'CANDIDATE'},
        {'target_id': '1', 'bls_period_d': 9.0, 'decision': 'REVIEW'},
        {'target_id': '2', 'bls_period_d': 24.0, 'decision': 'CANDIDATE'},
        {'target_id': '3', 'bls_period_d': 20.1, 'decision': 'FALSE_POSITIVE'},
        {'target_id': '5', 'bls_period_d': 8.0, 'decision': 'REVIEW'},
    ])
    return koi, sig


def test_koi_recovery_statuses_and_range():
    koi, sig = _recovery_inputs()
    rec = onyx_report.build_koi_recovery(koi, sig)
    st = {(r.target_id, r.catalog_period): r.status for r in rec.itertuples()}
    assert st[('1', 2.0)] == 'точно'
    assert st[('1', 5.0)] == 'не найден'
    assert st[('2', 12.0)] == 'гармоника'
    assert st[('3', 40.0)] == 'гармоника'
    assert st[('3', 150.0)] == 'вне диапазона'
    assert st[('4', 0.3)] == 'вне диапазона'
    assert st[('5', 8.0)] == 'точно'
    assert int(rec['in_range'].sum()) == 5


def test_koi_recovery_summary_counts_and_end_to_end():
    koi, sig = _recovery_inputs()
    summ = onyx_report.summarize_koi_recovery(onyx_report.build_koi_recovery(koi, sig))
    assert summ['n_total'] == 7
    assert summ['n_in_range'] == 5
    assert summ['n_out_of_range'] == 2
    assert summ['n_positive_in_range'] == 4
    assert summ['n_positive_found_exact'] == 2
    assert summ['n_end_to_end'] == 1
    assert summ['end_to_end_rate'] == pytest.approx(0.25)
    assert summ['end_to_end_ci'] == pytest.approx(onyx_report.wilson_interval(1, 4))
    by_disp = summ['by_disposition'].set_index('Группа')
    assert by_disp.loc['CONFIRMED', 'KOI в диапазоне'] == 2
    assert by_disp.loc['CANDIDATE', 'KOI в диапазоне'] == 2
    assert by_disp.loc['FALSE_POSITIVE', 'KOI в диапазоне'] == 1
    assert by_disp.loc['Все', 'KOI в диапазоне'] == 5
    assert by_disp.loc['CONFIRMED', 'Точно'].startswith('1/2 = 50.0%')
    by_bin = summ['by_period_bin']
    assert int(by_bin['KOI в диапазоне'].sum()) == 5
    assert list(by_bin['Группа']) == ['P 0.5–3 д', 'P 3–10 д', 'P 10–30 д', 'P 30–100 д']


def test_koi_recovery_counts_third_and_triple_as_harmonic():
    koi = pd.DataFrame([{'target_id': '7', 'label': 'CONFIRMED', 'catalog_period': 9.0},
                        {'target_id': '8', 'label': 'CONFIRMED', 'catalog_period': 3.0}])
    sig = pd.DataFrame([{'target_id': '7', 'bls_period_d': 3.01, 'decision': 'REVIEW'},
                        {'target_id': '8', 'bls_period_d': 9.02, 'decision': 'REVIEW'}])
    rec = onyx_report.build_koi_recovery(koi, sig)
    assert list(rec['status']) == ['гармоника', 'гармоника']


def test_koi_without_any_signal_is_not_found():
    koi = pd.DataFrame([{'target_id': '9', 'label': 'CONFIRMED', 'catalog_period': 4.0}])
    sig = pd.DataFrame(columns=['target_id', 'bls_period_d', 'decision'])
    rec = onyx_report.build_koi_recovery(koi, sig)
    assert rec['status'].iloc[0] == 'не найден'


def test_brier_and_reliability_table():
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 0.5, 0.5])
    assert onyx_report.brier_score(y, p) == pytest.approx(0.125)
    y2 = np.array([0, 0, 1, 1, 1, 1])
    p2 = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    tab = onyx_report.reliability_table(y2, p2, 5)
    assert len(tab) == 5
    assert list(tab['Число сигналов']) == [1, 1, 1, 1, 2]
    assert tab['Доля класса 1'].iloc[0] == '0.000'
    assert tab['Доля класса 1'].iloc[4] == '1.000'
    assert tab['Средняя оценка'].iloc[4] == '0.950'
    empty = onyx_report.reliability_table(np.array([1.0]), np.array([0.05]), 5)
    assert empty['Число сигналов'].tolist() == [1, 0, 0, 0, 0]
    assert empty['Средняя оценка'].iloc[1] == '-'


def test_oriented_auc_uses_training_direction():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([9.0, 8.0, 7.0, 3.0, 2.0, 1.0])
    assert onyx_report.oriented_auc(y, score, '>=') == pytest.approx(0.0)
    assert onyx_report.oriented_auc(y, score, '<') == pytest.approx(1.0)
    assert onyx_report.oriented_auc(np.array([1, 1]), np.array([1.0, 2.0]), '<') == 0.5


def test_ablation_feature_sets_remove_expected_columns():
    feats = onyx_report.STAR_FEATURE_COLS + ['transit_count', 'gap_fraction', 'bls_depth_snr', 'odd_even_sigma']
    sets = dict(onyx_report.ablation_feature_sets(feats))
    names = list(sets)
    no_star = sets[names[0]]
    no_geom = sets[names[1]]
    assert set(no_star) == {'transit_count', 'gap_fraction', 'bls_depth_snr', 'odd_even_sigma'}
    assert set(no_geom) == set(onyx_report.STAR_FEATURE_COLS + ['bls_depth_snr', 'odd_even_sigma'])
    assert 'star_teff' in names[0] and names[1] == 'Без transit_count и gap_fraction'


def _binary_features(ws):
    df = ws['features'].copy()
    df['target_bin'] = df['label'].apply(lambda x: 1 if str(x).upper() in ['CONFIRMED', 'CANDIDATE'] else 0)
    return df


def test_bootstrap_star_metrics_has_95_intervals_and_new_keys(demo):
    rng = np.random.RandomState(0)
    rows = []
    for s in range(30):
        cls = s % 2
        for k in range(2):
            p = float(np.clip(rng.normal(0.3 + 0.4 * cls, 0.2), 0, 1))
            dec = 'CANDIDATE' if p >= 0.6 else ('FALSE_POSITIVE' if p <= 0.4 else 'REVIEW')
            rows.append({'target_id': f's{s}', 'target_bin': cls, 'p_mean': p, 'decision': dec})
    df = pd.DataFrame(rows)
    out = demo.bootstrap_star_metrics(df, n_boot=200, seed=1)
    assert set(out) == {'acc_ci', 'auc_ci', 'cov_ci', 'prec_ci', 'rec_ci', 'acc_accepted_ci'}
    for lo, hi in out.values():
        assert 0.0 <= lo <= hi <= 1.0
    assert out == demo.bootstrap_star_metrics(df, n_boot=200, seed=1)
    wide = demo.bootstrap_star_metrics(df, n_boot=200, seed=1)['auc_ci']
    assert wide[1] > wide[0]


def test_out_of_fold_models_match_final_ensemble(demo, monkeypatch, tmp_path):
    ws = build_workspace(tmp_path)
    df = _binary_features(ws)
    tr = df[df['target_id'].isin(ws['groups']['tr_m']) & (df['match_status'] == 'MATCHED')]
    ctrl = df[df['target_id'].isin(ws['groups']['ctrl_m'])]
    feats = [c for c in onyx_features_cols(df)]
    created = []
    orig = demo.XGBClassifier

    def Spy(**k):
        created.append((k.get('n_estimators'), k.get('max_depth')))
        return orig(**k)

    monkeypatch.setattr(demo, 'XGBClassifier', Spy)
    _, model, _, _, _ = demo.train_and_eval_pipeline(tr, ctrl, feats, seed=5)
    assert model is not None
    n_stars = tr['target_id'].nunique()
    n_models = min(20, max(5, len(tr) * 2))
    n_splits = min(5, n_stars)
    assert created.count((60, 4)) == n_models * (n_splits + 1)
    assert created.count((100, 4)) == 1
    assert len(created) == n_models * (n_splits + 1) + 1


def onyx_features_cols(df):
    import onyx_features
    return onyx_features.compute_feature_columns(df, exclude_coverage=True)


def test_evaluate_ablation_returns_auc_with_interval(demo, tmp_path):
    ws = build_workspace(tmp_path)
    df = _binary_features(ws)
    feats = onyx_features_cols(df)
    tr = df[df['target_id'].isin(ws['groups']['tr_m']) & (df['match_status'] == 'MATCHED')]
    ctrl_all = df[df['target_id'].isin(ws['groups']['ctrl_m'] + ws['groups']['ctrl_u'])]
    for name, sub in onyx_report.ablation_feature_sets(feats):
        res = demo.evaluate_ablation(sub, tr, ctrl_all, set(ws['groups']['ctrl_m']), SHOWCASE, 7, df, 0.9, n_boot=50)
        assert res is not None
        assert res['n_features'] == len(sub)
        assert 0.0 <= res['auc'] <= 1.0
        lo, hi = res['auc_ci']
        assert lo <= hi
        assert 'transit_count' not in sub or 'gap_fraction' in sub


def test_summary_contains_new_report_sections(demo, monkeypatch, tmp_path):
    ws = build_workspace(tmp_path, with_version=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--quick', '--report-only', '--output-dir', ws['out_dir'], '--showcase-star', SHOWCASE, '--cache-dir', str(tmp_path / 'c')])
    demo.main()
    text = open(os.path.join(ws['out_dir'], 'summary_quick_report.md'), encoding='utf-8').read()
    for needle in ['Восстановление KOI', 'По диспозиции', 'По корзинам периода', 'вне диапазона поиска', 'Сквозная полнота',
                   'интервалом Уилсона', '95% ДИ', 'Оценка Брайера', 'Калибровка', 'Корзина p_mean', 'Абляции признаков',
                   'Без звёздных признаков', 'Без transit_count и gap_fraction', 'Примеры верных решений', 'Примеры ошибочных решений',
                   'Оценка модели (p_mean)', 'модель не обучалась на шумовых сигналах']:
        assert needle in text, needle
    assert 'Вероятность' not in text
    assert 'ложных тревог' not in text.lower()
    assert 'False Alarm' not in text
    assert os.path.exists(os.path.join(ws['out_dir'], 'koi_recovery_quick_report.csv'))
    rec = pd.read_csv(os.path.join(ws['out_dir'], 'koi_recovery_quick_report.csv'))
    assert set(rec['status']) <= {'точно', 'гармоника', 'не найден', 'вне диапазона'}
