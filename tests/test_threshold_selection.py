import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

import onyx_report
from synth_demo import SHOWCASE

INFO_KEYS = {'trained', 'criterion', 'target_precision', 'branch', 'thresholds', 'n_oof', 'oof_n_accepted', 'oof_n_correct',
             'oof_coverage', 'oof_precision', 'oof_precision_ci', 'oof_mean', 'oof_std', 'oof_y'}


def _legacy_select(y_tr, oof_mean, oof_std, target_precision):
    best_cov = -1.0
    best_thres = (0.75, 0.25, 0.35)
    best_acc = 0.0
    grid_th = np.linspace(0.50, 0.90, 9)
    grid_tl = np.linspace(0.50, 0.10, 9)
    grid_sm = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0]
    for s_m in grid_sm:
        for t_h in grid_th:
            for t_l in grid_tl:
                c_mask = (oof_mean >= t_h) & (oof_std <= s_m)
                f_mask = (oof_mean <= t_l) & (oof_std <= s_m)
                acc_mask = c_mask | f_mask
                if np.sum(acc_mask) >= 2:
                    pred_acc = np.where(c_mask[acc_mask], 1, 0)
                    acc_val = float(accuracy_score(y_tr[acc_mask], pred_acc))
                    cov_val = float(np.mean(acc_mask))
                    if acc_val >= target_precision:
                        if cov_val > best_cov or (cov_val == best_cov and acc_val > best_acc):
                            best_cov = cov_val
                            best_acc = acc_val
                            best_thres = (float(t_h), float(t_l), float(s_m))
    used_fallback = best_cov < 0
    if best_cov < 0:
        highest_acc = -1.0
        for s_m in grid_sm:
            for t_h in grid_th:
                for t_l in grid_tl:
                    c_mask = (oof_mean >= t_h) & (oof_std <= s_m)
                    f_mask = (oof_mean <= t_l) & (oof_std <= s_m)
                    acc_mask = c_mask | f_mask
                    if np.sum(acc_mask) >= 2:
                        pred_acc = np.where(c_mask[acc_mask], 1, 0)
                        acc_val = float(accuracy_score(y_tr[acc_mask], pred_acc))
                        if acc_val > highest_acc:
                            highest_acc = acc_val
                            best_thres = (float(t_h), float(t_l), float(s_m))
    return best_thres, used_fallback


def _oof_sample(seed, n, noise):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=n)
    p = np.clip(0.5 + (y - 0.5) * rng.uniform(0.0, 1.0, size=n) + rng.normal(0.0, noise, size=n), 0.0, 1.0)
    s = rng.uniform(0.0, 0.6, size=n)
    return y, p, s


def test_point_criterion_reproduces_previous_rule_on_random_data():
    seen = set()
    for seed in range(12):
        y, p, s = _oof_sample(seed, 12 + 7 * (seed % 9), [0.05, 0.2, 0.5][seed % 3])
        for target in ((0.90, 1.01) if seed % 2 == 0 else (0.80, 0.97)):
            legacy_thres, legacy_fallback = _legacy_select(y, p, s, target)
            thres, diag = onyx_report.select_abstention_thresholds(y, p, s, target_precision=target)
            assert thres == legacy_thres
            assert diag['thresholds'] == legacy_thres
            assert diag['branch'] == ('fallback' if legacy_fallback else 'main')
            assert diag['criterion'] == 'point'
            seen.add(diag['branch'])
    assert seen == {'main', 'fallback'}


def test_selection_diagnostics_match_recount():
    y, p, s = _oof_sample(3, 80, 0.2)
    thres, diag = onyx_report.select_abstention_thresholds(y, p, s, target_precision=0.9)
    cand = (p >= thres[0]) & (s <= thres[2])
    acc = cand | ((p <= thres[1]) & (s <= thres[2]))
    assert diag['n_oof'] == 80
    assert diag['oof_n_accepted'] == int(acc.sum())
    assert diag['oof_n_correct'] == int((cand[acc].astype(int) == y[acc]).sum())
    assert diag['oof_coverage'] == pytest.approx(acc.mean())
    assert diag['oof_precision'] == pytest.approx(diag['oof_n_correct'] / diag['oof_n_accepted'])
    assert diag['oof_precision_ci'] == pytest.approx(onyx_report.wilson_interval(diag['oof_n_correct'], diag['oof_n_accepted']))
    if diag['branch'] == 'main':
        assert diag['oof_precision'] >= 0.9


def test_fallback_branch_takes_maximum_accuracy_and_default_branch_keeps_defaults():
    y, p, s = _oof_sample(5, 60, 0.5)
    thres, diag = onyx_report.select_abstention_thresholds(y, p, s, target_precision=1.01)
    assert diag['branch'] == 'fallback'
    best = -1.0
    for s_m in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0]:
        for t_h in np.linspace(0.50, 0.90, 9):
            for t_l in np.linspace(0.50, 0.10, 9):
                cnt = onyx_report.abstention_counts(y, p, s, t_h, t_l, s_m)
                if cnt['n_accepted'] >= 2:
                    best = max(best, cnt['n_correct'] / cnt['n_accepted'])
    assert diag['oof_precision'] == pytest.approx(best)
    thres, diag = onyx_report.select_abstention_thresholds(y, p, np.full(60, 2.0), target_precision=0.9)
    assert diag['branch'] == 'default'
    assert thres == (0.75, 0.25, 0.35)
    assert diag['oof_n_accepted'] == 0
    assert np.isnan(diag['oof_precision'])
    assert 'по умолчанию' in onyx_report.threshold_selection_line(diag)
    assert 'не определена' in onyx_report.threshold_selection_line(diag)


def test_wilson_criterion_requires_lower_bound():
    y = np.array([1] * 10 + [0] * 10)
    p = np.array([0.9] * 10 + [0.1] * 9 + [0.9])
    s = np.full(20, 0.05)
    thres_p, diag_p = onyx_report.select_abstention_thresholds(y, p, s, target_precision=0.9, criterion='point')
    assert diag_p['branch'] == 'main'
    assert diag_p['oof_coverage'] == pytest.approx(1.0)
    assert diag_p['oof_precision'] == pytest.approx(0.95)
    assert onyx_report.wilson_interval(19, 20)[0] < 0.9
    assert onyx_report.wilson_interval(20, 20)[0] < 0.9
    thres_w, diag_w = onyx_report.select_abstention_thresholds(y, p, s, target_precision=0.9, criterion='wilson')
    assert diag_w['branch'] == 'fallback'
    assert diag_w['criterion'] == 'wilson'
    assert 'запасная' in onyx_report.threshold_selection_line(diag_w)
    assert 'wilson' in onyx_report.threshold_selection_line(diag_w)

    rng = np.random.RandomState(11)
    y2 = np.array([1] * 100 + [0] * 100)
    p2 = np.concatenate([rng.uniform(0.85, 0.99, 100), rng.uniform(0.01, 0.15, 100)])
    p2[:12] = 0.55
    y2[:6] = 0
    s2 = np.full(200, 0.05)
    _, d_point = onyx_report.select_abstention_thresholds(y2, p2, s2, target_precision=0.97, criterion='point')
    _, d_wil = onyx_report.select_abstention_thresholds(y2, p2, s2, target_precision=0.97, criterion='wilson')
    assert d_point['branch'] == 'main' and d_wil['branch'] == 'main'
    assert d_point['oof_coverage'] == pytest.approx(1.0)
    assert d_wil['oof_precision_ci'][0] >= 0.97
    assert d_wil['oof_coverage'] < d_point['oof_coverage']
    assert d_point['oof_precision_ci'][0] < 0.97


def test_unknown_criterion_is_rejected(demo):
    y, p, s = _oof_sample(1, 30, 0.2)
    with pytest.raises(ValueError):
        onyx_report.select_abstention_thresholds(y, p, s, criterion='median')
    with pytest.raises(ValueError):
        demo.train_and_eval_pipeline(pd.DataFrame(), pd.DataFrame(), ['f1'], seed=1, threshold_criterion='median')


def _toy_signals(seed, n_stars, per_star, prefix):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_stars):
        cls = i % 2
        for k in range(per_star):
            rows.append({'signal_id': f"{prefix}{i}_{k}", 'target_id': f"{prefix}{i}", 'target_bin': cls,
                         'f1': float(rng.normal(cls * 2.0, 0.7) + 3.0), 'f2': float(rng.normal(0.0, 1.0) + 3.0)})
    return pd.DataFrame(rows)


def test_train_and_eval_fills_info_and_keeps_return_tuple(demo):
    df_tr = _toy_signals(1, 14, 3, 'tr')
    df_te = _toy_signals(2, 6, 2, 'te')
    plain = demo.train_and_eval_pipeline(df_tr, df_te, ['f1', 'f2'], seed=42)
    info = {}
    with_info = demo.train_and_eval_pipeline(df_tr, df_te, ['f1', 'f2'], seed=42, info=info)
    assert len(plain) == 5 and len(with_info) == 5
    assert plain[4] == with_info[4]
    assert np.allclose(plain[0]['p_mean'].values, with_info[0]['p_mean'].values)
    assert list(plain[0]['decision']) == list(with_info[0]['decision'])
    assert set(info) == INFO_KEYS
    assert info['trained'] is True
    assert info['branch'] in ('main', 'fallback')
    assert info['criterion'] == 'point'
    assert info['thresholds'] == with_info[4]
    assert info['n_oof'] == len(df_tr)
    assert len(info['oof_mean']) == len(df_tr) and len(info['oof_std']) == len(df_tr)
    assert list(info['oof_y']) == list(df_tr['target_bin'])
    assert np.all((info['oof_mean'] >= 0.0) & (info['oof_mean'] <= 1.0))
    cnt = onyx_report.abstention_counts(info['oof_y'], info['oof_mean'], info['oof_std'], *info['thresholds'])
    assert cnt['n_accepted'] == info['oof_n_accepted']
    assert cnt['n_correct'] == info['oof_n_correct']
    assert info['oof_coverage'] == pytest.approx(cnt['n_accepted'] / len(df_tr))
    assert info['oof_precision'] == pytest.approx(cnt['n_correct'] / cnt['n_accepted'])
    legacy_thres, legacy_fallback = _legacy_select(info['oof_y'], info['oof_mean'], info['oof_std'], 0.90)
    assert legacy_thres == info['thresholds']
    assert (info['branch'] == 'fallback') == legacy_fallback


def test_train_and_eval_info_when_not_trained(demo):
    df_tr = _toy_signals(1, 14, 3, 'tr')
    df_tr['target_bin'] = 1
    df_te = _toy_signals(2, 6, 2, 'te')
    info = {}
    res = demo.train_and_eval_pipeline(df_tr, df_te, ['f1', 'f2'], seed=42, info=info, threshold_criterion='wilson')
    assert res[1] is None
    assert set(info) == INFO_KEYS
    assert info['trained'] is False
    assert info['branch'] == 'not_trained'
    assert info['criterion'] == 'wilson'
    assert info['n_oof'] == 0 and len(info['oof_mean']) == 0
    assert 'не выполнялся' in onyx_report.threshold_selection_line(info)


def test_wilson_criterion_reaches_pipeline_thresholds(demo):
    df_tr = _toy_signals(1, 14, 3, 'tr')
    df_te = _toy_signals(2, 6, 2, 'te')
    info = {}
    demo.train_and_eval_pipeline(df_tr, df_te, ['f1', 'f2'], seed=42, info=info, threshold_criterion='wilson', target_precision=0.80)
    expect, diag = onyx_report.select_abstention_thresholds(info['oof_y'], info['oof_mean'], info['oof_std'], target_precision=0.80, criterion='wilson')
    assert info['thresholds'] == expect
    assert info['branch'] == diag['branch']
    if info['branch'] == 'main':
        assert info['oof_precision_ci'][0] >= 0.80


def test_evaluate_ablation_forwards_criterion(demo, monkeypatch):
    seen = []

    def fake(df_train, df_test, feats, seed, full_features_pool=None, target_precision=0.90, info=None, threshold_criterion='point'):
        seen.append((target_precision, threshold_criterion))
        return df_test, None, None, None, (0.75, 0.25, 0.35)

    monkeypatch.setattr(demo, 'train_and_eval_pipeline', fake)
    empty = pd.DataFrame({'target_id': [], 'match_status': []})
    assert demo.evaluate_ablation(['f1'], empty, empty, set(), SHOWCASE, 1, empty, 0.85, n_boot=5, threshold_criterion='wilson') is None
    assert demo.evaluate_ablation(['f1'], empty, empty, set(), SHOWCASE, 1, empty, 0.85, n_boot=5) is None
    assert seen == [(0.85, 'wilson'), (0.85, 'point')]


def test_coverage_precision_table_counts_and_intervals():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    p = np.array([0.95, 0.85, 0.55, 0.05, 0.15, 0.45, 0.30, 0.75])
    s = np.array([0.05, 0.40, 0.05, 0.05, 0.05, 0.40, 0.05, 0.05])
    tab = onyx_report.coverage_precision_table([('A', y, p, s), ('B', y[:4], p[:4], s[:4])], 0.30)
    assert len(tab) == 2 * 2 * 5
    assert list(tab['Выборка'].unique()) == ['A', 'B']
    assert list(tab['s_max'].unique()) == ['1.00', '0.30']
    a_full = tab[(tab['Выборка'] == 'A') & (tab['s_max'] == '1.00')].reset_index(drop=True)
    assert list(a_full['t_high / t_low']) == ['0.50 / 0.50', '0.60 / 0.40', '0.70 / 0.30', '0.80 / 0.20', '0.90 / 0.10']
    assert list(a_full['Принято']) == [8, 6, 6, 4, 2]
    assert list(a_full['Охват']) == ['100.0%', '75.0%', '75.0%', '50.0%', '25.0%']
    assert a_full['Точность на принятых'].iloc[0] == '0.750 (6/8)'
    assert a_full['Точность на принятых'].iloc[1] == '0.667 (4/6)'
    assert a_full['Точность на принятых'].iloc[3] == '1.000 (4/4)'
    assert a_full['95% ДИ Уилсона'].iloc[0] == onyx_report.ci_text(onyx_report.wilson_interval(6, 8))
    a_sel = tab[(tab['Выборка'] == 'A') & (tab['s_max'] == '0.30')].reset_index(drop=True)
    assert list(a_sel['Принято']) == [6, 5, 5, 3, 2]
    assert a_sel['Точность на принятых'].iloc[0] == '0.667 (4/6)'
    one = onyx_report.coverage_precision_table([('A', y, p, s)], 1.0)
    assert len(one) == 5 and list(one['s_max'].unique()) == ['1.00']
    none = onyx_report.coverage_precision_table([('A', y, p, np.full(8, 0.9))], 0.30)
    row = none[(none['s_max'] == '0.30')].iloc[0]
    assert row['Принято'] == 0 and row['Точность на принятых'] == '-' and row['95% ДИ Уилсона'] == '-'
    empty = onyx_report.coverage_precision_table([('E', np.zeros(0), np.zeros(0), np.zeros(0))], 0.30)
    assert set(empty['Охват']) == {'-'}


def _section(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


def test_report_has_threshold_selection_subsection(fixes_default_report):
    text = fixes_default_report['text']
    sec3 = text[text.index('3. Раздел А'):text.index('4. Раздел Б')]
    assert sec3.index('Калибровка оценки модели') < sec3.index('Подбор порогов отказа от решения') < sec3.index('Абляции признаков')
    sub = _section(sec3, 'Подбор порогов отказа от решения', 'Абляции признаков')
    ws = fixes_default_report['ws']
    feats = ws['features']
    n_train = int(((feats['target_id'].isin(ws['groups']['tr_m'])) & (feats['match_status'] == 'MATCHED')).sum())
    assert f"OOF-сигналов: {n_train};" in sub
    for needle in ['OOF-охват:', 'OOF-точность на принятых:', 'сработавшая ветка:', 'критерий: point', '95% ДИ Уилсона',
                   '| OOF обучающей половины | 1.00 | 0.50 / 0.50 |', '| Контрольная половина (без Kepler-80) | 1.00 | 0.90 / 0.10 |',
                   '--threshold-criterion point']:
        assert needle in sub, needle
    table_rows = [ln for ln in sub.splitlines() if ln.startswith('| OOF обучающей половины |') or ln.startswith('| Контрольная половина (без Kepler-80) |')]
    assert len(table_rows) in (10, 20)
    first = [c.strip() for c in table_rows[0].strip('|').split('|')]
    assert first[3] == str(n_train) and first[4] == str(n_train) and first[5] == '100.0%'
    assert fixes_default_report['calls'] and set(fixes_default_report['calls']) == {'point'}


def test_report_wilson_criterion_fallback_is_stated(fixes_wilson_report):
    text = fixes_wilson_report['text']
    ws = fixes_wilson_report['ws']
    feats = ws['features']
    n_train = int(((feats['target_id'].isin(ws['groups']['tr_m'])) & (feats['match_status'] == 'MATCHED')).sum())
    assert onyx_report.wilson_interval(n_train, n_train)[0] < 0.90
    sub = _section(text, 'Подбор порогов отказа от решения', 'Абляции признаков')
    assert 'критерий: wilson' in sub
    assert 'сработавшая ветка: запасная' in sub
    assert 'Запасная ветка:' in sub and '--threshold-criterion wilson' in sub and 'не подтверждена' in sub
    assert fixes_wilson_report['calls'] and set(fixes_wilson_report['calls']) == {'wilson'}
    assert len(fixes_wilson_report['calls']) >= 4


def test_cli_rejects_unknown_criterion(demo, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--report-only', '--threshold-criterion', 'median'])
    with pytest.raises(SystemExit):
        demo.main()
