import numpy as np
import pandas as pd
import pytest

import onyx_report


def test_mean_abs_shap_table_ranks_by_mean_absolute_value():
    sv = np.array([[1.0, -3.0, 0.0, 0.5], [-1.0, 3.0, 0.0, 0.5], [2.0, -3.0, 0.0, -0.5]])
    tab, n = onyx_report.mean_abs_shap_table(sv, ['a', 'b', 'c', 'd'], top=3)
    assert n == 3
    assert list(tab['Признак']) == ['b', 'a', 'd']
    assert list(tab['Место']) == [1, 2, 3]
    assert tab['Средний |SHAP|'].iloc[0] == '3.0000'
    assert tab['mean_abs_shap'].iloc[1] == pytest.approx(4.0 / 3.0)
    assert tab['Доля от суммы по всем признакам'].iloc[0] == f"{3.0 / (3.0 + 4.0 / 3.0 + 0.5):.1%}"
    assert onyx_report.top_feature_label((tab, n)) == 'b (3.000)'


def test_mean_abs_shap_table_top_limit_and_three_dim_input():
    sv = np.random.RandomState(0).normal(size=(20, 15))
    tab, _ = onyx_report.mean_abs_shap_table(sv, [f"f{i}" for i in range(15)])
    assert len(tab) == 10
    sv3 = np.stack([-sv, sv], axis=-1)
    tab3, _ = onyx_report.mean_abs_shap_table(sv3, [f"f{i}" for i in range(15)])
    assert list(tab3['Признак']) == list(tab['Признак'])


def test_mean_abs_shap_table_without_values():
    cols = ['a', 'b']
    assert onyx_report.mean_abs_shap_table(None, cols) is None
    assert onyx_report.mean_abs_shap_table(np.zeros((0, 2)), cols) is None
    assert onyx_report.mean_abs_shap_table(np.zeros((3, 5)), cols) is None
    assert onyx_report.mean_abs_shap_table(np.full((3, 2), np.nan), cols) is None
    assert onyx_report.mean_abs_shap_table(np.ones((3, 2)), []) is None
    assert onyx_report.top_feature_label(None) == '-'


def test_shap_importance_none_cases_and_failure(demo):
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})

    class Scaler:
        def transform(self, x):
            return x

    class Broken:
        def shap_values(self, x):
            raise RuntimeError('no shap')

    class Good:
        def shap_values(self, x):
            return np.array([[1.0, -2.0]] * len(x))

    assert demo.shap_importance(None, Scaler(), df, ['a', 'b']) is None
    assert demo.shap_importance(Good(), None, df, ['a', 'b']) is None
    assert demo.shap_importance(Good(), Scaler(), df.iloc[0:0], ['a', 'b']) is None
    assert demo.shap_importance(Broken(), Scaler(), df, ['a', 'b']) is None
    tab, n = demo.shap_importance(Good(), Scaler(), df, ['a', 'b'])
    assert n == 2 and list(tab['Признак']) == ['b', 'a']


def _toy(seed, n_stars, prefix):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_stars):
        cls = i % 2
        for k in range(3):
            rows.append({'signal_id': f"{prefix}{i}_{k}", 'target_id': f"{prefix}{i}", 'target_bin': cls, 'match_status': 'MATCHED',
                         'f1': float(rng.normal(cls * 3.0, 0.5) + 3.0), 'f2': float(rng.normal(0.0, 1.0) + 3.0)})
    return pd.DataFrame(rows)


def test_evaluate_ablation_returns_main_feature_by_mean_shap(demo):
    tr = _toy(1, 14, 'tr')
    ctrl = _toy(2, 8, 'te')
    res = demo.evaluate_ablation(['f1', 'f2'], tr, ctrl, set(ctrl['target_id']), 'none', 5, tr, 0.9, n_boot=20)
    assert res['top_shap'].startswith('f1 (')
    assert res['top_shap'].endswith(')')
    assert float(res['top_shap'][4:-1]) > 0.0


def test_evaluate_ablation_without_shap_gives_dash(demo, monkeypatch):
    tr = _toy(1, 14, 'tr')
    ctrl = _toy(2, 8, 'te')
    monkeypatch.setattr(demo, 'shap_importance', lambda *a, **k: None)
    res = demo.evaluate_ablation(['f1', 'f2'], tr, ctrl, set(ctrl['target_id']), 'none', 5, tr, 0.9, n_boot=20)
    assert res['top_shap'] == '-'


def test_report_has_shap_table_and_new_ablation_column(fixes_default_report):
    text = fixes_default_report['text']
    sec6 = text[text.index('6. Примеры классификации'):text.index('7. Проверка каталога')]
    assert 'Важность признаков основной модели (среднее |SHAP| по выборке)' in sec6
    assert 'без Kepler-80' in sec6 and 'не по одному сигналу' in sec6
    rows = [ln for ln in sec6.splitlines() if ln.startswith('| ') and not ln.startswith('| ---') and not ln.startswith('| Место')]
    assert len(rows) == 10 and rows[0].startswith('| 1 |')
    vals = [float(r.split('|')[3]) for r in rows]
    assert vals == sorted(vals, reverse=True)
    abl = text[text.index('Абляции признаков'):text.index('Двунаправленная перекрёстная проверка')]
    assert 'Главный признак по среднему |SHAP|' in abl
    assert 'Топ-1 признак SHAP' not in abl
    assert 'по одному сигналу' in abl
    body = [ln for ln in abl.splitlines() if ln.startswith('| ') and not ln.startswith('| ---') and not ln.startswith('| Конфигурация')]
    for ln in body:
        last = ln.strip().strip('|').split('|')[-1].strip()
        assert last.endswith(')') and ' (' in last


def test_report_without_shap_values_states_it(demo, tmp_path):
    from synth_demo import build_workspace
    from conftest import run_demo_report
    root = tmp_path
    ws = build_workspace(root)
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(demo, 'shap_importance', lambda *a, **k: None)
        text, _ = run_demo_report(demo, root, ws)
    finally:
        mp.undo()
    sec6 = text[text.index('6. Примеры классификации'):text.index('7. Проверка каталога')]
    assert 'Таблица не построена' in sec6
    abl = text[text.index('Абляции признаков'):text.index('Двунаправленная перекрёстная проверка')]
    body = [ln for ln in abl.splitlines() if ln.startswith('| ') and not ln.startswith('| ---') and not ln.startswith('| Конфигурация')]
    assert body and all(ln.strip().strip('|').split('|')[-1].strip() == '-' for ln in body)
