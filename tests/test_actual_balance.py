import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

import onyx_report


def _diag_frame():
    rows = []
    for i in range(10):
        rows.append({'target_id': f"k{i}", 'n_quarters': 17 if i < 8 else 10, 'baseline_d': 1470.0 if i < 8 else 900.0})
    for i in range(8):
        rows.append({'target_id': f"c{i}", 'n_quarters': 17 if i < 4 else 12 + i, 'baseline_d': 1470.0 if i < 4 else 1000.0 + 10 * i})
    rows.append({'target_id': 'other', 'n_quarters': 1, 'baseline_d': 30.0})
    return pd.DataFrame(rows)


def test_actual_balance_tables_statistics_and_ks():
    diag = _diag_frame()
    koi = [f"k{i}" for i in range(10)]
    ctrl = [f"c{i}" for i in range(8)]
    res = onyx_report.actual_balance_tables(diag, [('KOI', koi), ('Control', ctrl)])
    stats = res['stats'].set_index('Группа')
    share_col = 'Доля звёзд с числом кварталов < 17'
    assert stats.loc['KOI', 'Обработано звёзд'] == 10 and stats.loc['KOI', 'Из них с диагностикой'] == 10
    assert stats.loc['KOI', 'Кварталов (среднее)'] == '15.60'
    assert stats.loc['KOI', 'Кварталов (медиана)'] == '17.0'
    assert stats.loc['KOI', share_col].startswith('2/10 = 20.0%')
    assert stats.loc['KOI', 'Базлайн, сут (среднее)'] == '1356.0'
    assert stats.loc['KOI', 'Базлайн, сут (медиана)'] == '1470.0'
    q_ctrl = np.array([17, 17, 17, 17, 16, 17, 18, 19], dtype=float)
    assert stats.loc['Control', 'Кварталов (среднее)'] == f"{q_ctrl.mean():.2f}"
    assert stats.loc['Control', share_col].startswith('1/8 = 12.5%')
    q_koi = np.array([17] * 8 + [10] * 2, dtype=float)
    ks_q = ks_2samp(q_koi, q_ctrl)
    b_koi = diag[diag['target_id'].isin(koi)]['baseline_d'].values
    b_ctrl = diag[diag['target_id'].isin(ctrl)]['baseline_d'].values
    ks_b = ks_2samp(b_koi, b_ctrl)
    assert res['ks_values']['n_quarters'] == pytest.approx((ks_q.statistic, ks_q.pvalue))
    assert res['ks_values']['baseline_d'] == pytest.approx((ks_b.statistic, ks_b.pvalue))
    ks = res['ks'].set_index('Параметр')
    assert ks.loc['Число уникальных кварталов', 'Статистика KS (D)'] == f"{ks_q.statistic:.3f}"
    assert ks.loc['Базлайн (сут)', 'p-значение'] == f"{ks_b.pvalue:.4f}"
    assert ks.loc['Число уникальных кварталов', 'KOI (min / Q1 / med / Q3 / max)'] == '10 / 17 / 17 / 17 / 17'
    assert res['notes'] == []
    part = onyx_report.actual_balance_tables(diag, [('KOI', koi + ['missing1', 'missing2']), ('Control', ctrl)])
    row = part['stats'].set_index('Группа').loc['KOI']
    assert row['Обработано звёзд'] == 12 and row['Из них с диагностикой'] == 10


def test_actual_balance_tables_empty_group():
    diag = _diag_frame()
    res = onyx_report.actual_balance_tables(diag, [('KOI', [f"k{i}" for i in range(10)]), ('Control', [])])
    stats = res['stats'].set_index('Группа')
    assert stats.loc['Control', 'Обработано звёзд'] == 0
    assert stats.loc['Control', 'Кварталов (среднее)'] == '-'
    assert stats.loc['Control', 'Доля звёзд с числом кварталов < 17'] == '-'
    assert set(res['ks']['Статистика KS (D)']) == {'-'} and set(res['ks']['p-значение']) == {'-'}
    assert res['ks_values'] == {}
    assert any('Control' in n and 'не посчитаны' in n for n in res['notes'])
    both_empty = onyx_report.actual_balance_tables(pd.DataFrame(columns=['target_id']), [('KOI', ['1']), ('Control', ['2'])])
    assert len(both_empty['stats']) == 2 and len(both_empty['ks']) == 0
    assert len(both_empty['notes']) == 4
    assert onyx_report.actual_balance_tables(None, [('KOI', []), ('Control', [])])['ks_values'] == {}


def test_actual_balance_tables_without_quarters_column():
    diag = _diag_frame().drop(columns=['n_quarters'])
    res = onyx_report.actual_balance_tables(diag, [('KOI', [f"k{i}" for i in range(10)]), ('Control', [f"c{i}" for i in range(8)])])
    stats = res['stats'].set_index('Группа')
    assert stats.loc['KOI', 'Кварталов (среднее)'] == '-'
    assert stats.loc['KOI', 'Доля звёзд с числом кварталов < 17'] == '-'
    assert stats.loc['KOI', 'Базлайн, сут (среднее)'] == '1356.0'
    assert list(res['ks']['Параметр']) == ['Базлайн (сут)']
    assert 'baseline_d' in res['ks_values'] and 'n_quarters' not in res['ks_values']
    assert any('n_quarters' in n for n in res['notes'])
    bad = _diag_frame()
    bad['n_quarters'] = bad['n_quarters'].astype(object)
    bad.loc[0, 'n_quarters'] = 'нет'
    res_bad = onyx_report.actual_balance_tables(bad, [('KOI', [f"k{i}" for i in range(10)]), ('Control', [f"c{i}" for i in range(8)])])
    assert res_bad['stats'].set_index('Группа').loc['KOI', 'Доля звёзд с числом кварталов < 17'].startswith('2/9 = ')


def test_report_actual_balance_section(fixes_default_report):
    text = fixes_default_report['text']
    sec2 = text[text.index('2. Состав выборки'):text.index('2.1. Диагностика обнаружения')]
    assert 'по справочному кэшу (план)' in sec2
    assert 'Баланс выборки по фактически скачанным данным' in sec2
    ws = fixes_default_report['ws']
    n_m = len(ws['groups']['tr_m']) + len(ws['groups']['ctrl_m'])
    n_u = len(ws['groups']['tr_u']) + len(ws['groups']['ctrl_u'])
    assert f"| Отмеченные (KOI) | {n_m} | {n_m} | 17.00 | 17.0 | 0/{n_m} = 0.0%" in sec2
    assert f"| Неотмеченные (Control) | {n_u} | {n_u} | {(17 * (n_u - 5) + 12 * 5) / n_u:.2f} | 17.0 | 5/{n_u} = {5 / n_u:.1%}" in sec2
    assert 'KS-тест по фактически скачанным данным' in sec2
    assert '| Число уникальных кварталов | 17 / 17 / 17 / 17 / 17 | 12 / 17 / 17 / 17 / 17 |' in sec2
    assert '| Базлайн (сут) |' in sec2


def test_report_states_missing_quarters_column(fixes_wilson_report):
    text = fixes_wilson_report['text']
    sec2 = text[text.index('2. Состав выборки'):text.index('2.1. Диагностика обнаружения')]
    assert 'нет колонки числа уникальных кварталов' in sec2
    assert '| Базлайн (сут) |' in sec2 and '| Число уникальных кварталов |' not in sec2
