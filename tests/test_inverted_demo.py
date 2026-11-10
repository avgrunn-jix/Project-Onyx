import os
import sys

import pandas as pd

import onyx_report
from synth_demo import SHOWCASE, add_inverted_control, build_workspace


def _run_report(demo, monkeypatch, tmp_path, ws):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--quick', '--report-only', '--output-dir', ws['out_dir'], '--showcase-star', SHOWCASE, '--cache-dir', str(tmp_path / 'c')])
    demo.main()
    with open(os.path.join(ws['out_dir'], 'summary_quick_report.md'), encoding='utf-8') as fh:
        return fh.read()


def test_inverted_star_rates_with_wilson():
    tab = onyx_report.inverted_star_rates([('A', ['1', '2', '3', '4']), ('B', ['5', '6'])], {'1', '2', '3', '5', '6'}, {'1': 2, '3': 1, '6': 0})
    a = tab.iloc[0]
    assert a['Звёзд с контролем'] == 3
    assert a['Звёзд с INVERTED-сигналом'].startswith('2/3 = 66.7%')
    assert a['INVERTED-сигналов всего'] == 3
    b = tab.iloc[1]
    assert b['Звёзд с INVERTED-сигналом'].startswith('0/2 = 0.0%')


def test_inverted_control_stats_uses_diagnostics_flag(demo):
    diag = {'1': pd.Series({'status': 'OK', 'inverted_control': 1}), '2': pd.Series({'status': 'OK', 'inverted_control': 0}),
            '3': pd.Series({'status': 'FAILED: x', 'inverted_control': 1})}
    df_inv = pd.DataFrame({'target_id': ['1', '1', '2'], 'match_status': ['INVERTED'] * 3})
    counts, controlled = demo.inverted_control_stats(df_inv, diag, {'1', '2'})
    assert counts == {'1': 2, '2': 1}
    assert controlled == {'1'}
    counts, controlled = demo.inverted_control_stats(df_inv, {}, {'1', '2', '9'})
    assert controlled == {'1', '2', '9'}
    counts, controlled = demo.inverted_control_stats(None, {}, {'1'})
    assert counts == {} and controlled == set()


def test_demo_report_inverted_section_and_diag_column(demo, monkeypatch, tmp_path):
    ws = add_inverted_control(build_workspace(tmp_path))
    text = _run_report(demo, monkeypatch, tmp_path, ws)
    assert '2.4. Контроль на перевёрнутой кривой' in text
    n_m = len(ws['groups']['tr_m']) + len(ws['groups']['ctrl_m'])
    k_m = sum(1 for s in ws['groups']['tr_m'] + ws['groups']['ctrl_m'] if ws['inverted_counts'][s] > 0)
    assert f"| Звёзды KOI (отмеченные) | {n_m} | {k_m}/{n_m} = " in text
    diag = pd.read_csv(os.path.join(ws['out_dir'], 'diagnostics_stars_quick_report.csv'), dtype={'target_id': str})
    assert 'INVERTED-сигналов' in diag.columns
    row = diag[diag['target_id'] == ws['groups']['tr_u'][0]].iloc[0]
    assert str(row['INVERTED-сигналов']) == str(ws['inverted_counts'][ws['groups']['tr_u'][0]])
    ctrl = pd.read_csv(os.path.join(ws['out_dir'], 'control_signals_quick_report.csv'), dtype={'target_id': str})
    assert not ctrl['signal_id'].astype(str).str.contains('_inv').any()
    assert 'INVERTED' not in set(ctrl['catalog_label'].astype(str))


def test_demo_report_without_inverted_file_keeps_old_behaviour(demo, monkeypatch, tmp_path):
    ws = build_workspace(tmp_path)
    text = _run_report(demo, monkeypatch, tmp_path, ws)
    assert 'Контроль не запускался' in text
    diag = pd.read_csv(os.path.join(ws['out_dir'], 'diagnostics_stars_quick_report.csv'), dtype={'target_id': str})
    assert set(diag['INVERTED-сигналов'].astype(str)) == {'-'}


def test_demo_passes_inverted_flag_to_extractor(demo, monkeypatch, tmp_path):
    import main as main_mod
    seen = {}

    def fake_run(cmd, *a, **k):
        seen['cmd'] = cmd
        return type('R', (), {'returncode': 0})()

    monkeypatch.setattr(main_mod.subprocess, 'run', fake_run)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--mode', 'demo', '--inverted-control', '--min-free-gb', '0', '--cache-dir', str(tmp_path / 'c')])
    main_mod.main()
    assert '--inverted-control' in seen['cmd']
    seen.clear()
    monkeypatch.setattr(sys, 'argv', ['main.py', '--mode', 'extract', '--inverted-control', '--min-free-gb', '0', '--cache-dir', str(tmp_path / 'c'), '--output', str(tmp_path / 'o.csv')])
    main_mod.main()
    assert '--inverted-control' in seen['cmd']
