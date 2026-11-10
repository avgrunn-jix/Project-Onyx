import os
import sys

import numpy as np
import pandas as pd

from synth_demo import SHOWCASE, FEATURE_COLUMNS, add_inverted_control, build_workspace


def _signals(rng, n_stars, per_star, real, prefix):
    rows = []
    for s in range(n_stars):
        for k in range(per_star):
            row = {c: float(rng.lognormal(0.0, 0.3)) for c in FEATURE_COLUMNS}
            row['bls_sde'] = float(rng.normal(25.0, 4.0) if real else rng.normal(7.6, 0.4))
            row['bls_depth_snr'] = float(rng.normal(20.0, 4.0) if real else rng.normal(7.5, 0.3))
            row['single_event_fraction'] = float(rng.uniform(0.01, 0.1) if real else rng.uniform(0.2, 0.6))
            row['target_id'] = f'{prefix}{s}'
            row['signal_id'] = f'{prefix}{s}_{k}'
            rows.append(row)
    return pd.DataFrame(rows)


def test_choose_real_threshold_keeps_target_share(demo):
    y = np.array([1] * 20 + [0] * 20)
    s = np.concatenate([np.linspace(0.5, 1.0, 20), np.linspace(0.0, 0.6, 20)])
    th = demo.choose_real_threshold(s, y, 0.95)
    assert np.mean(s[y == 1] >= th) >= 0.95
    assert th == np.sort(s[y == 1])[1]


def test_apply_noise_gate_only_touches_low_p_real_candidates(demo):
    dec = np.array(['CANDIDATE', 'CANDIDATE', 'FALSE_POSITIVE', 'REVIEW', 'CANDIDATE'], dtype=object)
    p = np.array([0.9, 0.2, 0.1, 0.05, np.nan])
    out, n = demo.apply_noise_gate(dec, p, 0.5)
    assert list(out) == ['CANDIDATE', 'REVIEW', 'FALSE_POSITIVE', 'REVIEW', 'REVIEW']
    assert n == 2
    assert list(dec) == ['CANDIDATE', 'CANDIDATE', 'FALSE_POSITIVE', 'REVIEW', 'CANDIDATE']


def test_threshold_rejects_noise_and_keeps_high_p_real_matched(demo):
    rng = np.random.default_rng(0)
    real_tr = _signals(rng, 20, 2, True, 'r')
    noise_tr = _signals(rng, 20, 2, False, 'n')
    feats = [c for c in FEATURE_COLUMNS if c != 'baseline_d']
    model, counts = demo.train_signal_noise(real_tr, noise_tr, feats, 7)
    assert model is not None and counts == {1: 40, 0: 40}
    assert not set(model['cols']) & set(demo.STAR_FEATURE_COLS)
    assert model['oof_real_kept'] >= 0.95
    real_ct = _signals(rng, 10, 2, True, 'rc')
    noise_ct = _signals(rng, 10, 2, False, 'nc')
    p_real = demo.predict_p_real(model, real_ct)
    p_noise = demo.predict_p_real(model, noise_ct)
    assert np.mean(p_noise < model['threshold']) >= 0.9
    high = p_real >= model['threshold']
    assert np.mean(high) >= 0.9
    dec_real, n_real = demo.apply_noise_gate(np.array(['CANDIDATE'] * len(real_ct), dtype=object), p_real, model['threshold'])
    assert all(dec_real[high] == 'CANDIDATE')
    dec_noise, n_noise = demo.apply_noise_gate(np.array(['CANDIDATE'] * len(noise_ct), dtype=object), p_noise, model['threshold'])
    assert n_noise >= 0.9 * len(noise_ct)


def test_train_signal_noise_requires_both_classes(demo):
    rng = np.random.default_rng(1)
    model, counts = demo.train_signal_noise(_signals(rng, 4, 1, True, 'r'), _signals(rng, 2, 1, False, 'n'), list(FEATURE_COLUMNS), 1)
    assert model is None
    assert counts == {1: 4, 0: 2}


def _report(demo, monkeypatch, tmp_path, ws):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--quick', '--report-only', '--output-dir', ws['out_dir'], '--showcase-star', SHOWCASE, '--cache-dir', str(tmp_path / 'c')])
    demo.main()
    with open(os.path.join(ws['out_dir'], 'summary_quick_report.md'), encoding='utf-8') as fh:
        return fh.read()


def test_demo_noise_gate_section_and_columns(demo, monkeypatch, tmp_path):
    ws = add_inverted_control(build_workspace(tmp_path), marked_rate=1.0, unmarked_rate=1.0)
    text = _report(demo, monkeypatch, tmp_path, ws)
    assert '4.1. Этап «сигнал или шум» (p_real)' in text
    assert 'Порог p_real:' in text
    assert 'CANDIDATE среди сигналов звёзд без отметки' in text
    for name in ('MATCHED (контроль', 'UNMATCHED (контроль)', 'INVERTED (контроль)'):
        assert name in text
    ctrl = pd.read_csv(os.path.join(ws['out_dir'], 'control_signals_quick_report.csv'), dtype={'target_id': str})
    assert {'p_real', 'decision_before_noise_gate', 'decision'} <= set(ctrl.columns)
    changed = ctrl[ctrl['decision'] != ctrl['decision_before_noise_gate']]
    assert set(changed['decision_before_noise_gate']) <= {'CANDIDATE'}
    assert set(changed['decision']) <= {'REVIEW'}


def test_demo_without_inverted_keeps_old_verdicts(demo, monkeypatch, tmp_path):
    ws = build_workspace(tmp_path)
    text = _report(demo, monkeypatch, tmp_path, ws)
    assert 'Этап пропущен: нет файла INVERTED-сигналов' in text
    ctrl = pd.read_csv(os.path.join(ws['out_dir'], 'control_signals_quick_report.csv'), dtype={'target_id': str})
    assert 'p_real' not in ctrl.columns
    assert 'decision_before_noise_gate' not in ctrl.columns
