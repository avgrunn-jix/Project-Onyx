import time

import numpy as np
import pytest
import lightkurve as lk

from test_search_refine import CADENCE, box_series, star_feats


def _run(extractor, lc, inverted, quick=True, baseline=None):
    t = np.asarray(lc.time.value, dtype=float)
    b = float(t.max() - t.min()) if baseline is None else baseline
    return extractor.search_signals(lc, '42', 'Kepler', [], b, CADENCE, star_feats(), quick, 4, time.time() + 600, True, 'v', inverted=inverted)


def test_invert_light_curve_mirrors_flux(extractor):
    t, f, dy = box_series(5.0, 1.0, 0.2, 2e-3, 50.0, seed=1)
    lc = lk.LightCurve(time=t, flux=f, flux_err=dy)
    inv = extractor.invert_light_curve(lc)
    assert np.allclose(np.asarray(inv.flux.value), 2.0 - f)
    assert np.allclose(np.asarray(inv.flux_err.value), dy)
    assert np.allclose(np.asarray(lc.flux.value), f)


def test_inverted_search_does_not_find_real_transits(extractor):
    t, f, dy = box_series(5.0, 1.3, 0.2, 2e-3, 300.0, seed=2)
    lc = lk.LightCurve(time=t, flux=f, flux_err=dy)
    main = _run(extractor, lc, False)
    assert any(abs(x['bls_period_d'] / 5.0 - 1.0) < 0.01 for x in main['features'])
    inv = _run(extractor, extractor.invert_light_curve(main['lc_residual']), True)
    assert inv['features'] == []


def test_unmasked_inversion_shows_transit_induced_artefact(extractor):
    t, f, dy = box_series(5.0, 1.3, 0.2, 2e-3, 300.0, seed=2)
    lc = lk.LightCurve(time=t, flux=f, flux_err=dy)
    inv = _run(extractor, extractor.invert_light_curve(lc), True)
    assert len(inv['features']) >= 1
    assert all(abs(x['bls_period_d'] / 5.0 - 1.0) >= 0.01 for x in inv['features'])
    assert extractor.INVERTED_MASK_MAIN_SIGNALS is True


def test_inverted_signals_have_main_columns(extractor):
    t, f, dy = box_series(3.0, 0.8, 0.15, 3e-3, 300.0, seed=8)
    lc_dips = lk.LightCurve(time=t, flux=f, flux_err=dy)
    main = _run(extractor, lc_dips, False)
    lc_bumps = lk.LightCurve(time=t, flux=2.0 - f, flux_err=dy)
    inv = _run(extractor, extractor.invert_light_curve(lc_bumps), True)
    assert main['features'] and inv['features']
    assert any(abs(x['bls_period_d'] / 3.0 - 1.0) < 0.01 for x in inv['features'])
    for x in inv['features']:
        assert list(x.keys()) == list(main['features'][0].keys())
        assert x['match_status'] == 'INVERTED' and x['label'] == 'INVERTED'
        assert x['signal_id'].startswith('42_inv')
        assert x['pipeline_version'] == main['features'][0]['pipeline_version']


def test_white_noise_same_detection_rate_in_main_and_inverted(extractor):
    n_main = 0
    n_inv = 0
    for seed in (31, 32, 33, 34):
        t, f, dy = box_series(None, 0, 0, 0, 300.0, seed=seed)
        lc = lk.LightCurve(time=t, flux=f, flux_err=dy)
        n_main += len(_run(extractor, lc, False)['features'])
        n_inv += len(_run(extractor, extractor.invert_light_curve(lc), True)['features'])
    assert n_main == n_inv == 0


class FakeTable:
    colnames = []


class CountingSearch:
    def __init__(self, coll, counter):
        self.coll = coll
        self.table = FakeTable()
        self.counter = counter

    def __len__(self):
        return len(self.coll)

    def download_all(self, download_dir=None):
        self.counter['download'] += 1
        return self.coll


def dips_collection():
    rng = np.random.default_rng(5)
    segs = []
    for q, (a, b) in enumerate([(0.0, 60.0), (62.0, 120.0)], start=1):
        t = np.arange(a, b, CADENCE)
        f = 1.0 + rng.normal(0.0, 1e-3, len(t))
        ph = ((t - 0.9 + 2.0) % 4.0) - 2.0
        f[np.abs(ph) < 0.1] -= 6e-3
        lc = lk.LightCurve(time=t, flux=f, flux_err=np.full(len(t), 1e-3))
        lc.meta['QUARTER'] = q
        segs.append(lc)
    return lk.LightCurveCollection(segs)


def test_inverted_control_in_process_single_star_downloads_once(extractor, monkeypatch, tmp_path):
    counter = {'search': 0, 'download': 0}
    coll = dips_collection()

    def fake_search(*a, **k):
        counter['search'] += 1
        return CountingSearch(coll, counter)

    calls = []
    orig = extractor.search_signals

    def spy(lc, *a, **k):
        calls.append((k.get('inverted', False), np.asarray(lc.flux.value, dtype=float).copy(), np.asarray(lc.time.value, dtype=float).copy()))
        res = orig(lc, *a, **k)
        calls[-1] = calls[-1] + (res,)
        return res

    monkeypatch.setattr(extractor.lk, 'search_lightcurve', fake_search)
    monkeypatch.setattr(extractor, '_dl_sem', None)
    monkeypatch.setattr(extractor, 'search_signals', spy)
    rows = [{'catalog_period': 4.0, 'label': 'CONFIRMED'}]
    feats, curves, tid, err, dl_size, timings = extractor.process_single_star('7654321', 'Kepler', rows, False, 0.0, 3.0, 4, quick_mode=True, cache_dir=str(tmp_path), inverted_control=True)
    assert err is None, err
    assert counter == {'search': 1, 'download': 1}
    assert [c[0] for c in calls] == [False, True]
    residual = calls[0][3]['lc_residual']
    assert np.allclose(calls[1][1], 2.0 - np.asarray(residual.flux.value, dtype=float))
    assert np.allclose(calls[1][2], np.asarray(residual.time.value, dtype=float))
    assert timings['inverted_control'] == 1
    assert any(abs(x['bls_period_d'] / 4.0 - 1.0) < 0.01 for x in feats)
    for x in timings['inverted_features']:
        assert x['match_status'] == 'INVERTED'
        assert x['pipeline_version'] == feats[0]['pipeline_version']
    assert not any('_inv' in k for k in curves)


def test_inverted_control_off_by_default(extractor, monkeypatch, tmp_path):
    counter = {'search': 0, 'download': 0}
    coll = dips_collection()
    monkeypatch.setattr(extractor.lk, 'search_lightcurve', lambda *a, **k: CountingSearch(coll, counter))
    monkeypatch.setattr(extractor, '_dl_sem', None)
    calls = []
    orig = extractor.search_signals

    def spy(*a, **k):
        calls.append(k.get('inverted', False))
        return orig(*a, **k)

    monkeypatch.setattr(extractor, 'search_signals', spy)
    out = extractor.process_single_star('7654321', 'Kepler', [], False, 0.0, 3.0, 4, quick_mode=True, cache_dir=str(tmp_path))
    assert out[3] is None
    assert calls == [False]
    assert out[5]['inverted_features'] == []
    assert out[5]['inverted_control'] == 0


def test_inverted_output_path_and_diag_fields(extractor):
    import os
    assert extractor.inverted_output_path(os.path.join('a', 'features_demo_quick.csv')) == os.path.join('a', 'features_demo_quick_inverted.csv')
    row = extractor.failed_diag_row('1', 'e', 1.0, 'v', True)
    assert row['inverted_control'] == 1 and row['inverted_count'] == 0
    assert list(row.keys())[-2:] == ['inverted_control', 'inverted_count']
    assert extractor.diag_inverted_fields(False, 5) == {'inverted_control': 0, 'inverted_count': 0}
