import time
from pathlib import Path

import lightkurve as lk
import numpy as np
import pytest

from test_reestimate_transit import prepared, run_search, search_like, spy_search_values, stub_search_stage
from test_restore_transit_points import CADENCE_D, PERIOD, T0, make_curve, near, run_star, same_value, star_collection, times_of

STAR_FEATS = {'star_radius': 1.0, 'star_logg': 4.4}
ARTIFACT_STAR = dict(sigma=1e-3, deep=0.0, box_depth=1.5e-3, box_period=4.0, box_t0=0.9, box_low_runs=3, seed=11)


def noise_curve(depth, sigma, period, t0, dur, n=16000, seed=5, runs_in=0, runs_out=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * CADENCE_D
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    in_tr = np.abs(phase) < dur / 2.0
    f = 1.0 + rng.normal(0.0, sigma, n) - depth * in_tr
    starts_in = rng.choice(np.where(in_tr)[0][5:-5], size=runs_in, replace=False)
    starts_out = rng.choice(np.where(np.abs(phase) > 3.0 * dur)[0][50:-50], size=runs_out, replace=False)
    art = np.zeros(n, dtype=bool)
    for j, i in enumerate(np.concatenate([starts_in, starts_out]).astype(int)):
        art[i:i + 2 + j % 2] = True
    f[art] = 1.0 - 20.0 * sigma
    lc = lk.LightCurve(time=t, flux=f, flux_err=np.full(n, sigma))
    return lc, in_tr, art


def real_search(extractor, lc_flat, lc_restore):
    t = times_of(lc_flat)
    baseline = float(np.max(t) - np.min(t))
    return extractor.search_signals(lc_flat, '24680', 'Kepler', [], baseline, CADENCE_D, STAR_FEATS, False, 1, time.time() + 600.0, True, 'test', lc_restore=lc_restore)


def assert_same_result(feats_a, curves_a, feats_b, curves_b):
    assert len(feats_a) == len(feats_b) >= 1
    for a, b in zip(feats_a, feats_b):
        assert set(a) == set(b)
        for key in a:
            assert same_value(a[key], b[key]), key
    assert set(curves_a) == set(curves_b)
    for key in curves_a:
        assert np.array_equal(curves_a[key], curves_b[key])


def window_case(n_pool_in, n_pool_out, n_search_in, period=10.0, epoch=5.0, dur=0.4):
    def inside(n, shift):
        i = np.arange(n)
        return epoch + (i % 30) * period + np.linspace(-0.2, 0.2, n) + shift

    def outside(n, shift):
        i = np.arange(n)
        return epoch + (i % 30) * period + 2.0 + 2.5 * i / max(n, 1) + shift

    t_s = np.sort(np.concatenate([inside(n_search_in, 0.0), outside(600, 0.0)]))
    t_r = np.sort(np.concatenate([inside(n_pool_in, 1e-4), outside(n_pool_out, 1e-4)]))
    lc_search = lk.LightCurve(time=t_s, flux=np.ones(len(t_s)), flux_err=np.full(len(t_s), 1e-4))
    lc_restore = lk.LightCurve(time=t_r, flux=np.full(len(t_r), 0.99), flux_err=np.full(len(t_r), 1e-4))
    return lc_search, lc_restore, period, epoch, dur


def uniform_pool_case(thin=1):
    n = 20000
    period, epoch, dur = 10.0, 3.0, 0.1
    rng = np.random.default_rng(21)
    t = np.arange(n) * CADENCE_D
    f = 1.0 + rng.normal(0.0, 2e-4, n)
    t_pool = np.sort(rng.uniform(t[0], t[-1], 2000))
    phase = ((t - epoch + 0.5 * period) % period) - 0.5 * period
    in_win = np.abs(phase) < dur * 1.25 / 2.0
    keep = ~in_win | (np.cumsum(in_win) % thin == 0)
    lc_search = lk.LightCurve(time=t[keep], flux=f[keep], flux_err=np.full(int(np.sum(keep)), 2e-4))
    lc_restore = lk.LightCurve(time=t_pool, flux=np.full(2000, 0.99), flux_err=np.full(2000, 2e-4))
    return lc_search, lc_restore, period, epoch, dur


def test_admission_constants_live_next_to_restore_constants(extractor):
    assert extractor.RESTORE_MIN_FRACTION == 0.05
    assert extractor.RESTORE_MIN_CHANCE_RATIO == 3.0
    src = Path(extractor.__file__).read_text(encoding='utf-8')
    block = "RESTORE_TRANSIT_POINTS = True\nRESTORE_WINDOW_FACTOR = 1.25\nRESTORE_MIN_FRACTION = 0.05\nRESTORE_MIN_CHANCE_RATIO = 3.0\nREESTIMATE_MIN_RESTORED_FRACTION = 0.2\n"
    assert src.count(block) == 1
    assert src.index(block) < src.index('\ndef ')


def test_restore_allowed_thresholds_are_inclusive(extractor):
    lc_search, lc_restore, period, epoch, dur = window_case(15, 85, 100)
    assert len(lc_restore) == 100
    lc_cand, n_in_pool = extractor.restore_transit_points(lc_search, lc_restore, period, epoch, dur)
    assert n_in_pool == 15
    assert extractor.restored_fraction(lc_search, 15, period, epoch, dur) == pytest.approx(15.0 / 115.0)
    assert extractor.restore_allowed(lc_search, lc_restore, 15, period, epoch, dur) is True

    lc_search, lc_restore, period, epoch, dur = window_case(14, 86, 100)
    lc_cand, n_in_pool = extractor.restore_transit_points(lc_search, lc_restore, period, epoch, dur)
    assert n_in_pool == 14
    assert extractor.restored_fraction(lc_search, 14, period, epoch, dur) > extractor.RESTORE_MIN_FRACTION
    assert extractor.restore_allowed(lc_search, lc_restore, 14, period, epoch, dur) is False

    lc_search, lc_restore, period, epoch, dur = window_case(15, 85, 285)
    assert extractor.restored_fraction(lc_search, 15, period, epoch, dur) == 0.05
    assert extractor.restore_allowed(lc_search, lc_restore, 15, period, epoch, dur) is True

    lc_search, lc_restore, period, epoch, dur = window_case(15, 85, 286)
    assert extractor.restored_fraction(lc_search, 15, period, epoch, dur) < extractor.RESTORE_MIN_FRACTION
    assert extractor.restore_allowed(lc_search, lc_restore, 15, period, epoch, dur) is False


def test_restore_allowed_rejects_degenerate_inputs(extractor):
    lc_search, lc_restore, period, epoch, dur = window_case(40, 10, 100)
    assert extractor.restore_allowed(lc_search, lc_restore, 40, period, epoch, dur) is True
    assert extractor.restore_allowed(lc_search, None, 40, period, epoch, dur) is False
    assert extractor.restore_allowed(lc_search, lc_restore, 0, period, epoch, dur) is False
    assert extractor.restore_allowed(lc_search, lc_restore, 40, 0.0, epoch, dur) is False
    assert extractor.restore_allowed(lc_search, lc_restore, 40, period, epoch, 0.0) is False
    lc_wide, pool_wide, period, epoch, dur = window_case(50, 0, 100)
    assert extractor.restore_allowed(lc_wide, pool_wide, 50, 0.6, epoch, dur) is False


def test_shallow_transit_with_chance_artifacts_is_not_restored(extractor, monkeypatch):
    sigma, depth, period, t0, dur = 5e-4, 1e-4, 2.0, 0.7, 0.25
    lc, in_tr, art = noise_curve(depth, sigma, period, t0, dur, runs_in=3, runs_out=7)
    lc_flat, lc_restore = prepared(extractor, lc)
    assert len(lc_restore) == int(np.sum(art))
    assert 20 <= len(lc_restore) <= 30
    refined = spy_search_values(extractor, monkeypatch)
    cand_times = []
    boxed = []
    reest = []
    orig_count = extractor.count_observed_transits
    orig_box = extractor.box_depth_ppm
    orig_reest = extractor.reestimate_transit

    def spy_count(t, *a, **k):
        cand_times.append(np.array(t, dtype=float, copy=True))
        return orig_count(t, *a, **k)

    def spy_box(*a, **k):
        boxed.append(1)
        return orig_box(*a, **k)

    def spy_reest(*a, **k):
        reest.append(1)
        return orig_reest(*a, **k)

    monkeypatch.setattr(extractor, 'count_observed_transits', spy_count)
    monkeypatch.setattr(extractor, 'box_depth_ppm', spy_box)
    monkeypatch.setattr(extractor, 'reestimate_transit', spy_reest)
    res = real_search(extractor, lc_flat, lc_restore)
    assert len(res['features']) == 1
    feat = res['features'][0]
    s = refined[0]
    assert abs(feat['bls_period_d'] / period - 1.0) < 0.01
    n_in_pool = res['restore_candidates'][0]
    assert 4 <= n_in_pool < len(lc_restore)
    assert res['restored_points'] == [0]
    assert res['reestimated'] == [0]
    assert boxed == [] and reest == []
    assert np.array_equal(cand_times[0], times_of(lc_flat))
    assert feat['bls_depth_ppm'] == float(s['depth'] * 1e6)
    assert feat['bls_duration_d'] == s['duration']
    assert feat['bls_epoch'] == s['epoch']
    assert abs(feat['bls_depth_ppm'] / (depth * 1e6) - 1.0) < 0.25
    assert extractor.restored_fraction(lc_flat, n_in_pool, s['period'], s['epoch'], s['duration']) < extractor.RESTORE_MIN_FRACTION
    assert extractor.restore_allowed(lc_flat, lc_restore, n_in_pool, s['period'], s['epoch'], s['duration']) is False
    lc_old, n_old = extractor.restore_transit_points(lc_flat, lc_restore, s['period'], s['epoch'], s['duration'])
    assert n_old == n_in_pool
    assert orig_box(lc_old, s['period'], s['epoch'], s['duration']) > 1.2 * feat['bls_depth_ppm']
    plain = real_search(extractor, lc_flat, None)
    assert plain['restore_candidates'] == [0] and plain['restored_points'] == [0]
    assert_same_result(res['features'], res['curves'], plain['features'], plain['curves'])
    assert np.array_equal(times_of(res['lc_residual']), times_of(plain['lc_residual']))


@pytest.mark.parametrize("depth", [0.05, 0.20])
@pytest.mark.parametrize("hours", [2.0, 4.0, 8.0])
def test_deep_transits_are_admitted_and_match_round_two_result(extractor, monkeypatch, depth, hours):
    lc, in_tr, phase, dur = make_curve(depth, hours)
    lc_flat, lc_restore = prepared(extractor, lc)
    grid, ep_s, dur_s = search_like(extractor, dur)
    n_tr = int(np.sum(in_tr))
    lc_cand, n_in_pool = extractor.restore_transit_points(lc_flat, lc_restore, PERIOD, ep_s, dur_s)
    assert n_in_pool == n_tr == len(lc_restore)
    assert extractor.restored_fraction(lc_flat, n_in_pool, PERIOD, ep_s, dur_s) >= 0.5
    assert n_in_pool >= 20.0 * len(lc_restore) * dur_s * extractor.RESTORE_WINDOW_FACTOR / PERIOD
    assert extractor.restore_allowed(lc_flat, lc_restore, n_in_pool, PERIOD, ep_s, dur_s) is True
    local = extractor.reestimate_transit(lc_cand, PERIOD, grid)
    stub_search_stage(extractor, monkeypatch, PERIOD, ep_s, dur_s)
    res = run_search(extractor, lc_flat, lc_restore)
    feat = res['features'][0]
    assert res['restore_candidates'] == [n_tr]
    assert res['restored_points'] == [n_tr]
    assert res['reestimated'] == [1]
    assert feat['bls_duration_d'] == local['duration']
    assert feat['bls_depth_ppm'] == float(local['depth'] * 1e6)
    assert feat['bls_epoch'] == float(ep_s + ((local['epoch'] - ep_s + 0.5 * PERIOD) % PERIOD) - 0.5 * PERIOD)
    assert abs(feat['bls_depth_ppm'] / (depth * 1e6) - 1.0) < 0.10
    assert abs(feat['bls_duration_d'] / dur - 1.0) < 0.15
    assert feat['n_transits_observed'] >= 20


@pytest.mark.parametrize("seed", [2, 3])
def test_borderline_five_sigma_transit_is_admitted_and_depth_recovered(extractor, seed):
    sigma = 2e-4
    depth = 5.0 * sigma
    dur = 3.0 / 24.0
    lc, in_tr, art = noise_curve(depth, sigma, PERIOD, T0, dur, seed=seed)
    kept, removed = extractor.clip_outliers_with_removed(lc)
    cut = np.isin(times_of(lc), times_of(removed))
    cut_share = float(np.sum(cut & in_tr)) / float(np.sum(in_tr))
    assert 0.35 < cut_share < 0.65
    assert int(np.sum(cut & ~in_tr)) == 0
    lc_flat, lc_restore = prepared(extractor, lc)
    clipped_depth = extractor.box_depth_ppm(lc_flat, PERIOD, T0, dur)
    assert abs(clipped_depth / (depth * 1e6) - 1.0) > 0.10
    res = real_search(extractor, lc_flat, lc_restore)
    assert len(res['features']) == 1
    feat = res['features'][0]
    assert abs(feat['bls_period_d'] / PERIOD - 1.0) < 0.01
    assert res['restore_candidates'] == [int(np.sum(cut & in_tr))]
    assert res['restored_points'] == res['restore_candidates']
    assert abs(feat['bls_depth_ppm'] / (depth * 1e6) - 1.0) < 0.10
    assert abs(feat['bls_depth_ppm'] / (depth * 1e6) - 1.0) < abs(clipped_depth / (depth * 1e6) - 1.0)
    assert feat['n_transits_observed'] >= 20


@pytest.mark.parametrize("thin", [1, 4])
def test_uniform_pool_is_rejected_by_chance_ratio_despite_fraction(extractor, monkeypatch, thin):
    lc_search, lc_restore, period, epoch, dur = uniform_pool_case(thin)
    assert len(lc_restore) == 2000
    lc_cand, n_in_pool = extractor.restore_transit_points(lc_search, lc_restore, period, epoch, dur)
    n_chance = 2000.0 * dur * extractor.RESTORE_WINDOW_FACTOR / period
    assert n_chance == pytest.approx(25.0)
    assert 10 <= n_in_pool < extractor.RESTORE_MIN_CHANCE_RATIO * n_chance
    fraction = extractor.restored_fraction(lc_search, n_in_pool, period, epoch, dur)
    assert fraction > extractor.RESTORE_MIN_FRACTION
    if thin > 1:
        assert fraction > extractor.REESTIMATE_MIN_RESTORED_FRACTION
    assert extractor.restore_allowed(lc_search, lc_restore, n_in_pool, period, epoch, dur) is False
    monkeypatch.setattr(extractor, 'RESTORE_MIN_CHANCE_RATIO', 0.1)
    assert extractor.restore_allowed(lc_search, lc_restore, n_in_pool, period, epoch, dur) is True


def test_search_keeps_search_curve_when_uniform_pool_fails_chance_ratio(extractor, monkeypatch):
    lc_search, lc_restore, period, epoch, dur = uniform_pool_case()
    stub_search_stage(extractor, monkeypatch, period, epoch, dur)
    reest = []
    monkeypatch.setattr(extractor, 'reestimate_transit', lambda *a, **k: reest.append(1))
    res = run_search(extractor, lc_search, lc_restore)
    assert len(res['features']) == 1
    feat = res['features'][0]
    lc_cand, n_in_pool = extractor.restore_transit_points(lc_search, lc_restore, period, epoch, dur)
    assert res['restore_candidates'] == [n_in_pool]
    assert n_in_pool >= 10
    assert res['restored_points'] == [0]
    assert res['reestimated'] == [0]
    assert reest == []
    assert feat['bls_depth_ppm'] == float(1e-4 * 1e6)
    assert feat['bls_duration_d'] == dur
    assert feat['bls_epoch'] == epoch
    plain = run_search(extractor, lc_search, None)
    assert_same_result(res['features'], res['curves'], plain['features'], plain['curves'])


def test_rejected_restore_equals_no_restore_flag_end_to_end(extractor, monkeypatch, tmp_path):
    refined = spy_search_values(extractor, monkeypatch)
    feats_on, curves_on, timings_on = run_star(extractor, monkeypatch, tmp_path, star_collection(**ARTIFACT_STAR))
    sig = feats_on[0]
    assert len(near(feats_on, 4.0)) == 1 and near(feats_on, 4.0)[0] is sig
    assert timings_on['restore_pool_points'] == 12
    assert timings_on['restore_candidates'][0] == 12
    assert timings_on['restored_points'][0] == 0
    assert all(n == 0 for n in timings_on['restored_points'])
    assert timings_on['n_reestimated'] == 0
    assert len(timings_on['restore_candidates']) == len(timings_on['restored_points']) == len(feats_on)
    assert sig['bls_depth_ppm'] == float(refined[0]['depth'] * 1e6)
    assert sig['bls_duration_d'] == refined[0]['duration']
    assert sig['bls_epoch'] == refined[0]['epoch']
    fields = extractor.diag_restore_fields(timings_on)
    assert list(fields) == ['restore_pool_points', 'restored_points', 'restore_candidates']
    assert fields['restore_candidates'].split(';')[0] == '12'
    assert fields['restored_points'].split(';')[0] == '0'

    feats_off, curves_off, timings_off = run_star(extractor, monkeypatch, tmp_path, star_collection(**ARTIFACT_STAR), restore_transits=False)
    assert timings_off['restore_pool_points'] == 0
    assert all(n == 0 for n in timings_off['restored_points'])
    assert all(n == 0 for n in timings_off['restore_candidates'])
    assert_same_result(feats_on, curves_on, feats_off, curves_off)

    monkeypatch.setattr(extractor, 'RESTORE_TRANSIT_POINTS', False)
    feats_flag, curves_flag, timings_flag = run_star(extractor, monkeypatch, tmp_path, star_collection(**ARTIFACT_STAR))
    assert timings_flag['restore_pool_points'] == 0
    assert_same_result(feats_flag, curves_flag, feats_off, curves_off)


def test_no_restore_flag_equals_search_without_restore_curve(extractor, monkeypatch, tmp_path):
    calls = []
    orig = extractor.search_signals

    def spy(lc, *a, **k):
        res = orig(lc, *a, **k)
        calls.append({'lc': lc.copy(), 'args': a, 'kwargs': dict(k), 'res': res})
        return res

    monkeypatch.setattr(extractor, 'search_signals', spy)
    feats_off, curves_off, timings_off = run_star(extractor, monkeypatch, tmp_path, star_collection(box_depth=1e-3), restore_transits=False)
    assert len(calls) == 1
    assert calls[0]['kwargs']['lc_restore'] is None
    assert calls[0]['res']['restore_candidates'] == [0] * len(feats_off)
    assert calls[0]['res']['restored_points'] == [0] * len(feats_off)
    plain_kwargs = {k: v for k, v in calls[0]['kwargs'].items() if k != 'lc_restore'}
    plain = orig(calls[0]['lc'], *calls[0]['args'], **plain_kwargs)
    assert_same_result(plain['features'], plain['curves'], feats_off, curves_off)
    feats_on, curves_on, timings_on = run_star(extractor, monkeypatch, tmp_path, star_collection(box_depth=1e-3))
    assert near(feats_off, 4.0) == [] and len(near(feats_on, 4.0)) == 1
    i_deep = feats_on.index(near(feats_on, 4.0)[0])
    assert timings_on['restore_candidates'][i_deep] == timings_on['restored_points'][i_deep] > 100
    assert timings_on['reestimated'][i_deep] == 1
    assert sum(timings_on['restored_points']) == timings_on['restored_points'][i_deep]
