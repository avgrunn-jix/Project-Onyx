import time

import lightkurve as lk
import numpy as np
import pytest

from test_restore_transit_points import CADENCE_D, PERIOD, SIGMA, STAR_PERIOD, STAR_TRANSITS, T0, make_curve, near, run_star, same_value, star_collection, times_of, window_length, window_points

STUB_SNR = 55.5
STUB_SDE = 30.0
STUB_POWER = 123.0


def phase_offset(epoch, ref, period):
    return ((epoch - ref + 0.5 * period) % period) - 0.5 * period


def search_like(extractor, dur):
    grid = extractor.duration_grid(False)
    dur_s = float(grid[grid >= 1.2 * dur][0])
    ep_s = T0 + 0.1 * dur
    return grid, ep_s, dur_s


def prepared(extractor, lc):
    kept, removed = extractor.clip_outliers_with_removed(lc)
    lc_flat, lc_restore = extractor.flatten_with_restore(kept, removed, window_length())
    return lc_flat, lc_restore


def stub_search_stage(extractor, monkeypatch, period, epoch, duration):
    def fake_spectrum(t, f, dy, dur_grid, min_period, max_period, freq_factor, max_duty=0.3):
        periods = np.linspace(min_period, max_period, 3000)
        power = 1.0 + STUB_POWER * np.exp(-0.5 * ((periods - period) / 0.05) ** 2)
        n = len(periods)
        return {'bls': None, 'dur_grid': np.sort(np.asarray(dur_grid, dtype=float)), 'max_duty': float(max_duty), 'period': periods, 'power': power, 'duration': np.full(n, duration), 'transit_time': np.full(n, epoch), 'depth': np.full(n, 1e-4), 'depth_snr': np.full(n, STUB_SNR)}

    def fake_refine(spec, idx, baseline_d):
        return {'period': float(period), 'epoch': float(epoch), 'duration': float(duration), 'depth': 1e-4, 'depth_snr': STUB_SNR, 'power': float(spec['power'][idx]), 'refined': True}

    monkeypatch.setattr(extractor, 'bls_spectrum', fake_spectrum)
    monkeypatch.setattr(extractor, 'refine_peak', fake_refine)
    monkeypatch.setattr(extractor, 'detrended_sde', lambda power, idx, window=501: STUB_SDE)


def spy_search_values(extractor, monkeypatch):
    seen = []
    orig_refine = extractor.refine_peak
    orig_alias = extractor.resolve_alias_period

    def spy_refine(*a, **k):
        r = orig_refine(*a, **k)
        seen.append({'period': r['period'], 'epoch': r['epoch'], 'duration': r['duration'], 'depth': r['depth'], 'depth_snr': r['depth_snr'], 'factor': 1.0})
        return r

    def spy_alias(*a, **k):
        r = orig_alias(*a, **k)
        if r['factor'] != 1.0:
            seen[-1].update({'period': r['period'], 'epoch': r['epoch'], 'depth': r['depth'], 'factor': r['factor']})
        return r

    monkeypatch.setattr(extractor, 'refine_peak', spy_refine)
    monkeypatch.setattr(extractor, 'resolve_alias_period', spy_alias)
    return seen


def run_search(extractor, lc_flat, lc_restore, max_signals=1):
    t = times_of(lc_flat)
    baseline = float(np.max(t) - np.min(t))
    star_feats = {'star_radius': 1.0, 'star_logg': 4.4}
    return extractor.search_signals(lc_flat, '13579', 'Kepler', [], baseline, CADENCE_D, star_feats, False, max_signals, time.time() + 600.0, False, 'test', lc_restore=lc_restore)


def test_restored_fraction_counts_search_points_in_restore_windows(extractor):
    assert extractor.REESTIMATE_MIN_RESTORED_FRACTION == 0.2
    period, epoch, dur = 10.0, 5.0, 0.4
    offsets = np.array([0.0, 0.1, -0.1, 0.24, -0.24, 0.26, 3.0])
    t = np.concatenate([5.0 + offsets, 15.0 + offsets, [1.0, 2.0, 8.0, 12.0]])
    lc_search = lk.LightCurve(time=np.sort(t), flux=np.ones(len(t)), flux_err=np.full(len(t), 1e-4))
    assert extractor.restored_fraction(lc_search, 0, period, epoch, dur) == 0.0
    assert extractor.restored_fraction(lc_search, 5, 0.0, epoch, dur) == 0.0
    assert extractor.restored_fraction(lc_search, 5, period, epoch, 0.0) == 0.0
    assert extractor.restored_fraction(lc_search, 10, period, epoch, dur) == pytest.approx(0.5)
    assert extractor.restored_fraction(lc_search, 2, period, epoch, dur) == pytest.approx(2.0 / 12.0)
    assert extractor.restored_fraction(lc_search, 3, period, epoch, dur) >= extractor.REESTIMATE_MIN_RESTORED_FRACTION
    empty = lc_search[np.zeros(len(t), dtype=bool)]
    assert extractor.restored_fraction(empty, 4, period, epoch, dur) == 1.0


@pytest.mark.parametrize("depth", [0.05, 0.20])
@pytest.mark.parametrize("hours", [2.0, 4.0, 8.0])
def test_reestimate_recovers_duration_epoch_and_depth_of_deep_transits(extractor, depth, hours):
    lc, in_tr, phase, dur = make_curve(depth, hours)
    lc_flat, lc_restore = prepared(extractor, lc)
    grid, ep_s, dur_s = search_like(extractor, dur)
    assert dur_s >= 1.2 * dur
    lc_cand, n_restored = extractor.restore_transit_points(lc_flat, lc_restore, PERIOD, ep_s, dur_s)
    assert n_restored == int(np.sum(in_tr))
    assert extractor.restored_fraction(lc_flat, n_restored, PERIOD, ep_s, dur_s) >= extractor.REESTIMATE_MIN_RESTORED_FRACTION
    wide_depth = extractor.box_depth_ppm(lc_cand, PERIOD, ep_s, dur_s)
    assert wide_depth < 0.9 * depth * 1e6
    local = extractor.reestimate_transit(lc_cand, PERIOD, grid)
    assert set(local) == {'duration', 'epoch', 'depth'}
    assert abs(local['duration'] / dur - 1.0) < 0.15
    assert abs(local['depth'] / depth - 1.0) < 0.10
    assert abs(phase_offset(local['epoch'], T0, PERIOD)) < 0.15 * dur
    assert abs(local['duration'] / dur - 1.0) < abs(dur_s / dur - 1.0)


@pytest.mark.parametrize("depth", [0.05, 0.20])
@pytest.mark.parametrize("hours", [2.0, 4.0, 8.0])
def test_search_uses_reestimated_values_for_deep_transits(extractor, monkeypatch, depth, hours):
    lc, in_tr, phase, dur = make_curve(depth, hours)
    lc_flat, lc_restore = prepared(extractor, lc)
    grid, ep_s, dur_s = search_like(extractor, dur)
    stub_search_stage(extractor, monkeypatch, PERIOD, ep_s, dur_s)
    res = run_search(extractor, lc_flat, lc_restore)
    assert len(res['features']) == 1
    feat = res['features'][0]
    assert res['reestimated'] == [1]
    assert res['restored_points'] == [int(np.sum(in_tr))]
    new_dur = feat['bls_duration_d']
    new_epoch = feat['bls_epoch']
    assert abs(new_dur / dur - 1.0) < 0.15
    assert abs(feat['bls_depth_ppm'] / (depth * 1e6) - 1.0) < 0.10
    assert abs(phase_offset(new_epoch, T0, PERIOD)) < 0.15 * dur
    assert abs(new_epoch - ep_s) < 0.5 * PERIOD
    assert new_dur != dur_s
    assert feat['bls_period_d'] == PERIOD
    assert feat['bls_depth_snr'] == STUB_SNR
    assert feat['bls_sde'] == STUB_SDE
    assert feat['bls_power_peak'] == pytest.approx(1.0 + STUB_POWER, rel=0.02)
    assert feat['n_transits_observed'] == extractor.count_observed_transits(times_of(lc), PERIOD, new_epoch, new_dur, CADENCE_D)
    assert feat['n_transits_observed'] >= 20
    assert feat['duty_cycle'] == new_dur / PERIOD
    assert feat['dur_at_grid_edge'] == extractor.duration_at_grid_edge(new_dur, grid, PERIOD)
    assert same_value(feat['duration_ratio'], extractor.duration_ratio(new_dur, PERIOD, 1.0, 4.4))
    assert same_value(feat['implied_radius_rjup'], extractor.implied_radius_rjup(1.0, feat['bls_depth_ppm']))
    assert feat['in_transit_scatter_ppm'] < 0.15 * depth * 1e6
    assert feat['shape_v_u_ratio'] == pytest.approx(1.0, abs=0.02)
    mask_search = np.asarray(lc_flat.create_transit_mask(period=PERIOD, transit_time=ep_s, duration=dur_s * 1.5), dtype=bool)
    mask_new = np.asarray(lc_flat.create_transit_mask(period=PERIOD, transit_time=new_epoch, duration=new_dur * 1.5), dtype=bool)
    assert int(np.sum(mask_search)) > int(np.sum(mask_new))
    assert np.array_equal(times_of(res['lc_residual']), times_of(lc_flat)[~mask_search])


def test_search_keeps_search_values_when_local_bls_fails(extractor, monkeypatch):
    lc, in_tr, phase, dur = make_curve(0.05, 4.0)
    lc_flat, lc_restore = prepared(extractor, lc)
    grid, ep_s, dur_s = search_like(extractor, dur)
    stub_search_stage(extractor, monkeypatch, PERIOD, ep_s, dur_s)
    calls = []

    def failing(lc_cand, period, dur_grid):
        calls.append((len(lc_cand), float(period)))
        return None

    monkeypatch.setattr(extractor, 'reestimate_transit', failing)
    res = run_search(extractor, lc_flat, lc_restore)
    assert calls == [(len(lc_flat) + int(np.sum(in_tr)), PERIOD)]
    feat = res['features'][0]
    assert res['reestimated'] == [0]
    assert feat['bls_epoch'] == ep_s
    assert feat['bls_duration_d'] == dur_s
    lc_cand, n_restored = extractor.restore_transit_points(lc_flat, lc_restore, PERIOD, ep_s, dur_s)
    assert feat['bls_depth_ppm'] == extractor.box_depth_ppm(lc_cand, PERIOD, ep_s, dur_s)


def test_reestimate_transit_returns_none_without_usable_result(extractor):
    grid = extractor.duration_grid(False)
    t = np.arange(0.0, 60.0, CADENCE_D)
    flat = lk.LightCurve(time=t, flux=np.ones(len(t)), flux_err=np.full(len(t), 1e-4))
    assert extractor.reestimate_transit(flat, 0.0, grid) is None
    assert extractor.reestimate_transit(flat, 0.1, grid) is None
    assert extractor.reestimate_transit(flat[:2], 5.0, grid) is None
    rng = np.random.default_rng(3)
    bump = lk.LightCurve(time=t, flux=1.0 + rng.normal(0.0, 1e-5, len(t)) + 1e-2 * (np.abs(((t - 1.0 + 2.5) % 5.0) - 2.5) < 0.1), flux_err=np.full(len(t), 1e-4))
    out = extractor.reestimate_transit(bump, 5.0, grid)
    assert out is None or out['depth'] > 0
    nan_lc = lk.LightCurve(time=t, flux=np.full(len(t), np.nan), flux_err=np.full(len(t), 1e-4))
    assert extractor.reestimate_transit(nan_lc, 5.0, grid) is None


def test_low_restored_fraction_keeps_search_duration_and_epoch_bitwise(extractor, monkeypatch, tmp_path):
    refined = spy_search_values(extractor, monkeypatch)
    boxed = []
    reest = []
    orig_box = extractor.box_depth_ppm
    orig_reest = extractor.reestimate_transit

    def spy_box(*a, **k):
        d = orig_box(*a, **k)
        boxed.append(d)
        return d

    def spy_reest(*a, **k):
        reest.append(1)
        return orig_reest(*a, **k)

    monkeypatch.setattr(extractor, 'box_depth_ppm', spy_box)
    monkeypatch.setattr(extractor, 'reestimate_transit', spy_reest)
    coll = star_collection(sigma=1e-3, deep=0.0, box_depth=1.5e-3, box_period=4.0, box_t0=0.9, box_low_runs=8, seed=11)
    feats, curves, timings = run_star(extractor, monkeypatch, tmp_path, coll)
    sig = feats[0]
    assert abs(sig['bls_period_d'] / 4.0 - 1.0) < 0.01
    assert timings['restored_points'][0] == 32
    assert timings['restore_candidates'][0] == 32
    assert timings['reestimated'][0] == 0
    assert timings['n_reestimated'] == 0
    assert reest == []
    assert sig['bls_epoch'] == refined[0]['epoch']
    assert sig['bls_duration_d'] == refined[0]['duration']
    assert sig['bls_period_d'] == refined[0]['period']
    assert sig['bls_depth_ppm'] == boxed[0]
    assert sig['bls_depth_ppm'] != refined[0]['depth'] * 1e6
    assert sig['duty_cycle'] == refined[0]['duration'] / refined[0]['period']


def test_deep_clipped_transit_is_reestimated_end_to_end(extractor, monkeypatch, tmp_path):
    refined = spy_search_values(extractor, monkeypatch)
    feats, curves, timings = run_star(extractor, monkeypatch, tmp_path, star_collection())
    sig = near(feats, STAR_PERIOD)[0]
    assert feats[0] is sig
    assert timings['reestimated'][0] == 1
    assert timings['n_reestimated'] == sum(timings['reestimated']) >= 1
    assert len(timings['reestimated']) == len(timings['restored_points']) == len(feats)
    assert sig['bls_period_d'] == refined[0]['period']
    assert sig['bls_depth_snr'] == pytest.approx(refined[0]['depth_snr'])
    assert sig['bls_duration_d'] < refined[0]['duration']
    assert abs(sig['bls_epoch'] - refined[0]['epoch']) < 0.5 * STAR_PERIOD
    assert abs(phase_offset(sig['bls_epoch'], 0.9, STAR_PERIOD)) < 0.05
    assert sig['n_transits_observed'] == STAR_TRANSITS
    assert sig['duty_cycle'] == sig['bls_duration_d'] / sig['bls_period_d']
    assert sig['bls_depth_ppm'] > 6000.0

    spy_calls = []
    monkeypatch.setattr(extractor, 'reestimate_transit', lambda *a, **k: spy_calls.append(1))
    feats_off, curves_off, timings_off = run_star(extractor, monkeypatch, tmp_path, star_collection(), restore_transits=False)
    assert spy_calls == []
    assert timings_off['n_reestimated'] == 0
    assert all(x == 0 for x in timings_off['reestimated'])
    assert near(feats_off, STAR_PERIOD) == []


def test_transit_edge_point_cut_off_by_upward_outlier_is_treated_as_single(extractor):
    lc, in_tr, phase, dur = make_curve(0.01, 8.0, up_idx=window_points([(2, [1])]))
    t_all = times_of(lc)
    k = np.round((t_all - T0) / PERIOD).astype(int)
    edge_t = t_all[in_tr & (k == 2)][0]
    kept, removed = extractor.clip_outliers_with_removed(lc)
    assert float(edge_t) not in set(times_of(kept).tolist())
    assert float(edge_t) not in set(times_of(removed).tolist())
    assert len(removed) == int(np.sum(in_tr)) - 2


def test_diag_restore_fields(extractor):
    assert extractor.diag_restore_fields(None) == {'restore_pool_points': 0, 'restored_points': '', 'restore_candidates': ''}
    assert extractor.diag_restore_fields({}) == {'restore_pool_points': 0, 'restored_points': '', 'restore_candidates': ''}
    out = extractor.diag_restore_fields({'restore_pool_points': np.int64(2796), 'restored_points': [2795, 0, np.int64(4)], 'restore_candidates': [2795, np.int64(8), 4]})
    assert out == {'restore_pool_points': 2796, 'restored_points': '2795;0;4', 'restore_candidates': '2795;8;4'}
    assert list(out) == ['restore_pool_points', 'restored_points', 'restore_candidates']
    old = extractor.diag_restore_fields({'restore_pool_points': 5, 'restored_points': [5]})
    assert old == {'restore_pool_points': 5, 'restored_points': '5', 'restore_candidates': ''}
    row = extractor.failed_diag_row('1', 'e', 1.0, 'v', True)
    assert list(row)[-5:] == ['restore_pool_points', 'restored_points', 'restore_candidates', 'inverted_control', 'inverted_count']
    assert list(row)[-6] == 'alias_factors'
    assert row['restore_pool_points'] == 0 and row['restored_points'] == '' and row['restore_candidates'] == ''


def test_diag_rows_in_main_put_restore_fields_before_inverted_fields(extractor):
    from pathlib import Path
    src = Path(extractor.__file__).read_text(encoding='utf-8')
    marker = "diag_row.update(diag_restore_fields(timings))"
    assert src.count(marker) == 2
    for chunk in src.split(marker)[1:]:
        assert chunk.lstrip().startswith("diag_row.update(diag_inverted_fields(")
    assert src.count("'n_reestimated': int(sum(main_res['reestimated']))") == 1
    feat_block = src[src.index("        feat = {"):src.index("        features_list.append(feat)")]
    assert src.count("'restore_candidates': list(main_res['restore_candidates'])") == 1
    for name in ('restore_pool_points', 'restored_points', 'restore_candidates', 'reestimated'):
        assert name not in feat_block
