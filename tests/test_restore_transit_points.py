import astropy.units as u
import lightkurve as lk
import numpy as np
import pytest

import onyx_features

CADENCE_D = 29.4 / 1440.0
PERIOD = 13.7
T0 = 5.0
SIGMA = 2e-4
GAP_TRANSIT = 3


def window_length(flatten_days=3.0):
    wl = int(flatten_days / CADENCE_D)
    if wl % 2 == 0:
        wl -= 1
    return wl


def make_curve(depth, hours, seed=4, n=16000, up_idx=None, low_idx=None):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * CADENCE_D
    dur = hours / 24.0
    phase = ((t - T0 + 0.5 * PERIOD) % PERIOD) - 0.5 * PERIOD
    in_tr = np.abs(phase) < dur / 2.0
    rel = 1.0 + rng.normal(0.0, SIGMA, n) - depth * in_tr
    if up_idx is not None:
        rel[up_idx(in_tr, phase, dur)] = 1.0 + 30 * SIGMA
    if low_idx is not None:
        rel[low_idx(in_tr, phase, dur)] = 1.0 - 20 * SIGMA
    slow = 1.0 + 5e-4 * np.sin(2.0 * np.pi * t / 40.0)
    f = slow * rel
    keep = np.abs(t - (T0 + GAP_TRANSIT * PERIOD)) > 1.0
    t, f, in_tr, phase = t[keep], f[keep], in_tr[keep], phase[keep]
    lc = lk.LightCurve(time=t, flux=f, flux_err=np.full(len(t), SIGMA))
    lc['centroid_col_detrended'] = u.Quantity(rng.normal(0.0, 1e-3, len(t)), u.pixel)
    lc['centroid_row_detrended'] = u.Quantity(rng.normal(0.0, 1e-3, len(t)), u.pixel)
    return lc, in_tr, phase, dur


def times_of(lc):
    return np.asarray(lc.time.value, dtype=float)


def present_transits(t, dur):
    need = 0.5 * dur / CADENCE_D
    n = 0
    for k in range(-1, int((np.max(t) - T0) / PERIOD) + 2):
        cnt = int(np.sum(np.abs(t - (T0 + k * PERIOD)) < dur / 2.0))
        if cnt > 0 and cnt >= need:
            n += 1
    return n


def window_points(k_positions):
    def pick(m, ph, d):
        idx = np.where(m)[0]
        k = np.round((idx * CADENCE_D - T0) / PERIOD).astype(int)
        out = []
        for k_tr, positions in k_positions:
            sel = idx[k == k_tr]
            out.extend(int(sel[j]) for j in positions)
        return np.array(out, dtype=int)
    return pick


def restored_candidate(extractor, lc, dur):
    kept, removed = extractor.clip_outliers_with_removed(lc)
    lc_flat, lc_restore = extractor.flatten_with_restore(kept, removed, window_length())
    lc_cand, n_restored = extractor.restore_transit_points(lc_flat, lc_restore, PERIOD, T0, dur)
    return kept, removed, lc_flat, lc_restore, lc_cand, n_restored


def test_clip_with_removed_default_matches_symmetric_and_returns_low_points(extractor):
    lc, in_tr, phase, dur = make_curve(0.01, 4.0, up_idx=lambda m, ph, d: np.where(~m)[0][500:505])
    kept, removed = extractor.clip_outliers_with_removed(lc)
    ref, mask = lc.remove_outliers(sigma=5, return_mask=True)
    assert np.array_equal(np.asarray(kept.flux.value, dtype=float), np.asarray(ref.flux.value, dtype=float))
    f = np.asarray(lc.flux.value, dtype=float)
    med = np.nanmedian(f)
    assert len(removed) == int(np.sum(mask & (f < med)))
    assert len(removed) > 0
    assert np.all(np.asarray(removed.flux.value, dtype=float) < med)
    assert int(np.sum(mask & (f > med))) >= 5
    assert removed.colnames == lc.colnames


@pytest.mark.parametrize("kwargs", [{}, {"upward_only": True}, {"isolated": True}])
def test_clip_outliers_is_thin_wrapper(extractor, kwargs):
    lc, in_tr, phase, dur = make_curve(0.01, 4.0, low_idx=lambda m, ph, d: np.where(~m)[0][[700, 720, 740]])
    kept, removed = extractor.clip_outliers_with_removed(lc, **kwargs)
    out = extractor.clip_outliers(lc, **kwargs)
    assert np.array_equal(times_of(out), times_of(kept))
    assert np.array_equal(np.asarray(out.flux.value, dtype=float), np.asarray(kept.flux.value, dtype=float))
    f = np.asarray(lc.flux.value, dtype=float)
    single_t = times_of(lc)[~in_tr & (f < 1.0 - 12 * SIGMA)]
    assert len(single_t) == 3
    kept_t = set(times_of(kept).tolist())
    removed_t = set(times_of(removed).tolist())
    assert not any(float(x) in removed_t for x in single_t)
    if len(removed) > 0:
        assert np.all(np.asarray(removed.flux.value, dtype=float) < np.nanmedian(f))
    if kwargs.get("upward_only"):
        assert len(removed) == 0
        assert all(float(x) in kept_t for x in single_t)
    elif kwargs.get("isolated"):
        assert len(removed) == 0
        assert not any(float(x) in kept_t for x in single_t)
    else:
        assert len(removed) > 0
        assert not any(float(x) in kept_t for x in single_t)


@pytest.mark.parametrize("depth", [0.01, 0.05, 0.20])
@pytest.mark.parametrize("hours", [2.0, 4.0, 8.0])
def test_restore_recovers_depth_and_transit_count(extractor, depth, hours):
    lc, in_tr, phase, dur = make_curve(depth, hours)
    kept, removed, lc_flat, lc_restore, lc_cand, n_restored = restored_candidate(extractor, lc, dur)
    t_all = times_of(lc)
    removed_t = set(times_of(removed).tolist())
    n_tr_points = int(np.sum(in_tr))
    n_tr_removed = sum(1 for x in t_all[in_tr] if float(x) in removed_t)
    assert n_tr_removed > 0.5 * n_tr_points
    n_present = present_transits(t_all, dur)
    assert n_present == present_transits(t_all, dur * 0.999)
    assert n_present >= 20
    assert extractor.count_observed_transits(times_of(lc_flat), PERIOD, T0, dur, CADENCE_D) < n_present
    clipped_depth = extractor.box_depth_ppm(lc_flat, PERIOD, T0, dur)
    assert not (np.isfinite(clipped_depth) and abs(clipped_depth / (depth * 1e6) - 1.0) < 0.10)
    assert n_restored == n_tr_removed
    assert len(lc_cand) == len(lc_flat) + n_restored
    t_cand = times_of(lc_cand)
    assert np.all(np.diff(t_cand) > 0)
    assert lc_cand.colnames == lc_flat.colnames
    restored_depth = extractor.box_depth_ppm(lc_cand, PERIOD, T0, dur)
    assert abs(restored_depth / (depth * 1e6) - 1.0) < 0.10
    assert extractor.count_observed_transits(t_cand, PERIOD, T0, dur, CADENCE_D) == n_present


def test_upward_points_inside_windows_are_not_restored(extractor):
    lc, in_tr, phase, dur = make_curve(0.01, 8.0, up_idx=window_points([(0, [5]), (2, [8]), (4, [7]), (7, [10])]))
    t_all = times_of(lc)
    f_all = np.asarray(lc.flux.value, dtype=float)
    up_t = t_all[in_tr & (f_all > 1.0 + 20 * SIGMA)]
    assert len(up_t) == 4
    kept, removed, lc_flat, lc_restore, lc_cand, n_restored = restored_candidate(extractor, lc, dur)
    for pool in (kept, removed, lc_restore, lc_cand):
        pool_t = set(times_of(pool).tolist())
        assert not any(float(x) in pool_t for x in up_t)
    assert np.all(np.asarray(lc_restore.flux.value, dtype=float) < 1.0)
    cand_t = set(times_of(lc_cand).tolist())
    low_tr = t_all[in_tr & (f_all < 1.0 - 0.005)]
    assert all(float(x) in cand_t for x in low_tr)
    assert n_restored == len(low_tr)


def test_single_low_outliers_outside_windows_are_not_in_pool(extractor):
    lc, in_tr, phase, dur = make_curve(0.01, 4.0, low_idx=lambda m, ph, d: np.where(np.abs(ph) > 2.0)[0][[100, 2500, 6000, 9000]])
    t_all = times_of(lc)
    f_all = np.asarray(lc.flux.value, dtype=float)
    out_low = t_all[(np.abs(phase) > 2.0) & (f_all < 1.0 - 12 * SIGMA)]
    assert len(out_low) == 4
    kept, removed, lc_flat, lc_restore, lc_cand, n_restored = restored_candidate(extractor, lc, dur)
    for pool in (kept, removed, lc_restore, lc_cand):
        pool_t = set(times_of(pool).tolist())
        assert not any(float(x) in pool_t for x in out_low)
    assert len(lc_restore) == int(np.sum(in_tr))
    assert n_restored == int(np.sum(in_tr))


def test_single_low_outliers_inside_windows_are_not_restored_but_runs_are(extractor):
    singles = [(0, [3]), (1, [9]), (5, [6]), (8, [12])]
    runs = [(6, [4, 5]), (10, [7, 8, 9])]
    lc, in_tr, phase, dur = make_curve(3e-4, 8.0, low_idx=window_points(singles + runs))
    t_all = times_of(lc)
    f_all = np.asarray(lc.flux.value, dtype=float)
    low_t = t_all[in_tr & (f_all < 1.0 - 12 * SIGMA)]
    assert len(low_t) == 9
    k_low = np.round((low_t - T0) / PERIOD).astype(int)
    single_t = low_t[np.isin(k_low, [k for k, _ in singles])]
    run_t = low_t[np.isin(k_low, [k for k, _ in runs])]
    assert len(single_t) == 4 and len(run_t) == 5
    assert run_t[1] - run_t[0] < 1.5 * CADENCE_D
    assert run_t[4] - run_t[2] < 2.5 * CADENCE_D
    kept, removed, lc_flat, lc_restore, lc_cand, n_restored = restored_candidate(extractor, lc, dur)
    kept_t = set(times_of(kept).tolist())
    assert not any(float(x) in kept_t for x in low_t)
    pool_t = set(times_of(lc_restore).tolist())
    cand_t = set(times_of(lc_cand).tolist())
    assert not any(float(x) in pool_t for x in single_t)
    assert not any(float(x) in cand_t for x in single_t)
    assert all(float(x) in pool_t for x in run_t)
    assert all(float(x) in cand_t for x in run_t)
    assert len(lc_restore) == 5
    assert n_restored == 5
    assert len(lc_cand) == len(lc_flat) + 5


def test_pool_is_removed_low_points_that_isolated_rule_would_keep(extractor):
    def low_idx(m, ph, d):
        free = np.where(np.abs(ph) > 2.0)[0]
        return np.concatenate([free[[100, 2500, 6000]], free[[4000, 4001]]])

    lc, in_tr, phase, dur = make_curve(0.05, 4.0, low_idx=low_idx)
    kept, removed = extractor.clip_outliers_with_removed(lc)
    t_all = times_of(lc)
    f_all = np.asarray(lc.flux.value, dtype=float)
    ref, drop = lc.remove_outliers(sigma=5, return_mask=True)
    drop = np.asarray(drop, dtype=bool)
    level, sigma = extractor.robust_level_sigma(f_all)
    iso = extractor.isolated_low_mask(t_all, f_all, level, sigma)
    assert int(np.sum(drop & (f_all < level) & iso)) == 3
    assert np.array_equal(times_of(removed), t_all[drop & (f_all < level) & ~iso])
    assert len(removed) == int(np.sum(in_tr)) + 2
    assert np.array_equal(times_of(kept), times_of(ref))
    assert np.array_equal(iso, (f_all < level - extractor.CLIP_SIGMA * sigma) & ~extractor.low_neighbor_mask(t_all, f_all, level, sigma))
    kept_iso, removed_iso = extractor.clip_outliers_with_removed(lc, isolated=True)
    assert len(removed_iso) == 0
    kept_up, removed_up = extractor.clip_outliers_with_removed(lc, upward_only=True)
    assert len(removed_up) == 0


def test_restore_window_uses_factor(extractor):
    assert extractor.RESTORE_WINDOW_FACTOR == 1.25
    t = np.arange(0.0, 40.0, 0.02)
    lc_search = lk.LightCurve(time=t, flux=np.ones(len(t)), flux_err=np.full(len(t), 1e-4))
    period, epoch, dur = 10.0, 5.0, 0.4
    offsets = np.array([0.0, 0.19, -0.21, 0.245, -0.255, 0.3, 5.0])
    t_r = 15.0 + offsets + 0.001
    lc_restore = lk.LightCurve(time=t_r, flux=np.full(len(t_r), 0.99), flux_err=np.full(len(t_r), 1e-4))
    lc_cand, n_restored = extractor.restore_transit_points(lc_search, lc_restore, period, epoch, dur)
    assert n_restored == 4
    cand_t = times_of(lc_cand)
    for off, expected in zip(offsets, [True, True, True, True, False, False, False]):
        present = bool(np.any(np.isclose(cand_t, 15.0 + off + 0.001, atol=1e-9)))
        assert present == expected
    assert np.all(np.diff(cand_t) >= 0)
    assert len(lc_search) == len(t)
    assert len(lc_restore) == len(t_r)


def test_restore_noop_cases_return_search_curve(extractor):
    t = np.arange(0.0, 40.0, 0.02)
    lc_search = lk.LightCurve(time=t, flux=np.ones(len(t)), flux_err=np.full(len(t), 1e-4))
    out, n = extractor.restore_transit_points(lc_search, None, 10.0, 5.0, 0.4)
    assert out is lc_search and n == 0
    empty = lc_search[np.zeros(len(t), dtype=bool)]
    out, n = extractor.restore_transit_points(lc_search, empty, 10.0, 5.0, 0.4)
    assert out is lc_search and n == 0
    far = lk.LightCurve(time=[12.5001, 17.5001], flux=[0.99, 0.99], flux_err=[1e-4, 1e-4])
    out, n = extractor.restore_transit_points(lc_search, far, 10.0, 5.0, 0.4)
    assert out is lc_search and n == 0


def test_drop_transit_points(extractor):
    assert extractor.drop_transit_points(None, 10.0, 5.0, 0.4) is None
    pool = lk.LightCurve(time=[5.01, 7.0, 15.1, 15.4, 25.0], flux=[0.99] * 5, flux_err=[1e-4] * 5)
    out = extractor.drop_transit_points(pool, 10.0, 5.0, 0.4)
    assert np.allclose(times_of(out), [7.0, 15.4])
    empty = pool[np.zeros(5, dtype=bool)]
    assert len(extractor.drop_transit_points(empty, 10.0, 5.0, 0.4)) == 0


def test_flatten_with_restore_divides_removed_points_by_interpolated_trend(extractor):
    lc, in_tr, phase, dur = make_curve(0.05, 4.0)
    kept, removed = extractor.clip_outliers_with_removed(lc)
    wl = window_length()
    ref_flat, ref_trend = kept.flatten(window_length=wl, polyorder=2, break_tolerance=5, return_trend=True)
    lc_flat, lc_restore = extractor.flatten_with_restore(kept, removed, wl)
    assert np.array_equal(np.asarray(lc_flat.flux.value, dtype=float), np.asarray(ref_flat.flux.value, dtype=float))
    assert len(lc_restore) == len(removed)
    trend = np.interp(times_of(removed), times_of(kept), np.asarray(ref_trend.flux.value, dtype=float))
    assert np.allclose(np.asarray(lc_restore.flux.value, dtype=float), np.asarray(removed.flux.value, dtype=float) / trend)
    assert np.allclose(np.asarray(lc_restore.flux_err.value, dtype=float), np.asarray(removed.flux_err.value, dtype=float) / trend)
    assert lc_restore.colnames == lc_flat.colnames
    assert lc_restore.flux.unit == lc_flat.flux.unit
    for col in ('centroid_col_detrended', 'centroid_row_detrended'):
        assert np.array_equal(np.asarray(lc_restore[col].value, dtype=float), np.asarray(removed[col].value, dtype=float))
    assert abs(float(np.median(np.asarray(lc_restore.flux.value, dtype=float))) - 0.95) < 0.002


def test_flatten_with_restore_without_removed_is_plain_flatten(extractor):
    lc, in_tr, phase, dur = make_curve(0.05, 4.0)
    kept, removed = extractor.clip_outliers_with_removed(lc)
    wl = window_length()
    ref = kept.flatten(window_length=wl, polyorder=2, break_tolerance=5)
    lc_flat, lc_restore = extractor.flatten_with_restore(kept, None, wl)
    assert lc_restore is None
    assert np.array_equal(np.asarray(lc_flat.flux.value, dtype=float), np.asarray(ref.flux.value, dtype=float))
    assert np.array_equal(np.asarray(lc_flat.flux_err.value, dtype=float), np.asarray(ref.flux_err.value, dtype=float))
    empty = removed[np.zeros(len(removed), dtype=bool)]
    lc_flat2, lc_restore2 = extractor.flatten_with_restore(kept, empty, wl)
    assert len(lc_restore2) == 0
    assert np.array_equal(np.asarray(lc_flat2.flux.value, dtype=float), np.asarray(ref.flux.value, dtype=float))


class FakeTable:
    colnames = []


class FakeSearch:
    def __init__(self, coll):
        self.coll = coll
        self.table = FakeTable()

    def __len__(self):
        return len(self.coll)

    def download_all(self, download_dir=None):
        return self.coll


STAR_PERIOD = 4.0
STAR_T0 = 0.9
STAR_SEGMENTS = [(0.0, 44.0), (46.0, 90.0)]
STAR_TRANSITS = 22


def star_collection(sigma=3e-4, deep=1e-2, deep_dur=0.3, box_depth=0.0, box_period=5.3, box_t0=2.0, box_dur=0.2, n_low=0, low_run=1, box_low_runs=0, seed=7):
    rng = np.random.default_rng(seed)
    segs = []
    for q, (a, b) in enumerate(STAR_SEGMENTS, start=1):
        t = np.arange(a, b, 0.0204)
        f = 1.0 + rng.normal(0.0, sigma, len(t))
        ph = ((t - STAR_T0 + 0.5 * STAR_PERIOD) % STAR_PERIOD) - 0.5 * STAR_PERIOD
        m = np.abs(ph) < deep_dur / 2.0
        if deep > 0:
            f[m] -= deep * np.cos(np.pi * ph[m] / deep_dur) ** 2
        phb = ((t - box_t0 + 0.5 * box_period) % box_period) - 0.5 * box_period
        mb = np.abs(phb) < box_dur / 2.0
        if box_depth > 0:
            f[mb] -= box_depth
        if box_low_runs > 0:
            in_box = np.where(mb)[0]
            starts = in_box[np.linspace(3, len(in_box) - 5, box_low_runs).astype(int)]
            for j in range(2):
                f[starts + j] = 1.0 - 12 * sigma
        if n_low > 0:
            free = np.where((np.abs(ph) > 1.0) & (np.abs(phb) > 1.0))[0]
            spots = free[np.linspace(50, len(free) - 50, n_low).astype(int)]
            for j in range(low_run):
                f[spots + j] = 1.0 - 12 * sigma
        lc = lk.LightCurve(time=t, flux=f, flux_err=np.full(len(t), sigma))
        lc.meta['QUARTER'] = q
        segs.append(lc)
    return lk.LightCurveCollection(segs)


def run_star(extractor, monkeypatch, tmp_path, coll, **kwargs):
    monkeypatch.setattr(extractor.lk, 'search_lightcurve', lambda *a, **k: FakeSearch(coll))
    monkeypatch.setattr(extractor, '_dl_sem', None)
    out = extractor.process_single_star('2468013', 'Kepler', [], False, 0.0, 3.0, 4, quick_mode=False, cache_dir=str(tmp_path), **kwargs)
    feats, curves, tid, err, dl_size, timings = out
    assert err is None, err
    return feats, curves, timings


def near(feats, period):
    return [x for x in feats if abs(x['bls_period_d'] / period - 1.0) < 0.01]


def same_value(a, b):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def test_search_accepts_deep_clipped_transit_only_with_restore(extractor, monkeypatch, tmp_path):
    refined = []
    orig_refine = extractor.refine_peak

    def spy_refine(*a, **k):
        r = orig_refine(*a, **k)
        refined.append(dict(r))
        return r

    monkeypatch.setattr(extractor, 'refine_peak', spy_refine)
    feats, curves, timings = run_star(extractor, monkeypatch, tmp_path, star_collection())
    hit = near(feats, STAR_PERIOD)
    assert len(hit) == 1
    assert feats[0] is hit[0]
    sig = hit[0]
    assert sig['n_transits_observed'] == STAR_TRANSITS
    search_depth_ppm = refined[0]['depth'] * 1e6
    assert search_depth_ppm < 1500.0
    assert sig['bls_depth_ppm'] > 4500.0
    assert sig['bls_depth_ppm'] > 3.0 * search_depth_ppm
    assert sig['bls_depth_snr'] == pytest.approx(refined[0]['depth_snr'])
    assert 1.0 - float(np.min(curves[sig['signal_id']][0])) > 0.004
    assert timings['restore_pool_points'] > 100
    assert timings['restored_points'][0] > 100
    assert timings['n_few_transits'] == 0

    feats_off, curves_off, timings_off = run_star(extractor, monkeypatch, tmp_path, star_collection(), restore_transits=False)
    assert near(feats_off, STAR_PERIOD) == []
    assert timings_off['n_few_transits'] >= 1
    assert timings_off['restore_pool_points'] == 0
    assert all(n == 0 for n in timings_off['restored_points'])


def test_no_restore_flag_runs_previous_search_path(extractor, monkeypatch, tmp_path):
    calls = []
    orig = extractor.search_signals

    def spy(lc, *a, **k):
        res = orig(lc, *a, **k)
        calls.append({'inverted': k.get('inverted', False), 'lc_restore': k.get('lc_restore', 'absent'), 'lc': lc.copy(), 'args': a, 'kwargs': dict(k), 'res': res})
        return res

    monkeypatch.setattr(extractor, 'search_signals', spy)
    coll = star_collection(box_depth=1e-3)
    feats_off, curves_off, timings_off = run_star(extractor, monkeypatch, tmp_path, coll, restore_transits=False, inverted_control=True)
    assert [c['inverted'] for c in calls] == [False, True]
    assert calls[0]['lc_restore'] is None
    assert calls[1]['lc_restore'] is None
    plain_kwargs = {k: v for k, v in calls[0]['kwargs'].items() if k != 'lc_restore'}
    plain = orig(calls[0]['lc'], *calls[0]['args'], **plain_kwargs)
    assert len(plain['features']) == len(feats_off) >= 1
    for a, b in zip(plain['features'], feats_off):
        assert set(a) == set(b)
        for key in a:
            assert same_value(a[key], b[key]), key
    for key in curves_off:
        assert np.array_equal(plain['curves'][key], curves_off[key])

    calls.clear()
    monkeypatch.setattr(extractor, 'RESTORE_TRANSIT_POINTS', False)
    run_star(extractor, monkeypatch, tmp_path, coll)
    assert calls[0]['lc_restore'] is None

    calls.clear()
    monkeypatch.setattr(extractor, 'RESTORE_TRANSIT_POINTS', True)
    feats_on, curves_on, timings_on = run_star(extractor, monkeypatch, tmp_path, coll, inverted_control=True)
    assert [c['inverted'] for c in calls] == [False, True]
    assert calls[0]['lc_restore'] is not None and len(calls[0]['lc_restore']) > 100
    assert calls[1]['lc_restore'] is None
    assert len(near(feats_on, STAR_PERIOD)) == 1
    assert near(feats_off, STAR_PERIOD) == []


@pytest.mark.parametrize("low_run", [1, 2])
def test_restore_changes_nothing_when_no_removed_points_in_windows(extractor, monkeypatch, tmp_path, low_run):
    coll = star_collection(sigma=1e-3, deep=0.0, box_depth=1.5e-3, box_period=4.0, box_t0=0.9, n_low=4, low_run=low_run, seed=11)
    feats_on, curves_on, timings_on = run_star(extractor, monkeypatch, tmp_path, coll)
    feats_off, curves_off, timings_off = run_star(extractor, monkeypatch, tmp_path, coll, restore_transits=False)
    if low_run == 1:
        assert timings_on['restore_pool_points'] == 0
    else:
        assert timings_on['restore_pool_points'] >= 16
    assert len(feats_on) == len(feats_off) >= 1
    assert all(n == 0 for n in timings_on['restored_points'])
    assert all(n == 0 for n in timings_on['restore_candidates'])
    assert len(near(feats_on, 4.0)) == 1
    for a, b in zip(feats_on, feats_off):
        assert set(a) == set(b)
        for key in a:
            assert same_value(a[key], b[key]), key
    for key in curves_off:
        assert np.array_equal(curves_on[key], curves_off[key])


def test_restore_pool_is_masked_together_with_search_curve(extractor, monkeypatch, tmp_path):
    seen = []
    orig = extractor.restore_transit_points

    def spy(lc_search, lc_restore, period, epoch, duration):
        r = orig(lc_search, lc_restore, period, epoch, duration)
        seen.append((float(period), 0 if lc_restore is None else len(lc_restore), r[1]))
        return r

    monkeypatch.setattr(extractor, 'restore_transit_points', spy)
    feats, curves, timings = run_star(extractor, monkeypatch, tmp_path, star_collection(box_depth=4.5e-4))
    assert len(seen) >= 2
    assert abs(seen[0][0] / STAR_PERIOD - 1.0) < 0.01
    assert seen[0][1] > 100 and seen[0][2] == seen[0][1]
    assert abs(seen[1][0] / 5.3 - 1.0) < 0.01
    assert seen[1][1] == 0 and seen[1][2] == 0
    assert [round(x['bls_period_d'], 1) for x in feats[:2]] == [4.0, 5.3]
    assert timings['restored_points'][:2] == [seen[0][2], 0]
    assert timings['restore_candidates'][:2] == [seen[0][2], 0]


def test_pipeline_version_is_bumped_for_restored_features():
    assert onyx_features.PIPELINE_VERSION == '8'
    assert onyx_features.build_pipeline_version(False, 3.0, 4).startswith('8|')
