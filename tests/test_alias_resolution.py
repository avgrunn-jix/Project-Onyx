import numpy as np
import pytest
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares


def box_curve(period, t0, depth, duration=0.24, step=0.04, span=40.0, sigma=1e-3, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, span, step)
    f = 1.0 + rng.normal(0.0, sigma, len(t))
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    f[np.abs(phase) < duration / 2.0] -= depth
    e = np.full(len(t), sigma)
    return t, f, e


def bls_grid(t, f, e, duration=0.24):
    periods = np.exp(np.linspace(np.log(1.0), np.log(15.0), 4000))
    res = BoxLeastSquares(t, f, dy=e).power(periods, duration)
    return periods, np.asarray(res.power, dtype=float)


def test_half_period_peak_resolved_to_true(extractor):
    t, f, e = box_curve(5.0, 1.3, 5e-3)
    periods, power = bls_grid(t, f, e)
    res = extractor.resolve_alias_period(t, f, e, 2.5, 1.3 + 2.5, 0.24, periods, power)
    assert res["factor"] == 2.0
    assert abs(res["period"] - 5.0) < 1e-9
    assert abs(((res["epoch"] - 1.3 + 2.5) % 5.0) - 2.5) < 0.05


def test_double_period_peak_resolved_to_true(extractor):
    t, f, e = box_curve(2.5, 0.7, 5e-3)
    periods, power = bls_grid(t, f, e)
    res = extractor.resolve_alias_period(t, f, e, 5.0, 0.7, 0.24, periods, power)
    assert res["factor"] == 0.5
    assert abs(res["period"] - 2.5) < 1e-9


def test_triple_period_peak_resolved_to_true(extractor):
    t, f, e = box_curve(2.0, 0.5, 5e-3)
    periods, power = bls_grid(t, f, e)
    res = extractor.resolve_alias_period(t, f, e, 6.0, 0.5, 0.24, periods, power)
    assert abs(res["period"] - 2.0) < 1e-9


def test_true_short_period_not_replaced(extractor):
    t, f, e = box_curve(2.5, 0.7, 5e-3)
    periods, power = bls_grid(t, f, e)
    res = extractor.resolve_alias_period(t, f, e, 2.5, 0.7, 0.24, periods, power)
    assert res["factor"] == 1.0
    assert res["period"] == 2.5


def test_true_long_period_not_shortened(extractor):
    t, f, e = box_curve(5.0, 1.3, 5e-3)
    periods, power = bls_grid(t, f, e)
    res = extractor.resolve_alias_period(t, f, e, 5.0, 1.3, 0.24, periods, power)
    assert res["factor"] == 1.0
    assert res["period"] == 5.0


def test_power_guard_blocks_correction(extractor):
    t, f, e = box_curve(5.0, 1.3, 5e-3)
    periods, power = bls_grid(t, f, e)
    power = power.copy()
    power[np.abs(periods / 5.0 - 1.0) <= 0.02] = 0.0
    res = extractor.resolve_alias_period(t, f, e, 2.5, 1.3, 0.24, periods, power)
    assert res["factor"] == 1.0


def test_is_alias_duplicate(extractor):
    found = [4.0]
    for p in (4.0, 8.0, 12.0, 2.0, 4.0 / 3.0, 4.02, 7.95):
        assert extractor.is_alias_duplicate(p, found)
    for p in (5.0, 6.0, 4.2, 1.0, 10.0):
        assert not extractor.is_alias_duplicate(p, found)


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


def eb_collection():
    rng = np.random.default_rng(11)
    segs = []
    period, t0, dur = 4.0, 0.9, 0.2
    for q, (a, b) in enumerate([(0.0, 44.0), (46.0, 90.0)], start=1):
        t = np.arange(a, b, 0.0204)
        f = 1.0 + rng.normal(0.0, 1e-3, len(t))
        ph = ((t - t0 + 0.5 * period) % period) - 0.5 * period
        f[np.abs(ph) < dur / 2.0] -= 8e-3
        ph2 = ((t - t0 - 0.5 * period + 0.5 * period) % period) - 0.5 * period
        f[np.abs(ph2) < dur / 2.0] -= 3e-3
        lc = lk.LightCurve(time=t, flux=f, flux_err=np.full(len(t), 1e-3))
        lc.meta["QUARTER"] = q
        segs.append(lc)
    return lk.LightCurveCollection(segs)


def run_star(extractor, monkeypatch, tmp_path):
    coll = eb_collection()
    monkeypatch.setattr(extractor.lk, "search_lightcurve", lambda *a, **k: FakeSearch(coll))
    rows = [{"catalog_period": 4.0, "label": "FALSE POSITIVE"}]
    out = extractor.process_single_star("1234567", "Kepler", rows, False, 0.0, 3.0, 4, quick_mode=False, cache_dir=str(tmp_path))
    feats, curves, tid, err, dl_size, timings = out
    assert err is None, err
    return feats, timings


def alias_pairs(periods):
    bad = []
    for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
            r = periods[j] / periods[i]
            for m in (1.0, 2.0, 3.0, 0.5, 1.0 / 3.0):
                if abs(r / m - 1.0) < 0.01:
                    bad.append((periods[i], periods[j]))
    return bad


def test_loop_has_no_alias_duplicates(extractor, monkeypatch, tmp_path):
    monkeypatch.setattr(extractor, "RESOLVE_ALIASES", True)
    orig_dup = extractor.is_alias_duplicate
    dup_calls = []

    def spy(p, found, *a, **k):
        r = orig_dup(p, found, *a, **k)
        dup_calls.append((float(p), list(found), r))
        return r

    monkeypatch.setattr(extractor, "is_alias_duplicate", spy)
    feats, timings = run_star(extractor, monkeypatch, tmp_path)
    periods = [float(x["bls_period_d"]) for x in feats]
    assert any(c[2] for c in dup_calls)
    assert len(periods) >= 1
    assert abs(periods[0] / 4.0 - 1.0) < 0.01
    assert alias_pairs(periods) == []
    assert [x["signal_index"] for x in feats] == list(range(len(feats)))
    assert "alias_corrections" in timings
    assert timings["alias_corrections"] >= 1
    assert "alias_factors" in timings
    assert isinstance(timings["alias_factors"], str) and len(timings["alias_factors"]) > 0
    factors_parts = timings["alias_factors"].split(";")
    assert len(factors_parts) == timings["alias_corrections"]
    assert "2.0" in factors_parts
    for item in feats:
        assert "alias_factors" not in item
        assert "alias_corrections" not in item
    print("RESOLVE_ALIASES=True periods:", periods, "corrections:", timings["alias_corrections"], "factors:", timings["alias_factors"], "dup checks:", dup_calls)


def test_loop_disabled_keeps_old_behaviour(extractor, monkeypatch, tmp_path):
    monkeypatch.setattr(extractor, "RESOLVE_ALIASES", False)

    def boom(*a, **k):
        raise AssertionError("resolve_alias_period must not be called")

    monkeypatch.setattr(extractor, "resolve_alias_period", boom)
    monkeypatch.setattr(extractor, "is_alias_duplicate", boom)
    feats, timings = run_star(extractor, monkeypatch, tmp_path)
    periods = [float(x["bls_period_d"]) for x in feats]
    assert timings["alias_corrections"] == 0
    assert timings["alias_factors"] == ""
    assert abs(periods[0] / 2.0 - 1.0) < 0.01
    assert all(abs(p / 4.0 - 1.0) >= 0.01 for p in periods)


def test_offline_astropy_bls_project_grid_two_pass(extractor):
    rng = np.random.default_rng(2026)
    baseline_d = 60.0
    cadence_d = 0.0204
    t = np.arange(0.0, baseline_d, cadence_d)
    sigma = 2e-4
    f = 1.0 + rng.normal(0.0, sigma, len(t))
    e = np.full(len(t), sigma)

    true_period = 3.0
    true_epoch = 1.15
    true_dur = 0.12
    depth = 1e-3

    ph1 = ((t - true_epoch + 0.5 * true_period) % true_period) - 0.5 * true_period
    f[np.abs(ph1) < true_dur / 2.0] -= depth
    ph2 = ((t - (true_epoch + 0.5 * true_period) + 0.5 * true_period) % true_period) - 0.5 * true_period
    f[np.abs(ph2) < true_dur / 2.0] -= 0.4 * depth

    dur_grid = np.linspace(0.04, 0.45, 10)
    max_p = min(100.0, baseline_d / 2.0)
    freq_factor = 500

    bls1 = BoxLeastSquares(t, f, dy=e)
    periods1 = bls1.autoperiod(dur_grid, minimum_period=0.5, maximum_period=max_p, frequency_factor=freq_factor)
    res1 = bls1.power(periods1, dur_grid)
    powers1 = np.asarray(res1.power, dtype=float)
    periods1_arr = np.asarray(res1.period, dtype=float)
    assert len(periods1_arr) == len(powers1)

    idx1 = int(np.nanargmax(powers1))
    p1_raw = float(periods1_arr[idx1])
    t0_1_raw = float(res1.transit_time[idx1])
    dur1_raw = float(res1.duration[idx1])

    alias1 = extractor.resolve_alias_period(t, f, e, p1_raw, t0_1_raw, dur1_raw, periods1_arr, powers1)
    p1 = alias1["period"]
    t0_1 = alias1["epoch"]
    assert abs(p1 - true_period) / true_period < 0.01

    found_periods = []
    assert not extractor.is_alias_duplicate(p1, found_periods)
    found_periods.append(p1)

    lc_cur = lk.LightCurve(time=t, flux=f, flux_err=e)
    mask1 = lc_cur.create_transit_mask(period=p1, transit_time=t0_1, duration=dur1_raw * 1.5)
    lc_masked = lc_cur[~mask1]
    t2 = np.asarray(lc_masked.time.value, dtype=float)
    f2 = np.asarray(lc_masked.flux.value, dtype=float)
    e2 = np.asarray(lc_masked.flux_err.value, dtype=float)

    bls2 = BoxLeastSquares(t2, f2, dy=e2)
    periods2 = bls2.autoperiod(dur_grid, minimum_period=0.5, maximum_period=max_p, frequency_factor=freq_factor)
    res2 = bls2.power(periods2, dur_grid)
    powers2 = np.asarray(res2.power, dtype=float)
    periods2_arr = np.asarray(res2.period, dtype=float)
    assert len(periods2_arr) == len(powers2)

    idx2 = int(np.nanargmax(powers2))
    p2_raw = float(periods2_arr[idx2])
    t0_2_raw = float(res2.transit_time[idx2])
    dur2_raw = float(res2.duration[idx2])

    alias2 = extractor.resolve_alias_period(t2, f2, e2, p2_raw, t0_2_raw, dur2_raw, periods2_arr, powers2)
    p2 = alias2["period"]
    assert extractor.is_alias_duplicate(p2, found_periods)

