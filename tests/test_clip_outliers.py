import numpy as np
import pytest
import lightkurve as lk


def make_lc():
    rng = np.random.default_rng(1)
    n = 400
    t = np.linspace(0.0, 10.0, n)
    sigma = 1e-3
    f = 1.0 + rng.normal(0.0, sigma, n)
    f[200] = 1.0 - 10 * sigma
    f[100] = 1.0 + 10 * sigma
    return lk.LightCurve(time=t, flux=f), sigma


def test_upward_only_keeps_dip(extractor):
    lc, sigma = make_lc()
    out = extractor.clip_outliers(lc, upward_only=True)
    flux = np.asarray(out.flux.value, dtype=float)
    assert np.any(np.isclose(flux, 1.0 - 10 * sigma))
    assert not np.any(np.isclose(flux, 1.0 + 10 * sigma))
    assert len(out) == len(lc) - 1


def test_symmetric_clips_dip(extractor):
    lc, sigma = make_lc()
    out = extractor.clip_outliers(lc, upward_only=False)
    flux = np.asarray(out.flux.value, dtype=float)
    assert not np.any(np.isclose(flux, 1.0 - 10 * sigma))
    assert not np.any(np.isclose(flux, 1.0 + 10 * sigma))


def test_default_follows_constant(extractor, monkeypatch):
    lc, sigma = make_lc()
    monkeypatch.setattr(extractor, "CLIP_UPWARD_ONLY", False)
    flux = np.asarray(extractor.clip_outliers(lc).flux.value, dtype=float)
    assert not np.any(np.isclose(flux, 1.0 - 10 * sigma))
    monkeypatch.setattr(extractor, "CLIP_UPWARD_ONLY", True)
    flux = np.asarray(extractor.clip_outliers(lc).flux.value, dtype=float)
    assert np.any(np.isclose(flux, 1.0 - 10 * sigma))
    assert extractor.CLIP_UPWARD_ONLY is True


def test_init_worker_sets_search_flags(extractor, monkeypatch):
    monkeypatch.setattr(extractor, "_dl_sem", None)
    monkeypatch.setattr(extractor, "CLIP_UPWARD_ONLY", True)
    monkeypatch.setattr(extractor, "RESOLVE_ALIASES", True)
    extractor.init_worker("dummy_sem", clip_upward=False, resolve_aliases=False)
    assert extractor.CLIP_UPWARD_ONLY is False
    assert extractor.RESOLVE_ALIASES is False
    extractor.init_worker("dummy_sem", clip_upward=True, resolve_aliases=True)
    assert extractor.CLIP_UPWARD_ONLY is True
    assert extractor.RESOLVE_ALIASES is True


def test_default_is_symmetric(extractor):
    assert extractor.CLIP_UPWARD_ONLY is False
    assert extractor.CLIP_ISOLATED is False
    lc, sigma = make_lc()
    flux = np.asarray(extractor.clip_outliers(lc).flux.value, dtype=float)
    expected = np.asarray(lc.remove_outliers(sigma=5).flux.value, dtype=float)
    assert np.array_equal(flux, expected)
    assert not np.any(np.isclose(flux, 1.0 - 10 * sigma))
    assert not np.any(np.isclose(flux, 1.0 + 10 * sigma))


def test_clip_upward_only_flag_reaches_workers(extractor, monkeypatch):
    monkeypatch.setattr(extractor, "_dl_sem", None)
    monkeypatch.setattr(extractor, "RESOLVE_ALIASES", True)
    monkeypatch.setattr(extractor, "CLIP_UPWARD_ONLY", False)
    extractor.init_worker("dummy_sem", clip_upward=True, resolve_aliases=True)
    assert extractor.CLIP_UPWARD_ONLY is True
    lc, sigma = make_lc()
    flux = np.asarray(extractor.clip_outliers(lc).flux.value, dtype=float)
    assert np.any(np.isclose(flux, 1.0 - 10 * sigma))
    assert not np.any(np.isclose(flux, 1.0 + 10 * sigma))


CADENCE_D = 29.4 / 1440.0


def make_transit_lc(depth, hours, seed=3, n=6000, period=10.0, sigma=1e-4):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * CADENCE_D
    f = 1.0 + rng.normal(0.0, sigma, n)
    phase = (t - 5.0) % period
    dur = hours / 24.0
    in_tr = phase < dur
    f[in_tr] = 1.0 - depth + rng.normal(0.0, sigma, int(in_tr.sum()))
    return lk.LightCurve(time=t, flux=f), in_tr, sigma


@pytest.mark.parametrize("depth", [0.01, 0.05, 0.20])
@pytest.mark.parametrize("hours", [2.0, 4.0, 8.0])
def test_isolated_keeps_all_transit_points(extractor, depth, hours):
    lc, in_tr, sigma = make_transit_lc(depth, hours)
    out = extractor.clip_outliers(lc, isolated=True)
    kept_t = set(np.asarray(out.time.value, dtype=float).tolist())
    tr_t = np.asarray(lc.time.value, dtype=float)[in_tr]
    assert len(tr_t) >= 4
    assert all(float(x) in kept_t for x in tr_t)


@pytest.mark.parametrize("depth", [0.01, 0.20])
def test_default_removes_transit_points(extractor, depth):
    lc, in_tr, sigma = make_transit_lc(depth, 4.0)
    out = extractor.clip_outliers(lc)
    assert len(out) < len(lc) - int(in_tr.sum()) // 2


def test_isolated_removes_single_outliers_both_sides(extractor):
    lc, in_tr, sigma = make_transit_lc(0.01, 4.0)
    f = np.asarray(lc.flux.value, dtype=float).copy()
    idx_up = 1000
    idx_dn = 3000
    assert not in_tr[idx_up] and not in_tr[idx_dn]
    f[idx_up] = 1.0 + 10 * sigma
    f[idx_dn] = 1.0 - 10 * sigma
    lc2 = lk.LightCurve(time=np.asarray(lc.time.value, dtype=float), flux=f)
    out = extractor.clip_outliers(lc2, isolated=True)
    kept = set(np.asarray(out.time.value, dtype=float).tolist())
    t = np.asarray(lc2.time.value, dtype=float)
    assert float(t[idx_up]) not in kept
    assert float(t[idx_dn]) not in kept
    assert float(t[idx_up - 1]) in kept and float(t[idx_up + 1]) in kept
    assert float(t[idx_dn - 1]) in kept and float(t[idx_dn + 1]) in kept


def test_isolated_low_point_at_transit_edge_keeps_neighbours(extractor):
    lc, in_tr, sigma = make_transit_lc(0.05, 4.0)
    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float).copy()
    first = int(np.where(in_tr)[0][0])
    f[first - 1] = 1.0 - 0.02
    lc2 = lk.LightCurve(time=t, flux=f)
    out = extractor.clip_outliers(lc2, isolated=True)
    kept = set(np.asarray(out.time.value, dtype=float).tolist())
    idx = np.where(in_tr)[0]
    assert all(float(t[i]) in kept for i in idx)
    assert float(t[first - 1]) in kept


def test_isolated_point_next_to_gap_is_removed(extractor):
    lc, in_tr, sigma = make_transit_lc(0.01, 2.0)
    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float).copy()
    f[2000] = 1.0 - 10 * sigma
    keep = np.ones(len(t), dtype=bool)
    keep[1998:2000] = False
    lc2 = lk.LightCurve(time=t[keep], flux=f[keep])
    out = extractor.clip_outliers(lc2, isolated=True)
    assert float(t[2000]) not in set(np.asarray(out.time.value, dtype=float).tolist())


def test_default_and_upward_modes_match_previous_behaviour(extractor):
    lc, in_tr, sigma = make_transit_lc(0.05, 4.0)
    default = np.asarray(extractor.clip_outliers(lc).flux.value, dtype=float)
    assert np.array_equal(default, np.asarray(lc.remove_outliers(sigma=5).flux.value, dtype=float))
    explicit = np.asarray(extractor.clip_outliers(lc, upward_only=False, isolated=False).flux.value, dtype=float)
    assert np.array_equal(explicit, default)
    up = np.asarray(extractor.clip_outliers(lc, upward_only=True).flux.value, dtype=float)
    assert np.array_equal(up, np.asarray(lc.remove_outliers(sigma_upper=5, sigma_lower=float('inf')).flux.value, dtype=float))
    assert len(up) > len(default)


def test_isolated_flag_reaches_workers(extractor, monkeypatch):
    monkeypatch.setattr(extractor, "_dl_sem", None)
    monkeypatch.setattr(extractor, "CLIP_ISOLATED", False)
    monkeypatch.setattr(extractor, "RESTORE_TRANSIT_POINTS", True)
    lc, in_tr, sigma = make_transit_lc(0.05, 4.0)
    extractor.init_worker("dummy_sem", clip_isolated=True, restore_transits=False)
    assert extractor.CLIP_ISOLATED is True
    assert extractor.RESTORE_TRANSIT_POINTS is False
    kept_t = set(np.asarray(extractor.clip_outliers(lc).time.value, dtype=float).tolist())
    assert all(float(x) in kept_t for x in np.asarray(lc.time.value, dtype=float)[in_tr])
    extractor.init_worker("dummy_sem", clip_isolated=False, restore_transits=True)
    assert extractor.CLIP_ISOLATED is False
    assert extractor.RESTORE_TRANSIT_POINTS is True
    flux = np.asarray(extractor.clip_outliers(lc).flux.value, dtype=float)
    assert np.array_equal(flux, np.asarray(lc.remove_outliers(sigma=5).flux.value, dtype=float))


def test_init_worker_keeps_flags_when_not_given(extractor, monkeypatch):
    monkeypatch.setattr(extractor, "_dl_sem", None)
    monkeypatch.setattr(extractor, "CLIP_UPWARD_ONLY", False)
    monkeypatch.setattr(extractor, "RESOLVE_ALIASES", True)
    monkeypatch.setattr(extractor, "CLIP_ISOLATED", True)
    monkeypatch.setattr(extractor, "RESTORE_TRANSIT_POINTS", False)
    extractor.init_worker("dummy_sem", clip_upward=False, resolve_aliases=True)
    assert extractor.CLIP_ISOLATED is True
    assert extractor.RESTORE_TRANSIT_POINTS is False


def test_cli_exposes_search_mode_flags(extractor):
    import inspect
    src = inspect.getsource(extractor.main)
    assert "--clip-isolated" in src
    assert "--clip-upward-only" in src
    assert "--no-restore-transit-points" in src
    assert "--clip-symmetric-legacy" not in src
    assert "CLIP_ISOLATED = args.clip_isolated" in src
    assert "RESTORE_TRANSIT_POINTS = not args.no_restore_transit_points" in src
    assert src.count("CLIP_ISOLATED, RESTORE_TRANSIT_POINTS)") >= 4


def test_mode_parameters_reach_worker_functions(extractor):
    import inspect
    for fn in (extractor.process_single_star, extractor.run_injection, extractor.init_worker):
        params = inspect.signature(fn).parameters
        assert "clip_isolated" in params
        assert "restore_transits" in params
        assert "clip_legacy" not in params
    names = list(inspect.signature(extractor.process_single_star).parameters)
    assert names[11:17] == ["clip_upward", "resolve_aliases", "inverted_control", "eb_periods", "clip_isolated", "restore_transits"]
    names = list(inspect.signature(extractor.run_injection).parameters)
    assert names[9:13] == ["clip_upward", "resolve_aliases", "clip_isolated", "restore_transits"]
    names = list(inspect.signature(extractor.init_worker).parameters)
    assert names == ["sem", "clip_upward", "resolve_aliases", "clip_isolated", "restore_transits"]
