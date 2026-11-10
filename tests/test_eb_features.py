import numpy as np
import pandas as pd
import pytest


def eclipse_series(period, epoch, duration, primary, secondary, span=200.0, sigma=1e-3, seed=0, cadence=0.0204, odd_extra=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, span, cadence)
    f = 1.0 + rng.normal(0.0, sigma, len(t))
    k = np.round((t - epoch) / period).astype(int)
    in_p = np.abs(t - (epoch + k * period)) < duration / 2.0
    f[in_p] -= primary + odd_extra * (k[in_p] % 2)
    ks = np.round((t - epoch - 0.5 * period) / period).astype(int)
    in_s = np.abs(t - (epoch + 0.5 * period + ks * period)) < duration / 2.0
    f[in_s] -= secondary
    return t, f


def test_eb_with_equal_eclipses_low_odd_even_and_significant_secondary(extractor):
    period, epoch, dur = 3.0, 0.7, 0.15
    t, f = eclipse_series(period, epoch, dur, 5e-3, 5e-3, seed=1)
    oe = extractor.odd_even_z(t, f, period, epoch, dur)
    sz, sph = extractor.secondary_search(t, f, period, epoch, dur)
    assert abs(oe) < 3.0
    assert sz > 20.0
    assert sph == pytest.approx(0.5, abs=0.02)


def test_eb_found_at_half_period_has_large_odd_even(extractor):
    period, epoch, dur = 3.0, 0.7, 0.15
    t, f = eclipse_series(period, epoch, dur, 8e-3, 3e-3, seed=2)
    oe = extractor.odd_even_z(t, f, period / 2.0, epoch, dur)
    assert oe < -10.0
    shifted = extractor.odd_even_z(t, f, period / 2.0, epoch + period / 2.0, dur)
    assert shifted > 10.0


def test_planet_without_secondary_stays_within_noise(extractor):
    period, epoch, dur = 4.0, 1.1, 0.15
    vals = []
    for seed in range(5):
        t, f = eclipse_series(period, epoch, dur, 2e-3, 0.0, seed=10 + seed)
        sz, sph = extractor.secondary_search(t, f, period, epoch, dur)
        oe = extractor.odd_even_z(t, f, period, epoch, dur)
        assert 0.2 - 1e-9 <= sph <= 0.8 + 1e-9
        assert abs(oe) < 4.0
        vals.append(sz)
    assert max(vals) < 5.0


def test_secondary_window_is_full(extractor):
    period, epoch, dur = 5.0, 0.0, 0.2
    t = np.arange(0.0, 100.0, 0.005)
    f = np.ones(len(t))
    ph = ((t - epoch) / period) % 1.0
    f[np.abs(ph - 0.5) * period < dur / 2.0] -= 1e-3
    f += np.random.default_rng(3).normal(0, 1e-5, len(t))
    sz, sph = extractor.secondary_search(t, f, period, epoch, dur)
    assert sph == pytest.approx(0.5, abs=0.25 * dur / period + 1e-9)
    win = np.abs(ph - sph) * period < dur / 2.0
    assert np.mean(1.0 - f[win]) == pytest.approx(1e-3, rel=0.05)


def test_implied_radius_and_duration_ratio(extractor):
    assert extractor.implied_radius_rjup(1.0, 10000.0) == pytest.approx(0.1 * 9.731, rel=1e-6)
    assert extractor.implied_radius_rjup(2.0, 2500.0) == pytest.approx(0.1 * 9.731, rel=1e-6)
    assert np.isnan(extractor.implied_radius_rjup(np.nan, 1000.0))
    assert np.isnan(extractor.implied_radius_rjup(1.0, -5.0))
    exp_earth = extractor.expected_duration_days(365.25, 1.0, 4.438)
    assert exp_earth == pytest.approx(0.541, abs=0.01)
    assert extractor.duration_ratio(exp_earth, 365.25, 1.0, 4.438) == pytest.approx(1.0)
    assert extractor.duration_ratio(2 * exp_earth, 365.25, 1.0, 4.438) == pytest.approx(2.0)
    exp_hj = extractor.expected_duration_days(3.0, 1.0, 4.438)
    assert exp_hj == pytest.approx(3.0 / np.pi * np.arcsin(1.0 / 8.85), rel=0.02)
    assert np.isnan(extractor.duration_ratio(0.1, 3.0, 1.0, np.nan))
    assert np.isnan(extractor.duration_ratio(0.1, 3.0, np.nan, 4.4))


def test_eb_period_match(extractor, tmp_path):
    path = tmp_path / 'eb.csv'
    pd.DataFrame({'kepid': [4852529, 1234567, 1234567], 'period_d': [2.0, 10.0, 0.7], 'BJD0': [0, 0, 0], 'Morph': [0.1, 0.2, 0.3]}).to_csv(path, index=False)
    cat = extractor.load_eb_periods(str(path))
    assert cat == {'4852529': [2.0], '1234567': [10.0, 0.7]}
    assert extractor.eb_periods_for('KIC 001234567', cat) == [10.0, 0.7]
    assert extractor.eb_periods_for('7654321', cat) == []
    assert extractor.eb_periods_for('1', None) is None
    eb = cat['1234567']
    assert extractor.eb_period_match(10.05, eb) == 1
    assert extractor.eb_period_match(5.02, eb) == 1
    assert extractor.eb_period_match(1.4, eb) == 1
    assert extractor.eb_period_match(10.2, eb) == 0
    assert extractor.eb_period_match(30.0, eb) == 0
    assert extractor.eb_period_match(3.0, []) == 0
    assert np.isnan(extractor.eb_period_match(3.0, None))
    pd.DataFrame({'kepid': [1]}).to_csv(tmp_path / 'noperiod.csv', index=False)
    assert extractor.load_eb_periods(str(tmp_path / 'noperiod.csv')) is None
    assert extractor.load_eb_periods(str(tmp_path / 'missing.csv')) is None


def test_eb_features_reach_feature_rows(extractor):
    import time
    import lightkurve as lk
    from test_search_refine import CADENCE, star_feats
    t, f = eclipse_series(3.0, 0.7, 0.15, 6e-3, 6e-3, span=150.0, seed=4)
    lc = lk.LightCurve(time=t, flux=f, flux_err=np.full(len(t), 1e-3))
    sf = star_feats()
    sf['star_radius'] = 1.0
    sf['star_logg'] = 4.4
    res = extractor.search_signals(lc, '1234567', 'Kepler', [], float(t.max() - t.min()), CADENCE, sf, True, 2, time.time() + 600, True, 'v', eb_periods=[3.0])
    assert res['features']
    top = res['features'][0]
    for col in ('odd_even_z', 'secondary_z', 'secondary_phase', 'implied_radius_rjup', 'duration_ratio', 'eb_period_match', 'odd_even_sigma', 'secondary_snr'):
        assert col in top
    assert top['eb_period_match'] == 1
    assert np.isfinite(top['implied_radius_rjup'])
    assert np.isfinite(top['duration_ratio'])
