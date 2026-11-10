import time

import numpy as np
import pytest
import lightkurve as lk


CADENCE = 0.0204


def box_series(period, epoch, duration, depth, span, sigma=1e-3, seed=0, cadence=CADENCE):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, span, cadence)
    f = 1.0 + rng.normal(0.0, sigma, len(t))
    if period is not None:
        ph = ((t - epoch + 0.5 * period) % period) - 0.5 * period
        f[np.abs(ph) < duration / 2.0] -= depth
    return t, f, np.full(len(t), sigma)


def star_feats():
    return {k: 0.0 for k in ['flux_std_ppm', 'flux_mad_ppm', 'flux_skew', 'flux_kurtosis', 'cdpp_ppm',
                             'pointwise_scatter_ppm', 'autocorr_max', 'ls_peak_power', 'gap_fraction',
                             'baseline_d', 'star_teff', 'star_radius', 'star_logg']}


@pytest.mark.parametrize('period,duration', [(0.7, 0.06), (3.0, 0.12), (12.0, 0.2), (40.0, 0.3)])
def test_refinement_recovers_true_period_on_1460_day_series(extractor, period, duration):
    span = 1460.0
    epoch = 0.37 * period + 1.1
    t, f, dy = box_series(period, epoch, duration, 1.5e-3, span, sigma=1e-3, seed=int(period * 10))
    baseline = float(t.max() - t.min())
    dur_grid = extractor.duration_grid(True)
    df = extractor.FREQ_FACTOR_QUICK * dur_grid.min() / baseline ** 2
    coarse_step = period ** 2 * df
    width = 8.0 * coarse_step
    spec = extractor.bls_spectrum(t, f, dy, dur_grid, period - width, period + width, extractor.FREQ_FACTOR_QUICK)
    idx = extractor.peak_index(spec['power'])
    assert abs(extractor.coarse_step_at(spec['period'], idx) / coarse_step - 1.0) < 0.05
    coarse = extractor.peak_values(spec, idx)
    refined = extractor.refine_peak(spec, idx, baseline)
    assert refined['power'] >= coarse['power']
    accumulated = abs(refined['period'] - period) * baseline / period
    assert accumulated < duration / 4.0, (refined['period'], coarse['period'], accumulated)
    assert np.min(np.abs(dur_grid - refined['duration'])) <= extractor.DUR_BIN_D + 1e-9


def test_refinement_keeps_coarse_values_when_power_does_not_improve(extractor):
    class FlatBLS:
        def power(self, periods, durations):
            n = len(periods)
            return type('R', (), {'power': np.zeros(n), 'period': np.asarray(periods), 'transit_time': np.zeros(n),
                                  'duration': np.full(n, durations[0]), 'depth': np.zeros(n), 'depth_snr': np.zeros(n)})()

    periods = np.linspace(5.0, 4.9, 11)
    spec = {
        'bls': FlatBLS(), 'dur_grid': extractor.duration_grid(True), 'max_duty': extractor.MAX_DUTY_CYCLE,
        'period': periods, 'power': np.linspace(1.0, 2.0, 11), 'duration': np.full(11, 0.11),
        'transit_time': np.full(11, 0.5), 'depth': np.full(11, 1e-3), 'depth_snr': np.full(11, 9.0),
    }
    out = extractor.refine_peak(spec, 10, 1000.0)
    assert out['refined'] is False
    assert out['period'] == periods[10]
    assert out['power'] == 2.0


def test_snr_taken_at_power_peak_not_spectrum_maximum(extractor):
    t, f, dy = box_series(5.0, 1.2, 0.15, 2e-3, 200.0, seed=4)
    spec = extractor.bls_spectrum(t, f, dy, extractor.duration_grid(True), 0.5, 60.0, extractor.FREQ_FACTOR_QUICK)
    idx = extractor.peak_index(spec['power'])
    spec['depth_snr'] = spec['depth_snr'].copy()
    j = (idx + len(spec['power']) // 2) % len(spec['power'])
    spec['depth_snr'][j] = 1e6
    vals = extractor.peak_values(spec, idx)
    assert vals['depth_snr'] == spec['depth_snr'][idx]
    assert vals['depth_snr'] < 1e6
    assert vals['power'] == float(np.nanmax(spec['power']))


def test_duration_grid_is_logarithmic(extractor):
    q = extractor.duration_grid(True)
    n = extractor.duration_grid(False)
    assert len(q) == 6 and len(n) == 12
    for g in (q, n):
        assert g[0] == pytest.approx(0.04) and g[-1] == pytest.approx(0.5)
        ratios = g[1:] / g[:-1]
        assert np.allclose(ratios, ratios[0])


def test_spectrum_respects_duty_cycle_limit(extractor):
    t, f, dy = box_series(None, 0, 0, 0, 120.0, seed=5)
    spec = extractor.bls_spectrum(t, f, dy, extractor.duration_grid(True), 0.5, 40.0, extractor.FREQ_FACTOR_QUICK)
    assert np.all(np.isfinite(spec['power']))
    assert np.all(spec['duration'] <= extractor.MAX_DUTY_CYCLE * spec['period'] + 1e-12)


def test_detrended_sde_removes_red_noise_trend(extractor):
    rng = np.random.default_rng(1)
    n = 20000
    x = np.arange(n)
    trend = 50.0 * np.exp(3.0 * x / n)
    power = trend + rng.normal(0, 1.0, n)
    idx = int(np.argmax(power))
    assert extractor.raw_sde(power, idx) > 2.0
    assert extractor.detrended_sde(power, idx) < 6.0
    flat = rng.normal(0, 1.0, n)
    flat[7000] = 25.0
    raw = extractor.raw_sde(flat, 7000)
    det = extractor.detrended_sde(flat, 7000)
    assert raw > 20.0 and abs(det / raw - 1.0) < 0.1


def test_n_transits_observed_with_gaps(extractor):
    period, epoch, dur = 10.0, 5.0, 0.4
    t = np.arange(0.0, 100.0, 0.02)
    keep = np.ones(len(t), dtype=bool)
    keep &= ~((t > 14.0) & (t < 26.0))
    mid4 = epoch + 4 * period
    in4 = np.where(np.abs(t - mid4) < dur / 2.0)[0]
    keep[in4[: int(0.7 * len(in4))]] = False
    mid6 = epoch + 6 * period
    in6 = np.where(np.abs(t - mid6) < dur / 2.0)[0]
    keep[in6[: int(0.3 * len(in6))]] = False
    tg = t[keep]
    n = extractor.count_observed_transits(tg, period, epoch, dur, 0.02)
    assert n == 10 - 2 - 1
    assert extractor.count_observed_transits(tg, period, epoch, dur, 0.02, coverage=0.25) == 8
    single = extractor.count_observed_transits(t[(t > 40) & (t < 60)], period, epoch, dur, 0.02)
    assert single == 2


def test_single_event_fraction(extractor):
    period, epoch, dur = 10.0, 5.0, 0.4
    t = np.arange(0.0, 100.0, 0.02)
    f = np.ones(len(t))
    for k in range(10):
        f[np.abs(t - (epoch + k * period)) < dur / 2] -= 1e-3
    assert extractor.single_event_fraction(t, f, period, epoch, dur, 0.02) == pytest.approx(0.1, abs=0.01)
    f2 = np.ones(len(t))
    f2[np.abs(t - (epoch + 3 * period)) < dur / 2] -= 1e-2
    assert extractor.single_event_fraction(t, f2, period, epoch, dur, 0.02) == pytest.approx(1.0)


def test_duration_at_grid_edge(extractor):
    g = extractor.duration_grid(True)
    assert extractor.duration_at_grid_edge(g[0], g, 10.0) == 1
    assert extractor.duration_at_grid_edge(g[0] + 0.004, g, 10.0) == 1
    assert extractor.duration_at_grid_edge(0.504, g, 10.0) == 1
    assert extractor.duration_at_grid_edge(g[-1], g, 10.0) == 1
    assert extractor.duration_at_grid_edge(g[2], g, 10.0) == 0
    allowed = extractor.allowed_durations(g, 0.6)
    assert extractor.duration_at_grid_edge(allowed[-1], g, 0.6) == 1


def test_harmonic_matching_includes_third_and_triple(extractor):
    m = extractor.match_period_ratio
    assert m(1.005) == 'MATCHED'
    assert m(0.505) == 'HARMONIC' and m(2.005) == 'HARMONIC'
    assert m(3.02) == 'HARMONIC' and m(1.0 / 3.0 * 1.005) == 'HARMONIC'
    assert m(1.5) is None and m(2.5) is None and m(3.2) is None


def _search(extractor, t, f, dy, quick=True, max_signals=4, row_dicts=None, inverted=False):
    lc = lk.LightCurve(time=t, flux=f, flux_err=dy)
    baseline = float(t.max() - t.min())
    return extractor.search_signals(lc, '1234567', 'Kepler', row_dicts or [], baseline, CADENCE, star_feats(), quick,
                                    max_signals, time.time() + 600, True, 'test', inverted=inverted)


@pytest.mark.parametrize('seed', [11, 12, 13])
def test_white_noise_gives_no_detections(extractor, seed):
    t, f, dy = box_series(None, 0, 0, 0, 400.0, seed=seed)
    res = _search(extractor, t, f, dy)
    assert res['features'] == []


def test_search_finds_planet_with_accurate_period(extractor):
    period, duration = 12.0, 0.2
    t, f, dy = box_series(period, 3.3, duration, 1.5e-3, 700.0, seed=21)
    res = _search(extractor, t, f, dy, quick=False, row_dicts=[{'catalog_period': period, 'label': 'CONFIRMED'}])
    feats = res['features']
    assert len(feats) >= 1
    top = feats[0]
    baseline = float(t.max() - t.min())
    assert abs(top['bls_period_d'] - period) * baseline / period < duration / 4.0
    assert top['match_status'] == 'MATCHED'
    assert top['n_transits_observed'] >= 50
    assert top['single_event_fraction'] < 0.1
    assert top['bls_sde'] >= 7.0 and 'bls_sde_raw' in top
    assert top['dur_at_grid_edge'] in (0, 1)


def test_raw_sde_and_version_columns_excluded_from_features(extractor):
    import onyx_features
    cols = ['bls_sde', 'bls_sde_raw', 'n_transits_observed', 'single_event_fraction', 'dur_at_grid_edge', 'pipeline_version']
    feats = onyx_features.compute_feature_columns(cols)
    assert feats == ['bls_sde', 'n_transits_observed', 'single_event_fraction', 'dur_at_grid_edge']
    assert int(onyx_features.PIPELINE_VERSION) >= 3
