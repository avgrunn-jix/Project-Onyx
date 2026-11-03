import numpy as np
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
    monkeypatch.setattr(extractor, "CLIP_UPWARD_ONLY", True)
    monkeypatch.setattr(extractor, "RESOLVE_ALIASES", True)
    extractor.init_worker("dummy_sem", clip_upward=False, resolve_aliases=False)
    assert extractor.CLIP_UPWARD_ONLY is False
    assert extractor.RESOLVE_ALIASES is False
    extractor.init_worker("dummy_sem", clip_upward=True, resolve_aliases=True)
    assert extractor.CLIP_UPWARD_ONLY is True
    assert extractor.RESOLVE_ALIASES is True

