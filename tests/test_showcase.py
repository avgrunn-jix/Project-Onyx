import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import onyx_showcase
from conftest import load_module

CYR = re.compile('[\u0400-\u04FF]')


def make_data(n_signals=4):
    rng = np.random.default_rng(11)
    t = np.arange(100.0, 300.0, 0.0204)
    f = 1.0 + rng.normal(0, 3e-4, len(t))
    rows = []
    specs = [(9.5215, 105.0, 0.12, 1351.0), (7.0535, 103.0, 0.1, 1000.0), (4.6454, 104.0, 0.09, 700.0), (3.0721, 101.5, 0.08, 500.0)]
    for p, t0, d, dep in specs[:n_signals]:
        ph = ((t - t0 + 0.5 * p) % p) - 0.5 * p
        f[np.abs(ph) < d / 2] -= dep * 1e-6
        rows.append(dict(bls_period_d=p, bls_sde=12.0, bls_epoch=t0, bls_duration_d=d, bls_depth_ppm=dep, p_mean=0.84, decision='REVIEW'))
    periods = np.linspace(0.5, 12.0, 800)
    sde = rng.normal(0, 1, 800)
    return dict(showcase_id='4852528', signals=pd.DataFrame(rows), times_raw=t, flux_raw=f, times_flat=t, flux_flat=f,
                bls_period=periods, sde_curve=sde, n_files=4, n_quarters=4, baseline_days=200.0)


def all_texts(fig):
    texts = []
    for ax in fig.axes:
        texts.append(ax.get_title())
        texts.append(ax.get_xlabel())
        texts.append(ax.get_ylabel())
        texts.extend(t.get_text() for t in ax.texts)
        leg = ax.get_legend()
        if leg is not None:
            texts.extend(t.get_text() for t in leg.get_texts())
    for leg in fig.legends:
        texts.extend(t.get_text() for t in leg.get_texts())
    return texts


def test_wide_is_wider_than_tall_and_has_seven_panels(tmp_path):
    out = tmp_path / 'w.png'
    fig = onyx_showcase.render_showcase(make_data(), str(out), layout='wide', lang='en', close=False)
    w, h = fig.get_size_inches()
    assert w > h
    assert len(fig.axes) == 7
    plt.close(fig)
    with Image.open(out) as im:
        assert im.size == (2550, 1920)


@pytest.mark.parametrize('layout', ['wide', 'tall'])
def test_english_has_no_cyrillic(tmp_path, layout):
    data = make_data()
    data['signals'].loc[1, 'decision'] = 'НЕ_ОЦЕНИВАЛСЯ'
    fig = onyx_showcase.render_showcase(data, str(tmp_path / 'e.png'), layout=layout, lang='en', close=False)
    texts = all_texts(fig)
    plt.close(fig)
    assert any('Raw photometry' in t for t in texts)
    assert not [t for t in texts if CYR.search(t)]


def test_russian_contains_cyrillic(tmp_path):
    fig = onyx_showcase.render_showcase(make_data(), str(tmp_path / 'r.png'), layout='wide', lang='ru', close=False)
    texts = all_texts(fig)
    plt.close(fig)
    assert any(CYR.search(t) for t in texts)


def test_wide_titles(tmp_path):
    fig = onyx_showcase.render_showcase(make_data(), str(tmp_path / 't.png'), layout='wide', lang='en', close=False)
    titles = [ax.get_title() for ax in fig.axes]
    plt.close(fig)
    assert 'Signal 1\nP = 9.5215 d, 1351 ppm\np = 0.84, REVIEW' in titles


def test_tall_matches_reference_render(tmp_path):
    import lightkurve as lk
    rng = np.random.default_rng(7)
    cols = []
    for q in range(4):
        t = 100.0 + q * 90 + np.arange(0, 85, 0.0204)
        f = 1.0 + rng.normal(0, 3e-4, len(t))
        for p, t0, d, dep in [(9.5215, 105.0, 0.12, 1.3e-3), (7.0535, 103.0, 0.1, 1.0e-3)]:
            ph = ((t - t0 + 0.5 * p) % p) - 0.5 * p
            f[np.abs(ph) < d / 2] -= dep
        cols.append(lk.LightCurve(time=t, flux=f, flux_err=np.full(len(t), 3e-4)))
    sig = pd.DataFrame([
        dict(bls_period_d=9.5215, bls_sde=14.0, bls_epoch=105.0, bls_duration_d=0.12, bls_depth_ppm=1351.0, p_mean=0.84, decision='REVIEW'),
        dict(bls_period_d=7.0535, bls_sde=11.0, bls_epoch=103.0, bls_duration_d=0.10, bls_depth_ppm=1000.0, p_mean=0.40, decision='REJECT'),
    ])
    stitched = lk.LightCurveCollection(cols).stitch().remove_nans()
    flat = stitched.flatten(window_length=101)
    bls = flat.to_periodogram(method='bls', minimum_period=0.5, maximum_period=12.0, frequency_factor=5000, duration=np.linspace(0.04, 0.35, 4))
    pw = bls.power.value
    sde = (pw - np.nanmean(pw)) / np.nanstd(pw)
    data = dict(showcase_id='4852528', signals=sig, times_raw=stitched.time.value, flux_raw=stitched.flux.value,
                times_flat=flat.time.value, flux_flat=flat.flux.value, bls_period=bls.period.value, sde_curve=sde,
                n_files=4, n_quarters=4, baseline_days=float(np.ptp(stitched.time.value)))

    class FakeSR(list):
        def download_all(self, download_dir=None):
            return lk.LightCurveCollection(cols)

    demo = load_module('3_demo_pipeline.py', 'onyx_demo')
    orig = lk.search_lightcurve
    lk.search_lightcurve = lambda *a, **k: FakeSR([1])
    try:
        demo.generate_showcase_figure('4852528', sig, str(tmp_path / 'demo.png'), True, cache_dir=None)
    finally:
        lk.search_lightcurve = orig
    onyx_showcase.render_showcase(data, str(tmp_path / 'direct.png'), layout='tall', lang='ru')
    a = np.asarray(Image.open(tmp_path / 'demo.png').convert('RGB'))
    b = np.asarray(Image.open(tmp_path / 'direct.png').convert('RGB'))
    assert a.shape == b.shape
    assert np.array_equal(a, b)


def test_match_verdicts_tolerance():
    sig = pd.DataFrame([dict(bls_period_d=9.5215), dict(bls_period_d=7.0), dict(bls_period_d=3.0)])
    scored = pd.DataFrame([
        dict(bls_period_d=9.55, p_mean=0.84, decision='REVIEW'),
        dict(bls_period_d=7.2, p_mean=0.5, decision='CANDIDATE'),
    ])
    out = onyx_showcase.match_verdicts(sig, scored)
    assert out.loc[0, 'p_mean'] == pytest.approx(0.84)
    assert out.loc[0, 'decision'] == 'REVIEW'
    assert pd.isnull(out.loc[1, 'p_mean'])
    assert pd.isnull(out.loc[2, 'decision'])


def test_missing_match_gives_na_title(tmp_path):
    data = make_data()
    data['signals'] = onyx_showcase.match_verdicts(data['signals'].drop(columns=['p_mean', 'decision']), None)
    fig = onyx_showcase.render_showcase(data, str(tmp_path / 'n.png'), layout='wide', lang='en', close=False)
    titles = [ax.get_title() for ax in fig.axes]
    plt.close(fig)
    assert 'Signal 1\nP = 9.5215 d, 1351 ppm\np = n/a, n/a' in titles


def test_make_showcase_reads_signals(tmp_path):
    ms = load_module('make_showcase.py', 'onyx_make_showcase')
    feats = tmp_path / 'f.csv'
    pd.DataFrame({'target_id': ['4852528', '4852528', '1', '4852528'], 'signal_index': [1, 0, 0, 2],
                  'bls_period_d': [7.05, 9.52, 2.0, 4.6]}).to_csv(feats, index=False)
    df = ms.read_star_signals(str(feats))
    assert list(df['bls_period_d']) == [9.52, 7.05, 4.6]
    assert ms.read_scored_signals(str(tmp_path / 'missing.csv')) is None


def text_objects(fig):
    objs = []
    for ax in fig.axes:
        objs.append(ax.title)
        objs.append(ax.xaxis.label)
        objs.append(ax.yaxis.label)
        objs.extend(ax.texts)
        objs.extend(ax.get_xticklabels())
        objs.extend(ax.get_yticklabels())
        leg = ax.get_legend()
        if leg is not None:
            objs.extend(leg.get_texts())
    for leg in fig.legends:
        objs.extend(leg.get_texts())
    objs.extend(fig.texts)
    return [o for o in objs if o.get_text().strip()]


@pytest.mark.parametrize('lang', ['ru', 'en'])
def test_wide_fonts_and_height_fit_page(tmp_path, lang):
    fig = onyx_showcase.render_showcase(make_data(), str(tmp_path / 'f.png'), layout='wide', lang=lang, close=False)
    w, h = fig.get_size_inches()
    min_font = min(o.get_fontsize() for o in text_objects(fig))
    plt.close(fig)
    assert min_font * 6.27 / w >= 6.3
    assert h * 6.27 / w <= 4.8


@pytest.mark.parametrize('lang', ['ru', 'en'])
def test_wide_text_inside_figure_and_not_overlapping(tmp_path, lang):
    fig = onyx_showcase.render_showcase(make_data(), str(tmp_path / 'o.png'), layout='wide', lang=lang, close=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    items = []
    for ax in fig.axes:
        items.append(ax.title)
        items.append(ax.xaxis.label)
        items.append(ax.yaxis.label)
    items.extend(fig.texts)
    for leg in fig.legends:
        items.extend(leg.get_texts())
    for ax in fig.axes:
        if ax.get_legend() is not None:
            items.extend(ax.get_legend().get_texts())
    boxes = []
    for o in items:
        if not o.get_text().strip():
            continue
        bb = o.get_window_extent(renderer)
        boxes.append((o, bb))
        assert bb.x0 >= fb.x0 - 1 and bb.x1 <= fb.x1 + 1
        assert bb.y0 >= fb.y0 - 1 and bb.y1 <= fb.y1 + 1
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i][1].overlaps(boxes[j][1]), (boxes[i][0].get_text(), boxes[j][0].get_text())
    plt.close(fig)
