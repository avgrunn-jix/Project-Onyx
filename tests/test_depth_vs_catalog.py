import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location('onyx_depth_vs_catalog', str(ROOT / 'bench' / 'depth_vs_catalog.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_features(path, with_transits=True):
    rows = [
        {'target_id': '5780885', 'bls_period_d': 4.88543, 'bls_depth_ppm': 6400.0, 'n_transits_observed': 275},
        {'target_id': '5780885', 'bls_period_d': 94.1761, 'bls_depth_ppm': 101.7, 'n_transits_observed': 13},
        {'target_id': 'KIC 11442793', 'bls_period_d': 8.7198, 'bls_depth_ppm': 151.6, 'n_transits_observed': 114},
        {'target_id': '11442793', 'bls_period_d': 14.52, 'bls_depth_ppm': 90.0, 'n_transits_observed': 80},
        {'target_id': '999', 'bls_period_d': 3.0, 'bls_depth_ppm': 50.0, 'n_transits_observed': 30},
    ]
    df = pd.DataFrame(rows)
    if not with_transits:
        df = df.drop(columns=['n_transits_observed'])
    df.to_csv(path, index=False, encoding='utf-8')


def koi_table():
    return pd.DataFrame([
        {'kepid': 5780885, 'kepoi_name': b'K00097.01', 'koi_period': 4.8854892, 'koi_depth': 8000.0, 'koi_disposition': 'CONFIRMED', 'extra': 1},
        {'kepid': 11442793, 'kepoi_name': 'K00351.03', 'koi_period': 8.7193, 'koi_depth': 85.0, 'koi_disposition': 'CONFIRMED', 'extra': 2},
        {'kepid': 11442793, 'kepoi_name': 'K00351.04', 'koi_period': 8.80, 'koi_depth': 1.0, 'koi_disposition': 'FALSE POSITIVE', 'extra': 3},
        {'kepid': 11442793, 'kepoi_name': 'K00351.05', 'koi_period': 7.26, 'koi_depth': 70.0, 'koi_disposition': 'CONFIRMED', 'extra': 4},
        {'kepid': 11442793, 'kepoi_name': 'K00351.06', 'koi_period': 14.45, 'koi_depth': None, 'koi_disposition': 'CONFIRMED', 'extra': 5},
    ])


def test_catalog_is_fetched_once_and_then_read_from_cache(tmp_path):
    dc = load_script()
    cache = tmp_path / 'nested' / 'koi_bench.csv'
    calls = []

    def fetch(kepids):
        calls.append(list(kepids))
        return koi_table()

    first, from_cache = dc.load_catalog(cache, [11442793, 5780885, 5780885, 999], fetch=fetch)
    assert from_cache is False
    assert calls == [[999, 5780885, 11442793]]
    assert cache.exists()
    assert list(first.columns) == dc.KOI_COLUMNS
    assert list(pd.read_csv(cache, encoding='utf-8').columns) == dc.KOI_COLUMNS
    assert 'K00097.01' in set(first['kepoi_name'])

    def no_network(kepids):
        raise AssertionError('network must not be used when the cache exists')

    second, from_cache = dc.load_catalog(cache, [5780885], fetch=no_network)
    assert from_cache is True
    assert len(second) == len(first) == 5
    assert list(second['kepoi_name']) == list(first['kepoi_name'])


def test_match_is_direct_period_within_one_percent(tmp_path):
    dc = load_script()
    koi = dc.normalize_catalog(koi_table())
    rows = koi[koi['kepid'] == 11442793]
    assert dc.match_koi(8.7198, rows)['kepoi_name'] == 'K00351.03'
    assert dc.match_koi(8.74, rows)['kepoi_name'] == 'K00351.03'
    assert dc.match_koi(8.7193 * 1.0101, rows)['kepoi_name'] == 'K00351.04'
    assert dc.match_koi(8.7193 * 0.989, rows) is None
    assert dc.match_koi(8.7193 * 2.0, rows) is None
    assert dc.match_koi(8.7193 / 2.0, rows) is None
    assert dc.match_koi(3.0, koi[koi['kepid'] == 999]) is None
    assert dc.to_kepid('KIC 11442793') == 11442793
    assert dc.to_kepid('005780885') == 5780885
    assert dc.to_kepid('abc') is None


def test_report_lists_every_signal_with_depth_ratio(tmp_path, capsys):
    dc = load_script()
    feats_path = tmp_path / 'extracted_features.csv'
    write_features(feats_path)
    cache = tmp_path / 'koi_bench.csv'
    dc.normalize_catalog(koi_table()).to_csv(cache, index=False, encoding='utf-8')
    dc.main(['--features', str(feats_path), '--catalog-cache', str(cache)])
    out = capsys.readouterr().out
    lines = [' '.join(l.split()) for l in out.splitlines()]
    assert lines[0].startswith('Таблица KOI: кэш (без сети)')
    assert 'kepid период, д bls_depth_ppm koi_depth bls/koi n_transits_observed kepoi_name koi_period koi_disposition' in lines
    assert '999 3.0000 50.0 - - 30 - - -' in lines
    assert '5780885 4.8854 6400.0 8000.0 0.800 275 K00097.01 4.8855 CONFIRMED' in lines
    assert '5780885 94.1761 101.7 - - 13 - - -' in lines
    assert '11442793 8.7198 151.6 85.0 1.784 114 K00351.03 8.7193 CONFIRMED' in lines
    assert '11442793 14.5200 90.0 н/д н/д 80 K00351.06 14.4500 CONFIRMED' in lines
    assert 'Сигналов: 5, сопоставлено с KOI: 3, с отношением глубин: 2' in lines
    assert 'Отношение bls_depth_ppm/koi_depth: медиана 1.292, минимум 0.800, максимум 1.784' in lines
    assert 'kepid без строк в таблице KOI: 999' in lines


def test_report_without_transit_column_and_missing_inputs(tmp_path):
    dc = load_script()
    feats_path = tmp_path / 'extracted_features.csv'
    write_features(feats_path, with_transits=False)
    feats = dc.load_features(feats_path)
    report = dc.build_report(feats, dc.normalize_catalog(koi_table()))
    lines = [' '.join(l.split()) for l in report.splitlines()]
    assert '5780885 4.8854 6400.0 8000.0 0.800 н/д K00097.01 4.8855 CONFIRMED' in lines
    assert 'Таблица KOI:' not in report
    with pytest.raises(SystemExit):
        dc.main(['--features', str(tmp_path / 'absent.csv'), '--catalog-cache', str(tmp_path / 'c.csv')])
    bad = tmp_path / 'bad.csv'
    pd.DataFrame([{'target_id': '1', 'bls_period_d': 2.0}]).to_csv(bad, index=False, encoding='utf-8')
    with pytest.raises(SystemExit):
        dc.load_features(bad)
    with pytest.raises(SystemExit):
        dc.normalize_catalog(pd.DataFrame([{'kepid': 1, 'koi_period': 2.0}]))


def test_script_default_cache_path_and_no_comments():
    dc = load_script()
    assert dc.DEFAULT_CACHE == ROOT / 'outputs' / '_diag_kepler7' / 'koi_bench.csv'
    src = (ROOT / 'bench' / 'depth_vs_catalog.py').read_text(encoding='utf-8')
    assert not any(line.lstrip().startswith('#') for line in src.splitlines())
    assert '"""' not in src
    assert "table='cumulative'" in src
    assert "ap.add_argument('--features', required=True)" in src
    assert "ap.add_argument('--catalog-cache', default=str(DEFAULT_CACHE))" in src
