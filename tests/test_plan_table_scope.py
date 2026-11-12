import sys

import pandas as pd
import pytest

from synth_demo import SHOWCASE, build_workspace


def test_select_stellar_info_keeps_only_requested_ids(demo):
    cache = {str(i): (13.0, 17) for i in range(100)}
    got = demo.select_stellar_info(cache, [3, '5', '7', 'missing'])
    assert got == {'3': (13.0, 17), '5': (13.0, 17), '7': (13.0, 17)}
    assert demo.select_stellar_info({}, ['1']) == {}


class _Stop(Exception):
    pass


def test_plan_table_counts_only_selected_stars(demo, monkeypatch, tmp_path):
    ws = build_workspace(tmp_path)
    labels = pd.read_csv(tmp_path / 'data' / 'labels' / 'kepler_labels.csv', dtype={'target_id': str})
    cache = {str(s): (13.5, 17) for s in labels['target_id'].unique()}
    cache.update({str(5000000 + i): (14.0, 12) for i in range(500)})
    pool = pd.DataFrame([{'kepid': str(9000000 + 31 * i), 'kepmag': 13.0 + (i % 5) * 0.3, 'st_quarters': '1' * 17, 'dataspan': 1400.0} for i in range(200)])
    monkeypatch.setattr(demo, 'fetch_stellar_params', lambda ids, cache_file: dict(cache))
    monkeypatch.setattr(demo, 'get_unmarked_pool', lambda *a, **k: pool)
    seen = []
    real = demo.compute_group_stats

    def spy(info):
        seen.append(dict(info))
        if len(seen) == 2:
            raise _Stop()
        return real(info)

    monkeypatch.setattr(demo, 'compute_group_stats', spy)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['x', '--quick', '--marked-stars', '10', '--unmarked-stars', '10', '--output-dir', ws['out_dir'], '--showcase-star', SHOWCASE, '--cache-dir', str(tmp_path / 'c')])
    with pytest.raises(_Stop):
        demo.main()
    marked_seen, unmarked_seen = seen
    assert len(cache) > 500
    assert len(marked_seen) == 10
    assert set(marked_seen) <= set(labels['target_id'])
    assert SHOWCASE in marked_seen
    assert len(unmarked_seen) == 10
    assert set(unmarked_seen).isdisjoint(set(labels['target_id']))
