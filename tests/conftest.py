import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CACHE = {}


def load_module(file_name, alias):
    if alias in _CACHE:
        return _CACHE[alias]
    spec = importlib.util.spec_from_file_location(alias, str(ROOT / file_name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    _CACHE[alias] = mod
    return mod


@pytest.fixture(scope="session")
def extractor():
    return load_module("1_data_extractor.py", "onyx_extractor")


@pytest.fixture(scope="session")
def demo():
    return load_module("3_demo_pipeline.py", "onyx_demo")


@pytest.fixture(scope="session")
def ml_pipeline():
    return load_module("2_ml_pipeline.py", "onyx_ml")


def run_demo_report(demo_mod, root, ws, extra=()):
    from synth_demo import SHOWCASE
    mp = pytest.MonkeyPatch()
    calls = []
    orig = demo_mod.train_and_eval_pipeline

    def spy(*a, **k):
        calls.append(k.get('threshold_criterion'))
        return orig(*a, **k)

    try:
        mp.chdir(root)
        mp.setattr(demo_mod, 'train_and_eval_pipeline', spy)
        mp.setattr(sys, 'argv', ['x', '--quick', '--report-only', '--output-dir', ws['out_dir'], '--showcase-star', SHOWCASE, '--cache-dir', str(Path(root) / 'c')] + list(extra))
        demo_mod.main()
    finally:
        mp.undo()
    return Path(ws['out_dir'], 'summary_quick_report.md').read_text(encoding='utf-8'), calls


def write_demo_diagnostics(ws, short_ids, with_quarters=True):
    import pandas as pd
    rows = []
    groups = ws['groups']
    for sid in groups['tr_m'] + groups['ctrl_m'] + groups['tr_u'] + groups['ctrl_u']:
        short = sid in short_ids
        row = {'target_id': sid, 'status': 'OK', 'n_files': 12 if short else 17, 'n_quarters': 12 if short else 17,
               'baseline_d': 1000.0 if short else 1470.0, 'signals_found': int((ws['features']['target_id'] == sid).sum())}
        if not with_quarters:
            row.pop('n_quarters')
        rows.append(row)
    pd.DataFrame(rows).to_csv(Path(ws['out_dir'], 'features_demo_quick_diagnostics.csv'), index=False)


@pytest.fixture(scope="session")
def fixes_default_report(demo, tmp_path_factory):
    from synth_demo import build_workspace
    root = tmp_path_factory.mktemp('report_default')
    ws = build_workspace(root)
    short = set(ws['groups']['tr_u'][:3] + ws['groups']['ctrl_u'][:2])
    write_demo_diagnostics(ws, short)
    text, calls = run_demo_report(demo, root, ws)
    return {'text': text, 'calls': calls, 'ws': ws}


@pytest.fixture(scope="session")
def fixes_wilson_report(demo, tmp_path_factory):
    import pandas as pd
    from synth_demo import build_workspace
    root = tmp_path_factory.mktemp('report_wilson')
    ws = build_workspace(root)
    feat_path = Path(ws['out_dir'], 'features_demo_quick.csv')
    df = pd.read_csv(feat_path, dtype={'target_id': str})
    df = df.drop(columns=[c for c in df.columns if c.startswith('centroid_shift_')])
    df.to_csv(feat_path, index=False)
    ws['features'] = df
    write_demo_diagnostics(ws, set(), with_quarters=False)
    text, calls = run_demo_report(demo, root, ws, extra=['--threshold-criterion', 'wilson'])
    return {'text': text, 'calls': calls, 'ws': ws}
