import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onyx_features import compute_feature_columns

RUN_DIR = ROOT / 'outputs' / 'demo_aws_2000_cloud'
FEATURES_SUBDIR = 'demo_aws_2000'
OUT_DIR = ROOT / 'outputs' / 'experiments' / 'e1_e2'
SHOWCASE_ID = '4852528'
SEED = 42

PART_TRAIN = 'обучение'
PART_VAL = 'валидация'
PART_CALIB = 'калибровка'
PART_TEST = 'тест'
PART_HIDDEN = 'отложено'
PARTS = [PART_TRAIN, PART_VAL, PART_CALIB, PART_TEST]

LABEL_POS = 'CONFIRMED'
LABEL_NEG = 'FALSE POSITIVE'
LABEL_CAND = 'CANDIDATE'

SPLIT_STARS_FILE = 'e1_split_stars.csv'
FEATURE_LIST_FILE = 'e1_feature_columns.json'


def features_dir(run_dir):
    return Path(run_dir) / FEATURES_SUBDIR


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path):
    with open(path, 'rb') as fh:
        return sum(1 for _ in fh)


def model_feature_columns(df_or_cols):
    return compute_feature_columns(df_or_cols, exclude_coverage=True)


def read_features(path):
    return pd.read_csv(path, dtype={'target_id': str, 'signal_id': str})


def labeled_mask(df, showcase_id=SHOWCASE_ID):
    return (df['match_status'] == 'MATCHED') & df['label'].isin([LABEL_POS, LABEL_NEG]) & (df['target_id'].astype(str) != str(showcase_id))


def candidate_mask(df, showcase_id=SHOWCASE_ID):
    return (df['match_status'] == 'MATCHED') & (df['label'] == LABEL_CAND) & (df['target_id'].astype(str) != str(showcase_id))


def load_demo_module():
    name = 'onyx_demo_exp'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(ROOT / '3_demo_pipeline.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def confidence_order(p, tiebreak=None):
    p = np.asarray(p, dtype=float)
    conf = np.maximum(p, 1.0 - p)
    tb = np.arange(len(p)) if tiebreak is None else np.asarray(tiebreak)
    return np.lexsort((tb, -conf))


def cumulative_errors(y, p, tiebreak=None):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = confidence_order(p, tiebreak)
    wrong = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    return np.cumsum(wrong)


def risk_coverage_curve(y, p, tiebreak=None):
    n = len(y)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    err = cumulative_errors(y, p, tiebreak)
    k = np.arange(1, n + 1, dtype=float)
    return k / n, err / k


def accepted_count(n, coverage):
    return min(n, max(1, int(math.ceil(coverage * n - 1e-9))))


def selective_risk(y, p, coverage, tiebreak=None):
    n = len(y)
    if n == 0:
        return float('nan')
    err = cumulative_errors(y, p, tiebreak)
    k = accepted_count(n, coverage)
    return float(err[k - 1] / k)


def aurc(y, p, tiebreak=None):
    n = len(y)
    if n == 0:
        return float('nan')
    err = cumulative_errors(y, p, tiebreak)
    return float(np.mean(err / np.arange(1, n + 1)))


def augrc(y, p, tiebreak=None):
    n = len(y)
    if n == 0:
        return float('nan')
    err = cumulative_errors(y, p, tiebreak)
    return float(np.mean(err / n))


def roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, p))


def average_precision(y, p):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float('nan')
    return float(average_precision_score(y, p))


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(y) else float('nan')


def accuracy(y, p):
    y = np.asarray(y).astype(int)
    return float(np.mean((np.asarray(p) >= 0.5).astype(int) == y)) if len(y) else float('nan')


COVERAGES = (0.50, 0.70, 0.90)


def metric_names():
    return ['roc_auc', 'ap', 'brier'] + [f"risk@{c:.2f}" for c in COVERAGES] + ['aurc', 'augrc']


def all_metrics(y, p, tiebreak=None):
    out = {'roc_auc': roc_auc(y, p), 'ap': average_precision(y, p), 'brier': brier(y, p)}
    n = len(y)
    err = cumulative_errors(y, p, tiebreak) if n else np.zeros(0)
    for c in COVERAGES:
        k = accepted_count(n, c) if n else 0
        out[f"risk@{c:.2f}"] = float(err[k - 1] / k) if n else float('nan')
    out['aurc'] = float(np.mean(err / np.arange(1, n + 1))) if n else float('nan')
    out['augrc'] = float(np.mean(err / n)) if n else float('nan')
    return out


def star_index_map(groups):
    groups = np.asarray(groups).astype(str)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    return uniq, members


def star_bootstrap_indices(groups, n_boot, seed):
    uniq, members = star_index_map(groups)
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), size=len(uniq))
        yield np.concatenate([members[i] for i in pick])


def percentile_ci(values, lo=2.5, hi=97.5):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float('nan'), float('nan'))
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


def bootstrap_metrics(y, preds_by_model, groups, n_boot=1000, seed=SEED, tiebreak=None):
    y = np.asarray(y).astype(int)
    tb = np.arange(len(y)) if tiebreak is None else np.asarray(tiebreak)
    names = metric_names()
    draws = {m: {k: [] for k in names} for m in preds_by_model}
    for idx in star_bootstrap_indices(groups, n_boot, seed):
        yb = y[idx]
        tbb = tb[idx]
        for m, reps in preds_by_model.items():
            vals = [all_metrics(yb, np.asarray(r)[idx], tbb) for r in reps]
            for k in names:
                draws[m][k].append(float(np.mean([v[k] for v in vals])))
    return {m: {k: np.asarray(v) for k, v in d.items()} for m, d in draws.items()}


def fmt_num(x, nd=3):
    if x is None or not np.isfinite(x):
        return 'н/д'
    s = f"{x:.{nd}f}".replace('.', ',')
    return s.replace('-', '−')


def fmt_ci(x, ci, nd=3):
    return f"{fmt_num(x, nd)} ({fmt_num(ci[0], nd)}; {fmt_num(ci[1], nd)})"


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(str(v) for v in row.values) + ' |')
    return '\n'.join(lines) + '\n'


def heading(level, text):
    return chr(35) * level + ' ' + text + '\n\n'


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write('\n')


def write_text(path, text):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)


def package_versions():
    import platform
    out = {'python': platform.python_version()}
    for name in ['numpy', 'pandas', 'scipy', 'sklearn', 'xgboost', 'lightgbm', 'matplotlib', 'shap', 'psutil']:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            out[name] = 'не установлен'
    return out
