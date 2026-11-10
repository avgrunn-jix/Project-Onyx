import os

import pandas as pd

PIPELINE_VERSION = '2'

EXCLUDE_COLS = [
    'signal_id',
    'target_id',
    'signal_index',
    'mission',
    'label',
    'match_status',
    'target_bin',
    'koi_fpflag_nt',
    'koi_fpflag_ss',
    'koi_fpflag_co',
    'koi_fpflag_ec',
    'p_candidate',
    'n_signals_star',
    'pipeline_version',
    'bls_epoch',
]

COVERAGE_COLS = [
    'baseline_d',
    'n_files',
    'n_quarters',
    'quarters',
    'st_quarters',
    'dataspan',
    'cadence_days',
    'mean_cadence_min',
]


def is_koi_column(col_name):
    s = str(col_name).strip().lower()
    return s.startswith('koi_') or s == 'kepoi_name'


def _to_col_list(df_or_cols):
    if hasattr(df_or_cols, 'columns'):
        return list(df_or_cols.columns)
    return list(df_or_cols)


def compute_feature_columns(df_or_cols, exclude_coverage=False, other_cols=None, extra_exclude=None):
    cols = _to_col_list(df_or_cols)
    excl = set(EXCLUDE_COLS)
    if exclude_coverage:
        excl.update(COVERAGE_COLS)
    if extra_exclude is not None:
        excl.update(extra_exclude)
    other_set = set(_to_col_list(other_cols)) if other_cols is not None else None
    result = []
    for c in cols:
        if c in excl or is_koi_column(c):
            continue
        if other_set is not None and c not in other_set:
            continue
        result.append(c)
    return result


class PipelineVersionError(RuntimeError):
    pass


def build_pipeline_version(quick, flatten_days, max_signals):
    return f"{PIPELINE_VERSION}|quick={int(bool(quick))}|flatten_days={float(flatten_days):g}|max_signals={int(max_signals)}"


def read_pipeline_versions(path):
    header = pd.read_csv(path, nrows=0)
    if 'pipeline_version' not in header.columns:
        return None
    col = pd.read_csv(path, usecols=['pipeline_version'], dtype=str)['pipeline_version']
    return sorted(set(col.dropna().astype(str)))


def check_pipeline_version(path, expected, allow_legacy=False, report_only=False):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    found = read_pipeline_versions(path)
    if found is not None and found == [expected]:
        return None
    if found is None:
        what = 'has no pipeline_version column'
    elif len(found) == 0:
        what = 'has an empty pipeline_version column'
    else:
        what = 'has pipeline_version ' + ', '.join(found)
    msg = f"Features file {path} {what}; current version is {expected}."
    if report_only:
        return msg + ' Report-only mode continues with these files as they are.'
    if allow_legacy:
        return msg + ' Continuing because --allow-legacy is set.'
    raise PipelineVersionError(msg + ' Resuming would mix incompatible rows. Use --no-resume, a different --output, or --allow-legacy.')
