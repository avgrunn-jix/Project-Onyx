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
