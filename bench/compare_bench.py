import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_NAME = 'extracted_features.csv'
DIAG_NAME = 'extracted_features_diagnostics.csv'


def load_run(run_dir):
    feats_path = os.path.join(run_dir, FEATURES_NAME)
    diag_path = os.path.join(run_dir, DIAG_NAME)
    feats = pd.DataFrame(columns=['target_id', 'bls_period_d'])
    diag = pd.DataFrame(columns=['target_id', 'status'])
    if os.path.exists(feats_path) and os.path.getsize(feats_path) > 0:
        feats = pd.read_csv(feats_path, dtype={'target_id': str}, encoding='utf-8')
    if os.path.exists(diag_path) and os.path.getsize(diag_path) > 0:
        diag = pd.read_csv(diag_path, dtype={'target_id': str}, encoding='utf-8')
    return feats, diag


def periods_for(feats, tid):
    sub = feats[feats['target_id'] == tid]
    return sorted(float(p) for p in sub['bls_period_d'].dropna())


def fmt_periods(periods):
    if not periods:
        return '-'
    return ', '.join(f'{p:.4f}' for p in periods)


def diag_value(diag, tid, col, default=''):
    if col not in diag.columns:
        return default
    sub = diag[diag['target_id'] == tid]
    if sub.empty:
        return default
    val = sub.iloc[-1][col]
    if pd.isna(val):
        return default
    return val


def build_report(new_dir, old_dir):
    f_new, d_new = load_run(new_dir)
    f_old, d_old = load_run(old_dir)
    tids = []
    for df in (d_new, d_old, f_new, f_old):
        for t in df['target_id'].dropna().astype(str):
            if t not in tids:
                tids.append(t)
    lines = []
    fewer = []
    aliased = []
    for tid in tids:
        p_new = periods_for(f_new, tid)
        p_old = periods_for(f_old, tid)
        st_new = diag_value(d_new, tid, 'status', 'нет данных')
        st_old = diag_value(d_old, tid, 'status', 'нет данных')
        t_new = diag_value(d_new, tid, 'time_total_s', '')
        t_old = diag_value(d_old, tid, 'time_total_s', '')
        af = str(diag_value(d_new, tid, 'alias_factors', ''))
        lines.append(f'Звезда {tid}')
        lines.append(f'  статус: new={st_new} | old={st_old}')
        lines.append(f'  время, с: new={t_new} | old={t_old}')
        lines.append(f'  сигналов: new={len(p_new)} | old={len(p_old)}')
        lines.append(f'  периоды new: {fmt_periods(p_new)}')
        lines.append(f'  периоды old: {fmt_periods(p_old)}')
        lines.append(f'  alias_factors (new): {af if af else "-"}')
        inv_new = diag_value(d_new, tid, 'inverted_count', '')
        inv_old = diag_value(d_old, tid, 'inverted_count', '')
        if str(inv_new) != '' or str(inv_old) != '':
            lines.append(f'  INVERTED-сигналов: new={inv_new} | old={inv_old}')
        lines.append('')
        if len(p_new) < len(p_old):
            fewer.append((tid, len(p_new), len(p_old)))
        if af:
            aliased.append((tid, af))
    lines.append('=== Звезды, где в новом режиме сигналов меньше, чем в старом ===')
    if fewer:
        for tid, n_new, n_old in fewer:
            lines.append(f'  {tid}: new={n_new}, old={n_old}')
    else:
        lines.append('  нет')
    lines.append('')
    lines.append('=== Звезды с непустыми alias_factors ===')
    if aliased:
        for tid, af in aliased:
            lines.append(f'  {tid}: {af}')
    else:
        lines.append('  нет')
    return '\n'.join(lines) + '\n'


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    out_root = os.path.join(ROOT, 'outputs')
    new_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(out_root, 'bench_new')
    old_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(out_root, 'bench_old')
    out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(out_root, 'bench_compare.txt')
    report = build_report(new_dir, old_dir)
    print(report)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == '__main__':
    main()
