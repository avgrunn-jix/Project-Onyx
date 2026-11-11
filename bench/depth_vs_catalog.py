import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / 'outputs' / '_diag_kepler7' / 'koi_bench.csv'
KOI_COLUMNS = ['kepid', 'kepoi_name', 'koi_period', 'koi_depth', 'koi_disposition']
REQUIRED_FEATURE_COLUMNS = ['target_id', 'bls_period_d', 'bls_depth_ppm']
PERIOD_TOL = 0.01


def to_kepid(value):
    digits = re.findall(r'\d+', str(value))
    return int(digits[-1]) if digits else None


def load_features(path):
    df = pd.read_csv(Path(path), dtype={'target_id': str}, encoding='utf-8')
    missing = [c for c in REQUIRED_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"В файле признаков {path} нет колонок: {', '.join(missing)}")
    df = df.copy()
    df['kepid'] = [to_kepid(v) for v in df['target_id']]
    df = df[df['kepid'].notna() & df['bls_period_d'].notna()].copy()
    df['kepid'] = df['kepid'].astype(int)
    return df


def query_koi(kepids):
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    where = 'kepid in (' + ','.join(str(int(k)) for k in kepids) + ')'
    table = NasaExoplanetArchive.query_criteria(table='cumulative', select=','.join(KOI_COLUMNS), where=where)
    return table.to_pandas()


def normalize_catalog(df):
    missing = [c for c in KOI_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"В таблице KOI нет колонок: {', '.join(missing)}")
    out = df[KOI_COLUMNS].copy()
    for col in ('kepoi_name', 'koi_disposition'):
        out[col] = [v.decode('utf-8') if isinstance(v, bytes) else v for v in out[col]]
    out['kepid'] = pd.to_numeric(out['kepid'], errors='coerce')
    out = out[out['kepid'].notna()].copy()
    out['kepid'] = out['kepid'].astype(int)
    out['koi_period'] = pd.to_numeric(out['koi_period'], errors='coerce')
    out['koi_depth'] = pd.to_numeric(out['koi_depth'], errors='coerce')
    return out.sort_values(['kepid', 'koi_period']).reset_index(drop=True)


def load_catalog(cache_path, kepids, fetch=None):
    cache_path = Path(cache_path)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return normalize_catalog(pd.read_csv(cache_path, encoding='utf-8')), True
    fetch = query_koi if fetch is None else fetch
    df = normalize_catalog(fetch(sorted(set(int(k) for k in kepids))))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False, encoding='utf-8')
    return df, False


def match_koi(period, koi_rows, tol=PERIOD_TOL):
    best = None
    best_err = None
    for _, row in koi_rows.iterrows():
        p_cat = row['koi_period']
        if pd.isna(p_cat) or not p_cat > 0:
            continue
        err = abs(float(period) / float(p_cat) - 1.0)
        if err <= tol and (best_err is None or err < best_err):
            best, best_err = row, err
    return best


def fmt_num(val, pattern):
    if val is None or pd.isna(val):
        return 'н/д'
    try:
        return pattern.format(float(val))
    except (TypeError, ValueError):
        return str(val)


def compare_rows(feats, koi):
    rows = []
    for _, sig in feats.sort_values(['kepid', 'bls_period_d']).iterrows():
        kepid = int(sig['kepid'])
        period = float(sig['bls_period_d'])
        depth = sig['bls_depth_ppm']
        n_tr = sig['n_transits_observed'] if 'n_transits_observed' in feats.columns else None
        hit = match_koi(period, koi[koi['kepid'] == kepid])
        koi_depth = None if hit is None else hit['koi_depth']
        ratio = None
        if koi_depth is not None and not pd.isna(koi_depth) and float(koi_depth) > 0 and not pd.isna(depth):
            ratio = float(depth) / float(koi_depth)
        rows.append({
            'kepid': kepid,
            'bls_period_d': period,
            'bls_depth_ppm': depth,
            'koi_depth': koi_depth,
            'ratio': ratio,
            'n_transits_observed': n_tr,
            'kepoi_name': None if hit is None else hit['kepoi_name'],
            'koi_period': None if hit is None else hit['koi_period'],
            'koi_disposition': None if hit is None else hit['koi_disposition'],
        })
    return rows


def build_report(feats, koi, from_cache=None, cache_path=None):
    rows = compare_rows(feats, koi)
    lines = []
    if from_cache is not None:
        source = 'кэш (без сети)' if from_cache else 'запрос в NASA Exoplanet Archive, результат сохранён в кэш'
        lines.append(f'Таблица KOI: {source}: {cache_path}')
    lines.append(f'Сопоставление по периоду: прямое совпадение, допуск {PERIOD_TOL * 100:.0f}%')
    lines.append('')
    header = ['kepid', 'период, д', 'bls_depth_ppm', 'koi_depth', 'bls/koi', 'n_transits_observed', 'kepoi_name', 'koi_period', 'koi_disposition']
    table = [header]
    for r in rows:
        matched = r['kepoi_name'] is not None
        table.append([
            str(r['kepid']),
            f"{r['bls_period_d']:.4f}",
            fmt_num(r['bls_depth_ppm'], '{:.1f}'),
            fmt_num(r['koi_depth'], '{:.1f}') if matched else '-',
            fmt_num(r['ratio'], '{:.3f}') if matched else '-',
            fmt_num(r['n_transits_observed'], '{:.0f}'),
            str(r['kepoi_name']) if matched else '-',
            fmt_num(r['koi_period'], '{:.4f}') if matched else '-',
            str(r['koi_disposition']) if matched else '-',
        ])
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for row in table:
        lines.append('  '.join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    ratios = sorted(r['ratio'] for r in rows if r['ratio'] is not None)
    n_matched = sum(1 for r in rows if r['kepoi_name'] is not None)
    lines.append('')
    lines.append(f'Сигналов: {len(rows)}, сопоставлено с KOI: {n_matched}, с отношением глубин: {len(ratios)}')
    if ratios:
        mid = len(ratios) // 2
        median = ratios[mid] if len(ratios) % 2 == 1 else 0.5 * (ratios[mid - 1] + ratios[mid])
        lines.append(f'Отношение bls_depth_ppm/koi_depth: медиана {median:.3f}, минимум {ratios[0]:.3f}, максимум {ratios[-1]:.3f}')
    known = set(int(k) for k in koi['kepid'])
    absent = sorted(set(int(k) for k in feats['kepid']) - known)
    if absent:
        lines.append('kepid без строк в таблице KOI: ' + ', '.join(str(k) for k in absent))
        if from_cache:
            lines.append('Если кэш собран для другого файла признаков, удалите его и запустите скрипт снова.')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--catalog-cache', default=str(DEFAULT_CACHE))
    args = ap.parse_args(argv)
    features_path = Path(args.features)
    if not features_path.exists():
        raise SystemExit(f'Файл признаков не найден: {features_path}')
    feats = load_features(features_path)
    koi, from_cache = load_catalog(args.catalog_cache, feats['kepid'].unique().tolist())
    print(build_report(feats, koi, from_cache=from_cache, cache_path=Path(args.catalog_cache)))


if __name__ == '__main__':
    main()
