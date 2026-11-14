import argparse
import datetime
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_common as ec

FRACTIONS = (0.50, 0.15, 0.15, 0.20)
TEST_IDS_FILE = 'split_test_ids.csv'
CALIB_IDS_FILE = 'split_calib_ids.csv'
EXPECTED_RUN_COMMIT = 'b1f1130fceb9dc252a55c760a286e71cc52606cc'
EXPECTED = {
    'features_rows': 3219, 'features_cols': 56, 'features_size': 2626241,
    'pipeline_version': '8|quick=0|flatten_days=3|max_signals=4',
    'match_status': {'UNMATCHED': 2404, 'MATCHED': 723, 'HARMONIC': 92},
    'matched_labels': {'FALSE POSITIVE': 321, 'CONFIRMED': 282, 'CANDIDATE': 120},
    'inverted_rows': 2502, 'diag_rows': 2000,
}
INPUT_FILES = [
    ('features_demo.csv', True), ('features_demo_inverted.csv', True), ('features_demo_failed.csv', True),
    ('diagnostics_stars.csv', True), ('summary.md', True), ('demo_stars_split.csv', True),
    ('run_commit.txt', False), ('versions.txt', False),
]
ROLE_LABELED = 'размеченная'
ROLE_CAND_ONLY = 'только CANDIDATE'
ROLE_MARKED_EMPTY = 'отмеченная без разметки'
ROLE_UNMARKED = 'неразмеченная'
ROLE_FAILED = 'сбой'
ROLE_SHOWCASE = 'витринная система'
STRATA = ('только CONFIRMED', 'только FALSE POSITIVE', 'оба класса')
BALANCE_COLS = [('bls_period_d', 'период, сут'), ('bls_depth_ppm', 'глубина, ppm'), ('bls_depth_snr', 'bls_depth_snr'), ('n_transits_observed', 'число наблюдённых транзитов'), ('transit_count', 'transit_count')]
PURPOSE = {
    ec.PART_TRAIN: 'обучение семейств и подбор гиперпараметров (E2, вложенная проверка)',
    ec.PART_VAL: 'проверка выбора семейства после фиксации решения (E2, п. 3)',
    ec.PART_CALIB: 'калибровка вероятностей и порогов отказа (E3)',
    ec.PART_TEST: 'итоговая оценка, открывается один раз',
}


def input_path(run_dir, name, in_subdir):
    return (ec.features_dir(run_dir) if in_subdir else Path(run_dir)) / name


def check_identity(run_dir):
    fd = ec.features_dir(run_dir)
    f = ec.read_features(fd / 'features_demo.csv')
    problems = []
    if f.shape != (EXPECTED['features_rows'], EXPECTED['features_cols']):
        problems.append(f"features_demo.csv shape {f.shape}")
    if (fd / 'features_demo.csv').stat().st_size != EXPECTED['features_size']:
        problems.append('features_demo.csv size')
    if sorted(f['pipeline_version'].astype(str).unique()) != [EXPECTED['pipeline_version']]:
        problems.append('pipeline_version')
    if f['match_status'].value_counts().to_dict() != EXPECTED['match_status']:
        problems.append('match_status counts')
    if f.loc[f['match_status'] == 'MATCHED', 'label'].value_counts().to_dict() != EXPECTED['matched_labels']:
        problems.append('MATCHED label counts')
    if len(pd.read_csv(fd / 'features_demo_inverted.csv', usecols=[0])) != EXPECTED['inverted_rows']:
        problems.append('features_demo_inverted.csv rows')
    if len(pd.read_csv(fd / 'diagnostics_stars.csv', usecols=[0])) != EXPECTED['diag_rows']:
        problems.append('diagnostics_stars.csv rows')
    commit = (Path(run_dir) / 'run_commit.txt').read_text(encoding='utf-8').strip()
    if commit != EXPECTED_RUN_COMMIT:
        problems.append('run_commit.txt')
    if problems:
        raise SystemExit('Run identity check failed: ' + '; '.join(problems))


def allocate_counts(n, fractions):
    raw = [n * fr for fr in fractions]
    counts = [int(np.floor(r + 1e-9)) for r in raw]
    rest = n - sum(counts)
    order = sorted(range(len(fractions)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[:rest]:
        counts[i] += 1
    return counts


def hash_part(target_id, seed, fractions):
    digest = hashlib.sha256(f"{seed}|{target_id}".encode('utf-8')).hexdigest()
    u = int(digest[:15], 16) / float(16 ** 15)
    edges = np.cumsum(fractions)
    for part, edge in zip(ec.PARTS, edges):
        if u < edge:
            return part
    return ec.PARTS[-1]


def sort_ids(ids):
    return sorted(ids, key=lambda s: (len(str(s)), str(s)))


def stratified_split(star_strata, seed, fractions):
    rng = np.random.RandomState(seed)
    assign = {}
    for stratum in STRATA:
        stars = sort_ids([s for s, st in star_strata.items() if st == stratum])
        if not stars:
            continue
        perm = [stars[i] for i in rng.permutation(len(stars))]
        counts = allocate_counts(len(stars), fractions)
        pos = 0
        for part, k in zip(ec.PARTS, counts):
            for s in perm[pos:pos + k]:
                assign[s] = part
            pos += k
    return assign


def build_star_table(features, diag, failed_ids, seed, fractions, showcase_id=ec.SHOWCASE_ID):
    lab = features[ec.labeled_mask(features, showcase_id)]
    cand = features[ec.candidate_mask(features, showcase_id)]
    n_pos = lab[lab['label'] == ec.LABEL_POS].groupby('target_id').size()
    n_neg = lab[lab['label'] == ec.LABEL_NEG].groupby('target_id').size()
    n_cand = cand.groupby('target_id').size()
    n_sig = features.groupby('target_id').size()
    strata = {}
    for tid in sort_ids(set(lab['target_id'])):
        p, q = int(n_pos.get(tid, 0)), int(n_neg.get(tid, 0))
        strata[tid] = STRATA[2] if p and q else (STRATA[0] if p else STRATA[1])
    assign = stratified_split(strata, seed, fractions)
    rows = []
    for _, d in diag.iterrows():
        tid = str(d['target_id'])
        marked = str(d['star_kind']) == 'MARKED'
        if tid == str(showcase_id):
            role = ROLE_SHOWCASE
        elif tid in failed_ids:
            role = ROLE_FAILED
        elif tid in strata:
            role = ROLE_LABELED
        elif int(n_cand.get(tid, 0)) > 0:
            role = ROLE_CAND_ONLY
        elif marked:
            role = ROLE_MARKED_EMPTY
        else:
            role = ROLE_UNMARKED
        if role == ROLE_LABELED:
            part, how = assign[tid], 'стратифицированное разбиение'
        elif role in (ROLE_CAND_ONLY, ROLE_MARKED_EMPTY, ROLE_UNMARKED):
            part, how = hash_part(tid, seed, fractions), 'хеш target_id и seed'
        else:
            part, how = '', ''
        rows.append({
            'target_id': tid, 'тип': 'отмеченная' if marked else 'неотмеченная', 'роль': role,
            'слой': strata.get(tid, ''), 'сигналов': int(n_sig.get(tid, 0)),
            'размеченных': int(n_pos.get(tid, 0) + n_neg.get(tid, 0)), 'CONFIRMED': int(n_pos.get(tid, 0)),
            'FALSE_POSITIVE': int(n_neg.get(tid, 0)), 'CANDIDATE': int(n_cand.get(tid, 0)),
            'часть_полная': part, 'способ': how,
        })
    stars = pd.DataFrame(rows)
    stars = stars.iloc[[i for i, _ in sorted(enumerate(stars['target_id']), key=lambda t: (len(t[1]), t[1]))]].reset_index(drop=True)
    return stars


def public_part(part):
    return ec.PART_HIDDEN if part in (ec.PART_CALIB, ec.PART_TEST) else part


def build_registry(features, stars, showcase_id=ec.SHOWCASE_ID):
    part_of = dict(zip(stars['target_id'], stars['часть_полная']))
    role_of = dict(zip(stars['target_id'], stars['роль']))
    lab = ec.labeled_mask(features, showcase_id)
    rows = []
    for i, r in features.iterrows():
        tid = str(r['target_id'])
        if tid == str(showcase_id):
            reason = 'витринная система'
        elif r['match_status'] != 'MATCHED':
            reason = str(r['match_status'])
        elif r['label'] == ec.LABEL_CAND:
            reason = 'CANDIDATE'
        elif lab.loc[i]:
            reason = ''
        else:
            reason = 'метка вне правила: ' + str(r['label'])
        part = part_of.get(tid, '')
        note = ''
        if reason == 'CANDIDATE' and part == ec.PART_TRAIN:
            note = 'звезда в обучающей части, сигнал CANDIDATE вне обучения'
        rows.append({
            'signal_id': str(r['signal_id']), 'target_id': tid, 'label': r['label'], 'match_status': r['match_status'],
            'в_разметке': int(reason == ''), 'y': (1 if r['label'] == ec.LABEL_POS else 0) if reason == '' else '',
            'причина_исключения': reason, 'роль_звезды': role_of.get(tid, ''), 'часть': public_part(part), 'примечание': note,
        })
    return pd.DataFrame(rows)


def write_csv(df, path):
    df.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')


def flow_table(features, diag, failed, stars):
    n_plan = len(diag)
    n_failed = len(failed)
    reasons = failed['error'].astype(str).str.replace(r'clipped=\d+', 'clipped=N', regex=True).value_counts()
    m = features[features['match_status'] == 'MATCHED']
    k80 = m[m['target_id'] == ec.SHOWCASE_ID]
    m2 = m[m['target_id'] != ec.SHOWCASE_ID]
    cand = m2[m2['label'] == ec.LABEL_CAND]
    lab = m2[m2['label'].isin([ec.LABEL_POS, ec.LABEL_NEG])]
    lines = [ec.heading(1, 'E1. Поток данных прогона 2000 звёзд')]
    rows = [
        ('План прогона (diagnostics_stars.csv)', n_plan, f"отмеченных {int((diag['star_kind'] == 'MARKED').sum())}, неотмеченных {int((diag['star_kind'] != 'MARKED').sum())}"),
        ('Обработано', n_plan - n_failed, ''),
        ('В features_demo_failed.csv', n_failed, '; '.join(f"{k} ({v})" for k, v in reasons.items()) + f"; типы звёзд: {failed['star_type'].value_counts().to_dict()}"),
        ('Звёзд с хотя бы одним сигналом', int(features['target_id'].nunique()), ''),
        ('Найдено сигналов', len(features), f"UNMATCHED {int((features['match_status'] == 'UNMATCHED').sum())}, HARMONIC {int((features['match_status'] == 'HARMONIC').sum())}"),
        ('MATCHED', len(m), f"CONFIRMED {int((m['label'] == ec.LABEL_POS).sum())}, FALSE POSITIVE {int((m['label'] == ec.LABEL_NEG).sum())}, CANDIDATE {int((m['label'] == ec.LABEL_CAND).sum())}"),
        ('минус Kepler-80 (витринная система)', len(m2), f"исключено сигналов: {len(k80)} ({k80['label'].value_counts().to_dict()})"),
        ('минус CANDIDATE (оцениваются отдельно)', len(lab), f"исключено сигналов: {len(cand)}"),
        ('Размеченных сигналов для E2–E4', len(lab), f"CONFIRMED {int((lab['label'] == ec.LABEL_POS).sum())} + FALSE POSITIVE {int((lab['label'] == ec.LABEL_NEG).sum())}; звёзд {lab['target_id'].nunique()}"),
    ]
    lines.append(ec.md_table(pd.DataFrame(rows, columns=['Шаг', 'Число', 'Пояснение'])))
    lines.append('\nРоли звёзд в файле разбиения:\n\n')
    lines.append(ec.md_table(stars['роль'].value_counts().rename_axis('Роль').reset_index(name='Звёзд')))
    lines.append('\nСлои размеченных звёзд:\n\n')
    lines.append(ec.md_table(stars[stars['слой'] != '']['слой'].value_counts().rename_axis('Слой').reset_index(name='Звёзд')))
    return ''.join(lines), {'labeled_signals': len(lab), 'labeled_pos': int((lab['label'] == ec.LABEL_POS).sum()), 'labeled_neg': int((lab['label'] == ec.LABEL_NEG).sum()), 'labeled_stars': int(lab['target_id'].nunique())}


def part_signals(features, stars):
    part_of = dict(zip(stars['target_id'], stars['часть_полная']))
    lab = features[ec.labeled_mask(features)].copy()
    lab['part'] = lab['target_id'].map(part_of)
    lab['y'] = (lab['label'] == ec.LABEL_POS).astype(int)
    return lab


def balance_check(lab):
    rows = []
    for col, name in BALANCE_COLS:
        for i, a in enumerate(ec.PARTS):
            for b in ec.PARTS[i + 1:]:
                xa = lab.loc[lab['part'] == a, col].dropna().values
                xb = lab.loc[lab['part'] == b, col].dropna().values
                res = ks_2samp(xa, xb)
                rows.append({'признак': name, 'часть A': a, 'часть B': b, 'n A': len(xa), 'n B': len(xb), 'медиана A': float(np.median(xa)), 'медиана B': float(np.median(xb)), 'КС D': float(res.statistic), 'p': float(res.pvalue)})
    return pd.DataFrame(rows)


def balance_md(bal, lab):
    lines = [ec.heading(1, 'E1. Проверка перекоса между частями разбиения')]
    lines.append('Сравнение по сигналам двухвыборочным критерием Колмогорова–Смирнова, все пары частей. Проверок много (5 признаков × 6 пар = 30), поправка на множественность не применялась; p < 0,05 при 30 проверках ожидаемо встречается и без перекоса (в среднем 1,5 раза).\n\n')
    show = bal.copy()
    for c in ['медиана A', 'медиана B']:
        show[c] = show[c].map(lambda v: ec.fmt_num(v, 2))
    show['КС D'] = show['КС D'].map(lambda v: ec.fmt_num(v, 3))
    show['p'] = show['p'].map(lambda v: ec.fmt_num(v, 3))
    lines.append(ec.md_table(show))
    lines.append('\nДоля класса 1 (CONFIRMED) по частям:\n\n')
    share = []
    for part in ec.PARTS:
        s = lab[lab['part'] == part]
        share.append({'часть': part, 'сигналов': len(s), 'звёзд': s['target_id'].nunique(), 'CONFIRMED': int(s['y'].sum()), 'доля класса 1': ec.fmt_num(s['y'].mean(), 3)})
    lines.append(ec.md_table(pd.DataFrame(share)))
    n_low = int((bal['p'] < 0.05).sum())
    lines.append(f"\nПроверок с p < 0,05: {n_low} из {len(bal)}. Разбиение не пересоздавалось.\n")
    return ''.join(lines)


def table1_md(lab, stars):
    rows = []
    for part in ec.PARTS:
        s = lab[lab['part'] == part]
        name = 'отложенный тест' if part == ec.PART_TEST else part
        rows.append({'Подмножество': name, 'Звёзд': s['target_id'].nunique(), 'Положительных': int(s['y'].sum()), 'Отрицательных': int((1 - s['y']).sum()), 'Назначение': PURPOSE[part]})
    rows.append({'Подмножество': 'итого', 'Звёзд': lab['target_id'].nunique(), 'Положительных': int(lab['y'].sum()), 'Отрицательных': int((1 - lab['y']).sum()), 'Назначение': 'размеченные сигналы MATCHED без Kepler-80 и без CANDIDATE'})
    n_test = int((lab['part'] == ec.PART_TEST).sum())
    text = ec.heading(1, 'Таблица 1. Разбиение размеченных сигналов по звёздам')
    text += 'Положительные — сигналы CONFIRMED, отрицательные — FALSE POSITIVE (только MATCHED). Звезда целиком в одной части; доли 50/15/15/20 % считаются по числу звёзд с размеченными сигналами, seed 42, стратификация по составу звезды (только CONFIRMED / только FALSE POSITIVE / оба класса).\n\n'
    text += ec.md_table(pd.DataFrame(rows))
    text += f"\nСигналов в отложенном тесте: {n_test}. Это меньше желаемых «нескольких сотен»: интервалы итоговых метрик на тесте будут широкими.\n"
    text += f"\nCANDIDATE-сигналы ({int(stars['CANDIDATE'].sum())}) и Kepler-80 (4 сигнала) в таблицу не входят.\n"
    return text


def git_out(args):
    try:
        return subprocess.run(['git'] + args, cwd=str(ec.ROOT), capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def rel_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ec.ROOT)).replace(chr(92), '/')
    except ValueError:
        return str(path).replace(chr(92), '/')


def provenance(run_dir, seed, fractions, flow, out_dir):
    files = {}
    for name, sub in INPUT_FILES:
        p = input_path(run_dir, name, sub)
        files[rel_path(p)] = {'sha256': ec.sha256_file(p), 'lines': ec.count_lines(p), 'bytes': p.stat().st_size}
    labels = ec.ROOT / 'data' / 'labels' / 'kepler_labels.csv'
    ldf = pd.read_csv(labels)
    log = git_out(['log', '-1', '--format=%H|%aI', '--', 'data/labels/kepler_labels.csv']).split('|')
    split_files = {}
    for name in [ec.SPLIT_STARS_FILE, TEST_IDS_FILE, CALIB_IDS_FILE, 'e1_signal_registry.csv', ec.FEATURE_LIST_FILE]:
        split_files[name] = ec.sha256_file(Path(out_dir) / name)
    f = ec.read_features(ec.features_dir(run_dir) / 'features_demo.csv')
    return {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'seed': seed, 'fractions': dict(zip(ec.PARTS, fractions)),
        'run_dir': rel_path(run_dir),
        'run_commit': (Path(run_dir) / 'run_commit.txt').read_text(encoding='utf-8').strip(),
        'repo_commit': git_out(['rev-parse', 'HEAD']), 'repo_branch': git_out(['rev-parse', '--abbrev-ref', 'HEAD']),
        'pipeline_version': sorted(f['pipeline_version'].astype(str).unique()),
        'input_files': files,
        'labels_file': {
            'path': 'data/labels/kepler_labels.csv', 'source': 'NASA Exoplanet Archive, таблица cumulative (KOI)',
            'access_date_note': 'дата последнего коммита файла в git; архив не запрашивался',
            'last_commit': log[0] if log else '', 'last_commit_date': log[1] if len(log) > 1 else '',
            'sha256_working_copy': ec.sha256_file(labels), 'rows': int(len(ldf)), 'columns': list(ldf.columns),
            'dispositions': {str(k): int(v) for k, v in ldf['label'].value_counts().items()},
        },
        'flow': flow, 'outputs_sha256': split_files, 'packages': ec.package_versions(),
        'showcase_excluded': {'target_id': ec.SHOWCASE_ID, 'reason': 'витринная система'},
    }


def load_diag(run_dir):
    d = pd.read_csv(ec.features_dir(run_dir) / 'diagnostics_stars.csv', dtype={'target_id': str})
    kind_col = 'тип' if 'тип' in d.columns else 'star_type'
    kinds = d[kind_col].astype(str)
    d['star_kind'] = np.where(kinds.str.startswith('Отмеченная') | (kinds == 'MARKED'), 'MARKED', 'UNMARKED')
    return d


def run(run_dir, out_dir, seed=ec.SEED, fractions=FRACTIONS, identity_check=True):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if identity_check:
        check_identity(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fd = ec.features_dir(run_dir)
    features = ec.read_features(fd / 'features_demo.csv')
    diag = load_diag(run_dir)
    failed = pd.read_csv(fd / 'features_demo_failed.csv', dtype={'target_id': str})
    stars = build_star_table(features, diag, set(failed['target_id']), seed, fractions)
    public = stars.copy()
    public['часть'] = public['часть_полная'].map(public_part)
    public = public.drop(columns=['часть_полная'])
    write_csv(public, out_dir / ec.SPLIT_STARS_FILE)
    for part, name in [(ec.PART_TEST, TEST_IDS_FILE), (ec.PART_CALIB, CALIB_IDS_FILE)]:
        write_csv(stars.loc[stars['часть_полная'] == part, ['target_id', 'роль', 'слой', 'способ']], out_dir / name)
    registry = build_registry(features, stars)
    write_csv(registry, out_dir / 'e1_signal_registry.csv')
    ec.write_json(out_dir / ec.FEATURE_LIST_FILE, {'source': 'onyx_features.compute_feature_columns(df, exclude_coverage=True), как в 3_demo_pipeline.py', 'features': ec.model_feature_columns(features)})
    flow_text, flow = flow_table(features, diag, failed, stars)
    ec.write_text(out_dir / 'e1_flow_table.md', flow_text)
    lab = part_signals(features, stars)
    bal = balance_check(lab)
    write_csv(bal, out_dir / 'e1_balance.csv')
    ec.write_text(out_dir / 'e1_balance.md', balance_md(bal, lab))
    ec.write_text(out_dir / 'table1.md', table1_md(lab, stars))
    prov = provenance(run_dir, seed, list(fractions), flow, out_dir)
    ec.write_json(out_dir / 'e1_provenance.json', prov)
    return stars, registry, lab


def parse_fractions(text):
    vals = tuple(float(v) for v in text.split(','))
    if len(vals) != 4 or abs(sum(vals) - 1.0) > 1e-9 or min(vals) <= 0:
        raise argparse.ArgumentTypeError('four positive fractions that sum to 1')
    return vals


def main(argv=None):
    ap = argparse.ArgumentParser(description='E1: data audit and star-level split of the 2000-star run')
    ap.add_argument('--run-dir', default=str(ec.RUN_DIR))
    ap.add_argument('--out-dir', default=str(ec.OUT_DIR))
    ap.add_argument('--seed', type=int, default=ec.SEED)
    ap.add_argument('--fractions', type=parse_fractions, default=FRACTIONS)
    ap.add_argument('--no-identity-check', action='store_true')
    args = ap.parse_args(argv)
    stars, registry, lab = run(args.run_dir, args.out_dir, args.seed, args.fractions, identity_check=not args.no_identity_check)
    for part in ec.PARTS:
        s = lab[lab['part'] == part]
        print(f"{part}: stars={s['target_id'].nunique()} signals={len(s)} pos={int(s['y'].sum())} neg={int((1 - s['y']).sum())}")
    print('registry rows', len(registry), 'labeled', int(registry['в_разметке'].sum()))


if __name__ == '__main__':
    main()
