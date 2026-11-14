import argparse
import json
import os
import sys
import warnings
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e2_compare as e2
import exp_common as ec

PERIOD_BINS = [(0.5, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, 100.0)]
MIN_BIN = 20
HIDDEN_PART_FILES = {ec.PART_CALIB: 'split_calib_ids.csv', ec.PART_TEST: 'split_test_ids.csv'}


def fit_full_train(family, params, df_tr, cols):
    X = df_tr[cols].values.astype(float)
    y = df_tr['y'].values.astype(int)
    g = df_tr['target_id'].astype(str).values
    model, _ = e2.fit_model(family, params, X, y, g, ec.SEED)
    return model


def validation_table(out, run, best_params, chosen, n_boot):
    df_tr, cols = e2.load_labeled_part(out, run, ec.PART_TRAIN)
    df_va, _ = e2.load_labeled_part(out, run, ec.PART_VAL)
    yv = df_va['y'].values.astype(int)
    gv = df_va['target_id'].astype(str).values
    preds = {}
    models = {}
    for f in e2.FAMILIES:
        models[f] = fit_full_train(f, best_params[f], df_tr, cols)
        preds[f] = models[f].predict_all(df_va[cols].values.astype(float))[0]
    draws = ec.bootstrap_metrics(yv, {f: [preds[f]] for f in e2.FAMILIES}, gv, n_boot=n_boot, seed=ec.SEED)
    tb = np.arange(len(yv))
    point = {f: ec.all_metrics(yv, preds[f], tb) for f in e2.FAMILIES}
    rows = []
    for f in e2.FAMILIES:
        s = point[f]
        rows.append({'Алгоритм': e2.NAMES[f] + (' (выбрана в E2)' if f == chosen else ''), 'ROC-AUC': ec.fmt_ci(s['roc_auc'], ec.percentile_ci(draws[f]['roc_auc'])),
                     f"AP (доля класса 1 = {ec.fmt_num(float(yv.mean()), 3)})": ec.fmt_ci(s['ap'], ec.percentile_ci(draws[f]['ap'])),
                     'Risk@50%': ec.fmt_ci(s['risk@0.50'], ec.percentile_ci(draws[f]['risk@0.50'])), 'Risk@70%': ec.fmt_ci(s[e2.RISK_KEY], ec.percentile_ci(draws[f][e2.RISK_KEY])),
                     'Risk@90%': ec.fmt_ci(s['risk@0.90'], ec.percentile_ci(draws[f]['risk@0.90'])), 'AURC': ec.fmt_ci(s['aurc'], ec.percentile_ci(draws[f]['aurc'])),
                     'AUGRC': ec.fmt_ci(s['augrc'], ec.percentile_ci(draws[f]['augrc'])), 'Brier': ec.fmt_ci(s['brier'], ec.percentile_ci(draws[f]['brier']))})
    val_leader = min(e2.FAMILIES, key=lambda f: (point[f][e2.RISK_KEY], e2.SIMPLICITY[f]))
    diff_rows = []
    for f in e2.FAMILIES:
        lo_hi = e2.paired_ci(draws, f, chosen, e2.RISK_KEY)
        a = e2.paired_ci(draws, f, chosen, 'aurc')
        diff_rows.append({'Алгоритм': e2.NAMES[f], 'ΔRisk@70% к выбранной': ec.fmt_ci(point[f][e2.RISK_KEY] - point[chosen][e2.RISK_KEY], lo_hi), 'ΔAURC к выбранной': ec.fmt_ci(point[f]['aurc'] - point[chosen]['aurc'], a)})
    lo, hi = e2.paired_ci(draws, chosen, val_leader, e2.RISK_KEY)
    lines = [ec.heading(1, 'Таблица 2б. Проверка семейств на валидационной части')]
    lines.append(f"Каждое семейство обучено на всей обучающей части ({len(df_tr)} сигналов, {df_tr['target_id'].nunique()} звёзд) с гиперпараметрами, подобранными в E2 на обучающей части (без нового подбора), и оценено на валидационной части: {len(df_va)} сигналов, {df_va['target_id'].nunique()} звёзд (CONFIRMED {int(yv.sum())}, FALSE POSITIVE {int(len(yv) - yv.sum())}). В скобках 95% бутстреп-интервал по звёздам (1000 повторов). При таком размере выборки интервалы широки, и различия между семействами в несколько сотых ими не разрешаются.\n\n")
    lines.append(ec.md_table(pd.DataFrame(rows)))
    lines.append('\nПарные разности к выбранной модели (тот же бутстреп по звёздам, модель минус выбранная):\n\n')
    lines.append(ec.md_table(pd.DataFrame(diff_rows)))
    best_risk = point[val_leader][e2.RISK_KEY]
    at_best = [f for f in e2.FAMILIES if point[f][e2.RISK_KEY] == best_risk]
    if chosen in at_best:
        verdict = f"На валидационной части наименьший risk@70% ({ec.fmt_num(best_risk)}) у моделей: " + ', '.join(e2.NAMES[f] for f in at_best) + f". Выбранная модель ({e2.NAMES[chosen]}) среди них; противоречия с таблицей 2 нет, но и различить эти модели по risk@70% на валидационной части нельзя."
    elif lo <= 0 <= hi:
        verdict = f"На валидационной части наименьший risk@70% у модели «{e2.NAMES[val_leader]}», но парный интервал разности с выбранной моделью ({ec.fmt_num(lo)}; {ec.fmt_num(hi)}) включает 0: противоречия с таблицей 2, различимого на этой выборке, нет."
    else:
        verdict = f"На валидационной части наименьший risk@70% у модели «{e2.NAMES[val_leader]}», и парный интервал разности risk@70% выбранной модели к ней ({ec.fmt_num(lo)}; {ec.fmt_num(hi)}) не включает 0: это противоречит выбору по таблице 2. Решение не меняется."
    lines.append('\n' + verdict + '\n')
    return ''.join(lines), point, draws, models, cols, val_leader, (lo, hi)


def bin_stats(sub):
    y = sub['y'].values
    p = sub['p_avg'].values
    n = len(sub)
    ok = n >= MIN_BIN and len(np.unique(y)) == 2
    return {'Сигналов': n, 'CONFIRMED': int(y.sum()), 'FALSE POSITIVE': int(n - y.sum()),
            'ROC-AUC': ec.fmt_num(ec.roc_auc(y, p), 3) if ok else 'не оценивается', 'Точность': ec.fmt_num(ec.accuracy(y, p), 3) if ok else 'не оценивается'}


def bins_md(out, run, chosen):
    oof = pd.read_csv(Path(out) / 'e2_oof_predictions.csv', dtype={'target_id': str, 'signal_id': str})
    oof = oof[oof['model'] == chosen]
    df_tr, _ = e2.load_labeled_part(out, run, ec.PART_TRAIN)
    d = df_tr[['signal_id', 'bls_period_d', 'bls_depth_ppm']].merge(oof[['signal_id', 'y', 'p_avg']], on='signal_id', how='inner')
    if len(d) != len(df_tr):
        raise SystemExit('OOF predictions do not cover the training part')
    lines = [ec.heading(1, f"E2. Эксплораторные корзины: {e2.NAMES[chosen]}")]
    lines.append(f"Внекорзинные предсказания выбранной модели на обучающей части (p усреднено по 5 повторам вложенной проверки), {len(d)} сигналов. Корзины не влияли на выбор модели. Сигналов в корзинах мало; корзина с числом сигналов меньше {MIN_BIN} или с одним классом помечена «не оценивается». Точность — доля верных при пороге 0,5 без отказа.\n\n")
    rows = []
    for lo, hi in PERIOD_BINS:
        sub = d[(d['bls_period_d'] >= lo) & ((d['bls_period_d'] < hi) if hi < 100 else (d['bls_period_d'] <= hi))]
        rows.append(dict({'Период, сут': f"{lo:g}–{hi:g}".replace('.', ',')}, **bin_stats(sub)))
    outside = d[(d['bls_period_d'] < PERIOD_BINS[0][0]) | (d['bls_period_d'] > PERIOD_BINS[-1][1])]
    lines.append(ec.heading(2, 'По периоду BLS'))
    lines.append(ec.md_table(pd.DataFrame(rows)))
    if len(outside):
        lines.append(f"\nВне корзин периода: {len(outside)} сигналов.\n")
    edges = np.quantile(d['bls_depth_ppm'].values, [0, 0.25, 0.5, 0.75, 1.0])
    q = pd.cut(d['bls_depth_ppm'], bins=edges, include_lowest=True, labels=False)
    rows = []
    for k in range(4):
        sub = d[q == k]
        rows.append(dict({'Квартиль глубины': f"Q{k + 1}", 'Глубина, ppm': f"{ec.fmt_num(edges[k], 0)}–{ec.fmt_num(edges[k + 1], 0)}"}, **bin_stats(sub)))
    lines.append('\n' + ec.heading(2, 'По глубине BLS (квартили размеченных сигналов обучающей части)'))
    lines.append(ec.md_table(pd.DataFrame(rows)))
    overall = bin_stats(d)
    lines.append(f"\nВся обучающая часть: ROC-AUC {overall['ROC-AUC']}, точность {overall['Точность']}.\n")
    return ''.join(lines)


def star_parts(out):
    stars = pd.read_csv(Path(out) / ec.SPLIT_STARS_FILE, dtype={'target_id': str})
    part = dict(zip(stars['target_id'], stars['часть'].fillna('')))
    for name, fname in HIDDEN_PART_FILES.items():
        ids = pd.read_csv(Path(out) / fname, dtype={'target_id': str})['target_id']
        for t in ids:
            part[t] = name
    return part, dict(zip(stars['target_id'], stars['роль']))


def candidates(out, run, model, cols, chosen):
    f = ec.read_features(ec.features_dir(run) / 'features_demo.csv')
    cand = f[ec.candidate_mask(f)].copy()
    p, s = model.predict_all(cand[cols].values.astype(float))
    part, role = star_parts(out)
    res = pd.DataFrame({'signal_id': cand['signal_id'].values, 'target_id': cand['target_id'].values, 'p': p, 'часть_звезды': [part.get(t, '') for t in cand['target_id']], 'роль_звезды': [role.get(t, '') for t in cand['target_id']]})
    if s is not None:
        res['s'] = s
    res.to_csv(Path(out) / 'candidate_scores.csv', index=False, encoding='utf-8', lineterminator='\n', float_format='%.10g')
    q = np.quantile(p, [0.25, 0.5, 0.75])
    lines = [ec.heading(1, f"E2. Оценки модели на сигналах CANDIDATE: {e2.NAMES[chosen]}")]
    lines.append('У этих сигналов нет истинных меток (диспозиция KOI — CANDIDATE), поэтому приведённые числа — распределение оценок модели, а не точность. В обучении, подборе и метриках эти сигналы не участвовали. Пороги отказа не применялись (они определятся в E3).\n\n')
    lines.append(f"Модель обучена на всей обучающей части с гиперпараметрами из E2. Сигналов CANDIDATE: {len(res)} на {res['target_id'].nunique()} звёздах.\n\n")
    lines.append(f"- Доля p ≥ 0,5: {ec.fmt_num(float(np.mean(p >= 0.5)), 3)} ({int(np.sum(p >= 0.5))} из {len(p)}).\n")
    lines.append(f"- Квартили p: Q1 = {ec.fmt_num(q[0])}, медиана = {ec.fmt_num(q[1])}, Q3 = {ec.fmt_num(q[2])}; минимум {ec.fmt_num(float(p.min()))}, максимум {ec.fmt_num(float(p.max()))}.\n")
    if s is not None:
        lines.append(f"- Межмодельный разброс s: медиана {ec.fmt_num(float(np.median(s)))}.\n")
    lines.append('\nЧисло сигналов CANDIDATE по частям разбиения звезды:\n\n')
    tab = res.groupby('часть_звезды').agg(сигналов=('p', 'size'), звёзд=('target_id', 'nunique'), доля_p_не_меньше_05=('p', lambda v: ec.fmt_num(float(np.mean(v >= 0.5)), 3))).reset_index()
    tab = tab.rename(columns={'часть_звезды': 'Часть', 'доля_p_не_меньше_05': 'доля p ≥ 0,5'})
    lines.append(ec.md_table(tab))
    lines.append('\nЧасть звезды для звёзд без размеченных сигналов (только CANDIDATE) назначена хешем от target_id и seed (E1). Для звёзд калибровочной и тестовой частей части взяты из split_calib_ids.csv и split_test_ids.csv только для подсчёта; меток этих звёзд скрипт не читает.\n')
    return ''.join(lines), res


def main(argv=None):
    ap = argparse.ArgumentParser(description='E2 steps 3-5: validation check, exploratory bins, CANDIDATE scores')
    ap.add_argument('--run-dir', default=str(ec.RUN_DIR))
    ap.add_argument('--out-dir', default=str(ec.OUT_DIR))
    ap.add_argument('--n-boot', type=int, default=e2.N_BOOT)
    args = ap.parse_args(argv)
    warnings.filterwarnings('ignore')
    out = Path(args.out_dir)
    summary = json.loads((out / 'e2_summary.json').read_text(encoding='utf-8'))
    chosen = summary['chosen']
    best = json.loads((out / 'e2_best_params.json').read_text(encoding='utf-8'))['params']
    text, point, draws, models, cols, val_leader, ci = validation_table(out, args.run_dir, best, chosen, args.n_boot)
    ec.write_text(out / 'table2b.md', text)
    ec.write_json(out / 'e2_validation.json', {'chosen': chosen, 'val_leader_risk70': val_leader, 'chosen_minus_leader_risk70_ci': list(ci), 'point': point,
                                                'ci': {f: {k: ec.percentile_ci(v) for k, v in d.items()} for f, d in draws.items()}})
    ec.write_text(out / 'e2_bins.md', bins_md(out, args.run_dir, chosen))
    ctext, _ = candidates(out, args.run_dir, models[chosen], cols, chosen)
    ec.write_text(out / 'candidate_summary.md', ctext)
    print('chosen', chosen, 'validation leader', val_leader, 'ci', ci)


if __name__ == '__main__':
    main()
