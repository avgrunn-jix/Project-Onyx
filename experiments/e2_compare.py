import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_common as ec

FAMILIES = ['lr', 'rf', 'xgb', 'lgbm', 'svm', 'mlp', 'xgb_ens']
TUNED = FAMILIES[:-1]
NAMES = {
    'lr': 'Логистическая регрессия (L2)', 'rf': 'Random Forest', 'xgb': 'XGBoost (одиночная модель)',
    'lgbm': 'LightGBM', 'svm': 'SVM (RBF-ядро)', 'mlp': 'MLP (компактная)', 'xgb_ens': 'XGBoost, ансамбль (текущая система)',
}
SIMPLICITY = {f: i for i, f in enumerate(FAMILIES)}
N_TRIALS = 12
INNER_SPLITS = 3
OUTER_SPLITS = 5
REPEAT_SEEDS = [42, 43, 44, 45, 46]
N_BOOT = 1000
TIMING_REPEATS = 5
C0 = 0.70
RISK_KEY = 'risk@0.70'
N_QUANTILES = 100
SPACES = {
    'rf': {'n_estimators': [100, 200, 400], 'max_depth': [None, 3, 5, 8], 'min_samples_leaf': [1, 2, 5, 10], 'max_features': ['sqrt', 0.3, 0.5]},
    'xgb': {'n_estimators': [50, 100, 200, 400], 'max_depth': [2, 3, 4, 6], 'learning_rate': loguniform(0.01, 0.3), 'subsample': [0.7, 0.85, 1.0], 'colsample_bytree': [0.5, 0.75, 1.0], 'min_child_weight': [1, 3, 5], 'reg_lambda': [1.0, 5.0]},
    'lgbm': {'n_estimators': [50, 100, 200, 400], 'num_leaves': [4, 8, 16, 31], 'learning_rate': loguniform(0.01, 0.3), 'min_child_samples': [5, 10, 20], 'subsample': [0.7, 0.85, 1.0], 'colsample_bytree': [0.5, 0.75, 1.0], 'reg_lambda': [0.0, 1.0, 5.0]},
    'svm': {'C': loguniform(0.1, 100.0), 'gamma': loguniform(3e-4, 0.3)},
    'mlp': {'hidden_layer_sizes': [(8,), (16,), (32,), (16, 8), (32, 16)], 'alpha': loguniform(1e-4, 10.0), 'activation': ['relu', 'tanh']},
}
DECISION_RULE = [
    'Кандидаты — модели, недоминируемые по Парето по четырём осям: risk при охвате 0,70 (меньше лучше), AURC (меньше лучше), стандартное отклонение ROC-AUC по внешним фолдам (меньше лучше), медианное время обучения одной модели (меньше лучше).',
    'Лидер — кандидат с минимальным risk при охвате C0 = 0,70 (точечная оценка, среднее по 5 повторам); при равенстве — меньший разброс ROC-AUC по фолдам; затем меньшее время обучения.',
    'Неотличимые от лидера — кандидаты, у которых 95% интервал парного бутстрепа по звёздам для разности risk@0,70 (модель минус лидер) включает 0.',
    'Если неотличимых (включая лидера) больше одного, выбирается самая простая по порядку: логистическая регрессия < Random Forest < XGBoost < LightGBM < SVM < MLP < ансамбль XGBoost. Иначе выбирается лидер.',
]


def py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return [py(x) for x in v]
    return v


def trial_configs(family):
    if family == 'lr':
        return [{'C': float(c)} for c in np.logspace(-3, 2, N_TRIALS)]
    if family == 'xgb_ens':
        return []
    out = []
    for p in ParameterSampler(SPACES[family], n_iter=N_TRIALS, random_state=ec.SEED):
        out.append({k: (tuple(v) if isinstance(v, (list, tuple)) else py(v)) for k, v in sorted(p.items())})
    return out


def complexity(family, params, n_features=38):
    if family == 'lr':
        return (params['C'],)
    if family == 'rf':
        mf = params['max_features']
        mf = 1.0 / np.sqrt(n_features) if mf == 'sqrt' else float(mf)
        depth = 99 if params['max_depth'] is None else params['max_depth']
        return (depth, params['n_estimators'], -params['min_samples_leaf'], mf)
    if family == 'xgb':
        return (params['max_depth'], params['n_estimators'], -params['min_child_weight'], -params['reg_lambda'])
    if family == 'lgbm':
        return (params['num_leaves'], params['n_estimators'], -params['min_child_samples'], -params['reg_lambda'])
    if family == 'svm':
        return (params['C'], params['gamma'])
    if family == 'mlp':
        sizes = [n_features] + list(params['hidden_layer_sizes']) + [1]
        return (sum(a * b for a, b in zip(sizes[:-1], sizes[1:])), -params['alpha'])
    return (0,)


def imputer():
    return SimpleImputer(strategy='median')


def quantile():
    return QuantileTransformer(n_quantiles=N_QUANTILES, output_distribution='normal', random_state=0)


class EnsembleModel:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y, groups):
        demo = ec.load_demo_module()
        self.pipe = demo.get_ml_pipeline()
        Xs = self.pipe.fit_transform(X)
        self.n_models = min(20, max(5, len(y) * 2))
        self.models = demo.fit_bootstrap_ensemble(Xs, np.asarray(y).astype(int), np.asarray(groups).astype(str), self.seed, self.n_models)
        return self

    def predict_all(self, X):
        Xs = self.pipe.transform(X)
        preds = np.array([m.predict_proba(Xs)[:, 1] for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0)


class SklearnModel:
    def __init__(self, est):
        self.est = est

    def predict_all(self, X):
        return self.est.predict_proba(X)[:, 1], None


def build_estimator(family, params, seed):
    if family == 'lr':
        return Pipeline([('imp', imputer()), ('qt', quantile()), ('m', LogisticRegression(C=params['C'], max_iter=5000))])
    if family == 'rf':
        return Pipeline([('imp', imputer()), ('m', RandomForestClassifier(random_state=seed, n_jobs=1, **params))])
    if family == 'xgb':
        from xgboost import XGBClassifier
        return Pipeline([('imp', imputer()), ('m', XGBClassifier(random_state=seed, n_jobs=1, eval_metric='logloss', tree_method='hist', **params))])
    if family == 'lgbm':
        from lightgbm import LGBMClassifier
        return Pipeline([('imp', imputer()), ('m', LGBMClassifier(random_state=seed, n_jobs=1, subsample_freq=1, verbose=-1, **params))])
    if family == 'svm':
        return Pipeline([('imp', imputer()), ('qt', quantile()), ('m', SVC(kernel='rbf', C=params['C'], gamma=params['gamma'], random_state=seed))])
    if family == 'mlp':
        return Pipeline([('imp', imputer()), ('qt', quantile()), ('m', MLPClassifier(solver='lbfgs', max_iter=2000, random_state=seed, hidden_layer_sizes=tuple(params['hidden_layer_sizes']), alpha=params['alpha'], activation=params['activation']))])
    raise ValueError(family)


def group_splits(y, groups, n_splits, seed):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def fit_model(family, params, X, y, groups, seed):
    y = np.asarray(y).astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if family == 'xgb_ens':
            model = EnsembleModel(seed).fit(X, y, groups)
        elif family == 'svm':
            cal = CalibratedClassifierCV(build_estimator('svm', params, seed), method='sigmoid', cv=group_splits(y, groups, INNER_SPLITS, seed))
            cal.fit(X, y)
            model = SklearnModel(cal)
        else:
            est = build_estimator(family, params, seed)
            est.fit(X, y)
            model = SklearnModel(est)
    n_conv = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return model, n_conv


def tune(family, X, y, groups, seed):
    configs = trial_configs(family)
    if not configs:
        return {}, [], 0
    y = np.asarray(y).astype(int)
    splits = group_splits(y, groups, INNER_SPLITS, seed)
    trials = []
    n_conv = 0
    for i, params in enumerate(configs):
        oof = np.zeros(len(y))
        for tr, va in splits:
            model, c = fit_model(family, params, X[tr], y[tr], groups[tr], seed)
            n_conv += c
            oof[va] = model.predict_all(X[va])[0]
        trials.append({'trial': i, 'params': {k: py(v) for k, v in params.items()}, 'logloss': float(log_loss(y, np.clip(oof, 1e-15, 1 - 1e-15), labels=[0, 1]))})
    best = min(range(len(configs)), key=lambda i: (round(trials[i]['logloss'], 6), complexity(family, configs[i], X.shape[1])))
    return configs[best], trials, n_conv


def load_labeled_part(out_dir, run_dir, part):
    if part not in (ec.PART_TRAIN, ec.PART_VAL):
        raise ValueError('only the training and validation parts can be loaded here')
    stars = pd.read_csv(Path(out_dir) / ec.SPLIT_STARS_FILE, dtype={'target_id': str})
    ids = set(stars.loc[(stars['часть'] == part) & (stars['роль'] == 'размеченная'), 'target_id'])
    f = ec.read_features(ec.features_dir(run_dir) / 'features_demo.csv')
    df = f[ec.labeled_mask(f) & f['target_id'].isin(ids)].copy()
    df['y'] = (df['label'] == ec.LABEL_POS).astype(int)
    df = df.sort_values(['target_id', 'signal_index'], key=lambda s: s.astype(int) if s.name == 'target_id' else s).reset_index(drop=True)
    cols = ec.model_feature_columns(f)
    return df, cols


def dataset_fingerprint(df, cols):
    h = hashlib.sha256()
    h.update('|'.join(df['signal_id'].astype(str)).encode('utf-8'))
    h.update('|'.join(cols).encode('utf-8'))
    h.update(np.ascontiguousarray(df[cols].values.astype(float)).tobytes())
    return h.hexdigest()[:16]


def outer_task(family, seed, fold, tr, te, X, y, groups):
    t0 = time.perf_counter()
    if family == 'xgb_ens':
        best, trials, n_conv = {}, [], 0
    else:
        best, trials, n_conv = tune(family, X[tr], y[tr], groups[tr], seed)
    model, c = fit_model(family, best, X[tr], y[tr], groups[tr], seed)
    p, s = model.predict_all(X[te])
    return {
        'family': family, 'seed': seed, 'fold': fold, 'test_idx': [int(i) for i in te], 'p': [float(v) for v in p],
        's': [float(v) for v in s] if s is not None else None, 'best': {k: py(v) for k, v in best.items()},
        'trials': trials, 'n_conv_warnings': n_conv + c, 'seconds': time.perf_counter() - t0,
    }


def run_outer(X, y, groups, cache_dir, fingerprint, workers, families):
    from joblib import Parallel, delayed
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    results = []
    for seed in REPEAT_SEEDS:
        for fold, (tr, te) in enumerate(group_splits(y, groups, OUTER_SPLITS, seed)):
            for family in families:
                path = cache_dir / f"{family}_{seed}_{fold}.json"
                if path.exists():
                    rec = json.loads(path.read_text(encoding='utf-8'))
                    if rec.get('fingerprint') == fingerprint and rec['test_idx'] == [int(i) for i in te]:
                        results.append(rec)
                        continue
                jobs.append((family, seed, fold, tr, te, path))
    print(f"outer tasks: cached {len(results)}, to run {len(jobs)}", flush=True)

    def run_one(job):
        family, seed, fold, tr, te, path = job
        warnings.filterwarnings('ignore')
        rec = outer_task(family, seed, fold, tr, te, X, y, groups)
        rec['fingerprint'] = fingerprint
        path.write_text(json.dumps(rec), encoding='utf-8')
        return rec

    done = Parallel(n_jobs=workers, backend='loky', verbose=5)(delayed(run_one)(j) for j in jobs)
    return results + list(done)


def collect_oof(results, n):
    p = {f: {s: np.full(n, np.nan) for s in REPEAT_SEEDS} for f in FAMILIES}
    s_ens = {s: np.full(n, np.nan) for s in REPEAT_SEEDS}
    fold_of = {s: np.full(n, -1) for s in REPEAT_SEEDS}
    for r in results:
        idx = np.asarray(r['test_idx'])
        p[r['family']][r['seed']][idx] = r['p']
        fold_of[r['seed']][idx] = r['fold']
        if r['s'] is not None:
            s_ens[r['seed']][idx] = r['s']
    return p, s_ens, fold_of


def fold_auc_spread(y, p_by_seed, fold_of):
    per_seed = []
    all_aucs = []
    for s in REPEAT_SEEDS:
        aucs = [ec.roc_auc(y[fold_of[s] == k], p_by_seed[s][fold_of[s] == k]) for k in range(OUTER_SPLITS)]
        all_aucs.extend(aucs)
        per_seed.append(float(np.std(aucs, ddof=1)))
    return float(np.mean(per_seed)), all_aucs


def measure_time(family, params, X, y, groups):
    from threadpoolctl import threadpool_limits
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        proc = None
    fit_t, pred_t, mem = [], [], []
    with threadpool_limits(limits=1):
        for _ in range(TIMING_REPEATS):
            rss0 = proc.memory_info().rss if proc else None
            t0 = time.perf_counter()
            model, _ = fit_model(family, params, X, y, groups, ec.SEED)
            fit_t.append(time.perf_counter() - t0)
            if proc:
                mem.append((proc.memory_info().rss - rss0) / 2 ** 20)
            t0 = time.perf_counter()
            model.predict_all(X)
            pred_t.append(time.perf_counter() - t0)
    return {'fit_median_s': float(np.median(fit_t)), 'fit_min_s': float(np.min(fit_t)), 'fit_max_s': float(np.max(fit_t)),
            'predict_median_s': float(np.median(pred_t)), 'memory_mb': float(np.median(mem)) if mem else None}


def cpu_info():
    name = platform.processor()
    try:
        import subprocess
        out = subprocess.run(['wmic', 'cpu', 'get', 'name'], capture_output=True, text=True, timeout=20).stdout.split('\n')
        lines = [ln.strip() for ln in out if ln.strip() and ln.strip().lower() != 'name']
        if lines:
            name = lines[0]
    except Exception:
        pass
    return {'processor': name, 'logical_cores': os.cpu_count(), 'threads_used': 1}


def pareto_front(summary):
    axes = [RISK_KEY, 'aurc', 'fold_auc_std', 'fit_median_s']
    front = []
    for m in summary:
        dominated = False
        for o in summary:
            if o == m:
                continue
            le = all(summary[o][a] <= summary[m][a] for a in axes)
            lt = any(summary[o][a] < summary[m][a] for a in axes)
            if le and lt:
                dominated = True
                break
        if not dominated:
            front.append(m)
    return front


def paired_ci(draws, m, ref, key):
    d = draws[m][key] - draws[ref][key]
    return ec.percentile_ci(d)


def decide(summary, draws, front):
    leader = min(front, key=lambda m: (summary[m][RISK_KEY], summary[m]['fold_auc_std'], summary[m]['fit_median_s']))
    tied = []
    for m in front:
        if m == leader:
            tied.append(m)
            continue
        lo, hi = paired_ci(draws, m, leader, RISK_KEY)
        if lo <= 0 <= hi:
            tied.append(m)
    chosen = min(tied, key=lambda m: SIMPLICITY[m]) if len(tied) > 1 else leader
    return leader, tied, chosen


def risk_coverage_figure(y, p_avg, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = [(0.165, 0.471, 0.839), (0.922, 0.408, 0.204), (0.106, 0.686, 0.478), (0.929, 0.631, 0.000), (0.910, 0.482, 0.643), (0.000, 0.514, 0.000), (0.384, 0.314, 0.839)]
    styles = ['-', '--', '-.', ':', '-', '--', '-']
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    ax.axvline(C0, color=(0.710, 0.706, 0.678), lw=1, zorder=1)
    ax.text(C0 + 0.01, 0.98, 'C0 = 0,70', transform=ax.get_xaxis_transform(), va='top', fontsize=8, color=(0.333, 0.329, 0.306))
    for i, f in enumerate(FAMILIES):
        cov, risk = ec.risk_coverage_curve(y, p_avg[f])
        ax.plot(cov, risk, color=colors[i], ls=styles[i], lw=2 if f == 'xgb_ens' else 1.6, label=NAMES[f], zorder=3)
    ax.set_xlabel('Охват (доля принятых сигналов)')
    ax.set_ylabel('Риск (доля ошибок среди принятых)')
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, color=(0.902, 0.898, 0.878), lw=0.6)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    ax.set_title('Кривые «риск–охват», внекорзинные предсказания обучающей части\n(среднее по 5 повторам вложенной проверки)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(df, p, s_ens, fold_of, timing, draws, results):
    y = df['y'].values
    tb = np.arange(len(y))
    summary = {}
    long_rows = []
    for f in FAMILIES:
        per = [ec.all_metrics(y, p[f][s], tb) for s in REPEAT_SEEDS]
        point = {k: float(np.mean([v[k] for v in per])) for k in ec.metric_names()}
        avg = ec.all_metrics(y, np.mean([p[f][s] for s in REPEAT_SEEDS], axis=0), tb)
        spread, fold_aucs = fold_auc_spread(y, p[f], fold_of)
        n_trials = len(trial_configs(f))
        n_conv = sum(r['n_conv_warnings'] for r in results if r['family'] == f)
        summary[f] = dict(point, fold_auc_std=spread, fold_auc_min=float(np.nanmin(fold_aucs)), fold_auc_max=float(np.nanmax(fold_aucs)), n_trials=n_trials, n_conv_warnings=n_conv, **timing[f])
        if f == 'xgb_ens':
            sv = np.concatenate([s_ens[s] for s in REPEAT_SEEDS])
            summary[f]['s_mean'] = float(np.nanmean(sv))
            summary[f]['s_median'] = float(np.nanmedian(sv))
        for k in ec.metric_names():
            ci = ec.percentile_ci(draws[f][k])
            long_rows.append({'model': f, 'name': NAMES[f], 'metric': k, 'estimator': 'mean_over_repeats', 'value': point[k], 'ci_low': ci[0], 'ci_high': ci[1]})
            long_rows.append({'model': f, 'name': NAMES[f], 'metric': k, 'estimator': 'averaged_predictions', 'value': avg[k], 'ci_low': np.nan, 'ci_high': np.nan})
            for s, v in zip(REPEAT_SEEDS, per):
                long_rows.append({'model': f, 'name': NAMES[f], 'metric': k, 'estimator': f"seed_{s}", 'value': v[k], 'ci_low': np.nan, 'ci_high': np.nan})
        for k in ['fold_auc_std', 'fold_auc_min', 'fold_auc_max', 'n_trials', 'n_conv_warnings', 'fit_median_s', 'fit_min_s', 'fit_max_s', 'predict_median_s', 'memory_mb', 's_mean', 's_median']:
            if k in summary[f]:
                long_rows.append({'model': f, 'name': NAMES[f], 'metric': k, 'estimator': 'value', 'value': summary[f][k], 'ci_low': np.nan, 'ci_high': np.nan})
    return summary, pd.DataFrame(long_rows)


def ci_of(draws, f, k):
    return ec.percentile_ci(draws[f][k])


def table2_md(df, summary, draws, front, best_auc, best_aurc, indist, cpu, final_params):
    y = df['y'].values
    pos_share = float(np.mean(y))
    lines = [ec.heading(1, 'Таблица 2. Сравнение семейств моделей на обучающей части (вложенная перекрёстная проверка)')]
    lines.append(f"Данные: только звёзды части «обучение» — {df['target_id'].nunique()} звёзд, {len(df)} размеченных сигналов (CONFIRMED {int(y.sum())}, FALSE POSITIVE {int(len(y) - y.sum())}); калибровочная и тестовая части не открывались. Внешняя проверка StratifiedGroupKFold(5) по звёздам, 5 повторов (seed 42–46); подбор гиперпараметров только внутри обучающей части внешнего фолда (StratifiedGroupKFold(3) по звёздам, критерий logloss, 12 испытаний). Точечные оценки — среднее по 5 повторам, в скобках — 95% бутстреп-интервал по звёздам (1000 повторов, перцентили 2,5 и 97,5). Время — медиана из 5 обучений одной модели на всей обучающей части в один поток ({cpu['processor']}, логических ядер {cpu['logical_cores']}).\n\n")
    rows = []
    for f in FAMILIES:
        s = summary[f]
        rows.append({'Алгоритм': NAMES[f], 'ROC-AUC': ec.fmt_ci(s['roc_auc'], ci_of(draws, f, 'roc_auc')),
                     f"AP (доля класса 1 = {ec.fmt_num(pos_share, 3)})": ec.fmt_ci(s['ap'], ci_of(draws, f, 'ap')),
                     'Risk@70%': ec.fmt_ci(s[RISK_KEY], ci_of(draws, f, RISK_KEY)), 'AURC': ec.fmt_ci(s['aurc'], ci_of(draws, f, 'aurc')),
                     'Brier': ec.fmt_ci(s['brier'], ci_of(draws, f, 'brier')), 'Время, с': ec.fmt_num(s['fit_median_s'], 3)})
    lines.append(ec.md_table(pd.DataFrame(rows)))
    lines.append('\n' + ec.heading(2, 'Расширенная таблица'))
    rows = []
    for f in FAMILIES:
        s = summary[f]
        mem = 'н/д' if s.get('memory_mb') is None else ec.fmt_num(s['memory_mb'], 1)
        rows.append({'Алгоритм': NAMES[f], 'Risk@50%': ec.fmt_ci(s['risk@0.50'], ci_of(draws, f, 'risk@0.50')), 'Risk@70%': ec.fmt_ci(s[RISK_KEY], ci_of(draws, f, RISK_KEY)),
                     'Risk@90%': ec.fmt_ci(s['risk@0.90'], ci_of(draws, f, 'risk@0.90')), 'AUGRC': ec.fmt_ci(s['augrc'], ci_of(draws, f, 'augrc')),
                     'Испытаний': s['n_trials'] if s['n_trials'] else '0 (без подбора)', 'СО ROC-AUC по фолдам': ec.fmt_num(s['fold_auc_std'], 3),
                     'ROC-AUC фолдов, мин–макс': f"{ec.fmt_num(s['fold_auc_min'], 3)}–{ec.fmt_num(s['fold_auc_max'], 3)}",
                     'Обучение, с (мин–макс)': f"{ec.fmt_num(s['fit_median_s'], 3)} ({ec.fmt_num(s['fit_min_s'], 3)}–{ec.fmt_num(s['fit_max_s'], 3)})",
                     'Предсказание, с': ec.fmt_num(s['predict_median_s'], 4), 'Память, МБ': mem})
    lines.append(ec.md_table(pd.DataFrame(rows)))
    ens = summary['xgb_ens']
    lines.append(f"\nСО ROC-AUC по фолдам — стандартное отклонение ROC-AUC по 5 внешним фолдам внутри повтора, усреднённое по 5 повторам. Охват C: доля принятых сигналов с наибольшей уверенностью max(p, 1−p), класс по порогу 0,5, при равной уверенности порядок по signal_id; число принятых — ceil(C·n). AURC — среднее риска по всем охватам k/n, AUGRC — среднее доли ошибок среди всех сигналов (охват × риск). Память: psutil не установлен, поэтому «н/д».\n")
    lines.append(f"\nАнсамбль XGBoost (текущая система): гиперпараметры из 3_demo_pipeline.py без подбора — fit_bootstrap_ensemble (n_estimators=60, max_depth=4, learning_rate=0.08, scale_pos_weight = отношение классов), {min(20, max(5, 2 * len(y)))} моделей по формуле min(20, max(5, 2·n)) из train_and_eval_pipeline, бутстреп по звёздам; предобработка get_ml_pipeline (медиана, log1p(max(x, 0)), RobustScaler). Межмодельный разброс s на внекорзинных сигналах: среднее {ec.fmt_num(ens['s_mean'], 3)}, медиана {ec.fmt_num(ens['s_median'], 3)}.\n")
    conv = {NAMES[f]: summary[f]['n_conv_warnings'] for f in FAMILIES if summary[f]['n_conv_warnings']}
    if conv:
        lines.append(f"\nПредупреждения о несходимости (ConvergenceWarning) за всю вложенную проверку: {conv}.\n")
    lines.append('\n' + ec.heading(2, 'Парето-анализ'))
    lines.append('Оси: risk@70% (меньше лучше), AURC (меньше лучше), СО ROC-AUC по фолдам (меньше лучше), время обучения (меньше лучше).\n\n')
    lines.append('Недоминируемые модели: ' + ', '.join(NAMES[f] for f in front) + '.\n\n')
    lines.append(f"Лучшая по ROC-AUC: {NAMES[best_auc]}; лучшая по AURC: {NAMES[best_aurc]}. Ниже — парный бутстреп по звёздам (одни и те же выборки звёзд для обеих моделей), разность «модель минус лучшая».\n\n")
    rows = []
    for f in FAMILIES:
        a = paired_ci(draws, f, best_auc, 'roc_auc')
        b = paired_ci(draws, f, best_aurc, 'aurc')
        rows.append({'Алгоритм': NAMES[f], 'ΔROC-AUC к лучшей': ec.fmt_ci(summary[f]['roc_auc'] - summary[best_auc]['roc_auc'], a), 'ΔAURC к лучшей': ec.fmt_ci(summary[f]['aurc'] - summary[best_aurc]['aurc'], b),
                     'Неотличима по обоим': 'да' if f in indist else 'нет', 'Недоминируема': 'да' if f in front else 'нет'})
    lines.append(ec.md_table(pd.DataFrame(rows)))
    lines.append('\n' + ec.heading(2, 'Гиперпараметры, подобранные на всей обучающей части'))
    lines.append('Тот же подбор (12 испытаний, StratifiedGroupKFold(3) по звёздам, seed 42) на всей обучающей части; эти значения используются для замеров времени, проверки на валидационной части и оценки CANDIDATE.\n\n')
    for f in TUNED:
        lines.append(f"- {NAMES[f]}: `{json.dumps(final_params[f], ensure_ascii=False)}`\n")
    return ''.join(lines)


def decision_md(summary, draws, front, leader, tied, chosen, oof_sha, n_signals, n_stars):
    lines = [ec.heading(1, 'E2. Решение о выборе семейства модели')]
    lines.append('Правило объявлено в коде e2_compare.py (DECISION_RULE, decide) до просмотра валидационной части; файл сформирован скриптом автоматически. Валидационная, калибровочная и тестовая части при выборе не открывались.\n\n')
    lines.append(ec.heading(2, 'Правило'))
    for i, r in enumerate(DECISION_RULE, 1):
        lines.append(f"{i}. {r}\n")
    lines.append('\n' + ec.heading(2, 'Данные'))
    lines.append(f"Внекорзинные предсказания вложенной проверки на обучающей части: {n_signals} сигналов, {n_stars} звёзд, 5 повторов × StratifiedGroupKFold(5). sha256 файла e2_oof_predictions.csv: `{oof_sha}`.\n\n")
    rows = []
    for f in FAMILIES:
        s = summary[f]
        lo, hi = paired_ci(draws, f, leader, RISK_KEY)
        rows.append({'Алгоритм': NAMES[f], 'Risk@70%': ec.fmt_num(s[RISK_KEY]), 'AURC': ec.fmt_num(s['aurc']), 'СО ROC-AUC по фолдам': ec.fmt_num(s['fold_auc_std']), 'Время обучения, с': ec.fmt_num(s['fit_median_s']),
                     'Испытаний подбора': s['n_trials'], 'Недоминируема': 'да' if f in front else 'нет',
                     'ΔRisk@70% к лидеру (95% ДИ)': '—' if f == leader else ec.fmt_ci(s[RISK_KEY] - summary[leader][RISK_KEY], (lo, hi))})
    lines.append(ec.md_table(pd.DataFrame(rows)))
    lines.append('\n' + ec.heading(2, 'Результат'))
    lines.append('Недоминируемые: ' + ', '.join(NAMES[f] for f in front) + '.\n\n')
    lines.append(f"Лидер по risk@70%: {NAMES[leader]}.\n\n")
    lines.append('Неотличимы от лидера по парному интервалу risk@70% (среди недоминируемых, включая лидера): ' + ', '.join(NAMES[f] for f in tied) + '.\n\n')
    lines.append(f"**Выбрано: {NAMES[chosen]}.**\n\n")
    if chosen == leader and len(tied) == 1:
        why = 'лидер по risk@70% отличим от остальных недоминируемых моделей, поэтому выбирается он.'
    elif chosen == leader:
        why = 'среди неотличимых моделей лидер оказался и самым простым по объявленному порядку.'
    else:
        why = f"лидер ({NAMES[leader]}) статистически неотличим по risk@70% от других недоминируемых моделей, поэтому по правилу выбирается самая простая из них."
    lines.append('Обоснование: ' + why + '\n\n')
    lines.append('Решение после коммита не меняется; результаты валидационной части (п. 3) и последующих шагов только описываются.\n')
    return ''.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description='E2: comparison of six model families on the training part')
    ap.add_argument('--run-dir', default=str(ec.RUN_DIR))
    ap.add_argument('--out-dir', default=str(ec.OUT_DIR))
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    warnings.filterwarnings('ignore')
    out = Path(args.out_dir)
    df, cols = load_labeled_part(out, args.run_dir, ec.PART_TRAIN)
    X = df[cols].values.astype(float)
    y = df['y'].values.astype(int)
    groups = df['target_id'].astype(str).values
    fp = dataset_fingerprint(df, cols)
    print(f"train part: {len(df)} signals, {len(set(groups))} stars, {len(cols)} features, fingerprint {fp}", flush=True)
    results = run_outer(X, y, groups, out / '_e2_cache', fp, args.workers, FAMILIES)
    p, s_ens, fold_of = collect_oof(results, len(y))
    for f in FAMILIES:
        for s in REPEAT_SEEDS:
            if np.isnan(p[f][s]).any():
                raise SystemExit(f"missing OOF predictions for {f} seed {s}")
    final_params = {}
    final_trials = {}
    for f in TUNED:
        best, trials, _ = tune(f, X, y, groups, ec.SEED)
        final_params[f] = {k: py(v) for k, v in best.items()}
        final_trials[f] = trials
        print('final tuning', f, final_params[f], flush=True)
    final_params['xgb_ens'] = {}
    ec.write_json(out / 'e2_best_params.json', {'params': final_params, 'trials': final_trials, 'fingerprint': fp, 'features': cols})
    timing = {f: measure_time(f, final_params[f], X, y, groups) for f in FAMILIES}
    print('timing done', flush=True)
    oof_rows = []
    for f in FAMILIES:
        for i in range(len(y)):
            row = {'signal_id': df['signal_id'].iat[i], 'target_id': groups[i], 'y': int(y[i]), 'model': f}
            for s in REPEAT_SEEDS:
                row[f"p_seed{s}"] = p[f][s][i]
                row[f"fold_seed{s}"] = int(fold_of[s][i])
            row['p_avg'] = float(np.mean([p[f][s][i] for s in REPEAT_SEEDS]))
            row['s_avg'] = float(np.mean([s_ens[s][i] for s in REPEAT_SEEDS])) if f == 'xgb_ens' else np.nan
            oof_rows.append(row)
    oof_path = out / 'e2_oof_predictions.csv'
    pd.DataFrame(oof_rows).to_csv(oof_path, index=False, encoding='utf-8', lineterminator='\n', float_format='%.10g')
    oof_sha = ec.sha256_file(oof_path)
    preds = {f: [p[f][s] for s in REPEAT_SEEDS] for f in FAMILIES}
    draws = ec.bootstrap_metrics(y, preds, groups, n_boot=args.n_boot, seed=ec.SEED)
    print('bootstrap done', flush=True)
    summary, long_df = summarize(df, p, s_ens, fold_of, timing, draws, results)
    long_df.to_csv(out / 'e2_results.csv', index=False, encoding='utf-8', lineterminator='\n', float_format='%.10g')
    front = pareto_front(summary)
    best_auc = max(FAMILIES, key=lambda f: (summary[f]['roc_auc'], -SIMPLICITY[f]))
    best_aurc = min(FAMILIES, key=lambda f: (summary[f]['aurc'], SIMPLICITY[f]))
    indist = []
    for f in FAMILIES:
        a = paired_ci(draws, f, best_auc, 'roc_auc')
        b = paired_ci(draws, f, best_aurc, 'aurc')
        if a[0] <= 0 <= a[1] and b[0] <= 0 <= b[1]:
            indist.append(f)
    leader, tied, chosen = decide(summary, draws, front)
    cpu = cpu_info()
    ec.write_text(out / 'table2.md', table2_md(df, summary, draws, front, best_auc, best_aurc, indist, cpu, final_params))
    risk_coverage_figure(y, {f: np.mean([p[f][s] for s in REPEAT_SEEDS], axis=0) for f in FAMILIES}, out / 'e2_risk_coverage.png')
    ec.write_text(out / 'E2_decision.md', decision_md(summary, draws, front, leader, tied, chosen, oof_sha, len(y), len(set(groups))))
    ec.write_json(out / 'e2_summary.json', {'summary': summary, 'pareto_front': front, 'best_auc': best_auc, 'best_aurc': best_aurc, 'indistinguishable_from_best': indist,
                                            'leader': leader, 'tied_with_leader': tied, 'chosen': chosen, 'oof_sha256': oof_sha, 'cpu': cpu, 'packages': ec.package_versions(), 'fingerprint': fp})
    print('chosen', chosen, 'leader', leader, 'front', front, flush=True)


if __name__ == '__main__':
    main()
