import argparse
import importlib.util
import inspect
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
STAR = '5780885'
DISTINCT_TOL = 0.01
TOP_N = 5


class Tee:
    def __init__(self, path):
        self.fh = open(path, 'w', encoding='utf-8')

    def __call__(self, *parts):
        line = ' '.join(str(p) for p in parts)
        print(line, flush=True)
        self.fh.write(line + '\n')
        self.fh.flush()

    def close(self):
        self.fh.close()


def fmt(x, nd=6):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return 'nan'
    return f'{x:.{nd}g}'


def distinct_top(periods, power, n=TOP_N, tol=DISTINCT_TOL):
    p = np.asarray(periods, dtype=float)
    w = np.asarray(power, dtype=float)
    ok = np.where(np.isfinite(w) & np.isfinite(p) & (p > 0))[0]
    order = ok[np.argsort(-w[ok], kind='stable')]
    chosen = []
    for i in order:
        if all(abs(p[i] / p[j] - 1.0) > tol for j in chosen):
            chosen.append(int(i))
            if len(chosen) >= n:
                break
    return chosen


def git_rev(code_dir):
    try:
        r = subprocess.run(['git', '-C', str(code_dir), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def load_module(code_dir):
    sys.path.insert(0, str(code_dir))
    spec = importlib.util.spec_from_file_location('onyx_extractor_diag', str(code_dir / '1_data_extractor.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['onyx_extractor_diag'] = mod
    spec.loader.exec_module(mod)
    return mod


def install(mod, log):
    st = {'iter': 0, 'spec': None, 'ev': {}, 'closed': set(), 'outcomes': []}
    orig = {n: getattr(mod, n, None) for n in ('bls_spectrum', 'refine_peak', 'resolve_alias_period', 'is_alias_duplicate', 'count_observed_transits', '_power_near', '_box_depth', '_class_depths_ok', 'raw_sde', 'detrended_sde', 'peak_index')}
    c = lambda name, default=None: getattr(mod, name, default)
    orig_tw = getattr(mod, 'transit_windows', None)
    orig_clip = getattr(mod, 'clip_outliers', None)

    if orig_clip is not None:
        def clip_outliers(lc, *a, **k):
            out = orig_clip(lc, *a, **k)
            tb = np.asarray(lc.time.value, dtype=float)
            fb = np.asarray(lc.flux.value, dtype=float)
            keep = np.isin(tb, np.asarray(out.time.value, dtype=float))
            st['removed_t'] = tb[~keep]
            st['removed_f'] = fb[~keep]
            med = np.nanmedian(fb)
            log(f"clip_outliers: было {len(tb)} точек, осталось {int(keep.sum())}, удалено {int((~keep).sum())} (ниже медианы {int(np.sum(fb[~keep] < med))}, выше {int(np.sum(fb[~keep] > med))}); медиана потока {fmt(med)}, медиана удалённых {fmt(np.nanmedian(fb[~keep]) if (~keep).any() else np.nan)}")
            return out
        mod.clip_outliers = clip_outliers

    def finalize():
        it = st['iter']
        if it == 0 or it in st['closed']:
            return
        st['closed'].add(it)
        ev = st['ev']
        if ev.get('nan_spec'):
            out = 'спектр без конечных значений, поиск прерван'
        elif ev.get('gate_fail'):
            out = 'отброшен по ' + ev['gate_fail'] + ', поиск прерван (break)'
        elif 'resolve' in ev:
            r = ev['resolve']
            fac = r.get('factor')
            head = f"множитель алиаса {fmt(fac)}, период после resolve {fmt(r.get('period'))}"
            if ev.get('dup') is True:
                out = head + '; дубликат алиаса (is_alias_duplicate=True), транзиты замаскированы, итерация пропущена'
            elif 'ntr' in ev:
                need = c('MIN_TRANSITS_OBSERVED')
                if ev['ntr'] < need:
                    out = head + f"; отброшен по числу транзитов ({ev['ntr']} < {need}), транзиты замаскированы"
                else:
                    out = head + f"; принят (наблюдаемых транзитов {ev['ntr']})"
            elif ev.get('dup') is False:
                out = head + '; не дубликат; принят или отброшен по числу транзитов (в этой версии не наблюдается)'
            else:
                out = head
        else:
            out = 'resolve_alias_period не вызван (прерывание по SDE/SNR внутри кода или resolve_aliases выключен)'
        st['outcomes'].append((it, out, ev.get('final_period')))
        log(f'[iter {it}] ИСХОД: {out}')

    def top_block(periods, power, extra=None):
        idxs = distinct_top(periods, power)
        log(f'  пять самых высоких разных пиков сырой мощности (разнос > {DISTINCT_TOL * 100:.0f}% по периоду):')
        for k, i in enumerate(idxs, 1):
            if extra is not None:
                log(f"    {k}. P={fmt(periods[i])} d  power={fmt(power[i])}  dur={fmt(extra['duration'][i])} d  depth={fmt(extra['depth'][i] * 1e6)} ppm  snr={fmt(extra['depth_snr'][i])}")
            else:
                log(f'    {k}. P={fmt(periods[i])} d  power={fmt(power[i])}  (duration/depth/snr: нет в этой версии)')
        return idxs

    if orig['bls_spectrum'] is not None:
        def bls_spectrum(*a, **k):
            finalize()
            st['iter'] += 1
            st['ev'] = {}
            t = np.asarray(a[0]) if a else np.asarray(k.get('t'))
            log('')
            log(f"===== iter {st['iter']}: bls_spectrum, точек в кривой {len(t)}, t=[{fmt(np.nanmin(t))}, {fmt(np.nanmax(t))}] =====")
            spec = orig['bls_spectrum'](*a, **k)
            st['spec'] = spec
            log(f"  частот в спектре: {len(spec['period'])}, период {fmt(np.nanmin(spec['period']))}..{fmt(np.nanmax(spec['period']))}")
            idxs = top_block(spec['period'], spec['power'], spec)
            pi = orig['peak_index'](spec['power']) if orig['peak_index'] else None
            if pi is None:
                st['ev']['nan_spec'] = True
            else:
                log(f"  выбираемый кодом пик (peak_index): idx={pi} P={fmt(spec['period'][pi])} power={fmt(spec['power'][pi])}")
            return spec
        mod.bls_spectrum = bls_spectrum

    if orig['refine_peak'] is not None:
        def refine_peak(spec, idx, baseline_d):
            res = orig['refine_peak'](spec, idx, baseline_d)
            coarse_p = float(spec['period'][idx])
            raw = orig['raw_sde'](spec['power'], idx) if orig['raw_sde'] else np.nan
            det = orig['detrended_sde'](spec['power'], idx) if orig['detrended_sde'] else np.nan
            snr = float(res['depth_snr'])
            log(f"  выбранный пик: P до refine={fmt(coarse_p)} -> после refine={fmt(res['period'])} (refined={res.get('refined')}), power до={fmt(spec['power'][idx])} после={fmt(res['power'])}")
            log(f"    SDE сырой={fmt(raw)}  SDE после вычитания медианы={fmt(det)}  SNR={fmt(snr)}  dur={fmt(res['duration'])} d  depth={fmt(res['depth'] * 1e6)} ppm  epoch={fmt(res['epoch'])}")
            min_sde, min_snr = c('MIN_SDE'), c('MIN_SNR')
            why = []
            if min_sde is not None and det < min_sde:
                why.append(f'SDE {fmt(det)} < MIN_SDE {min_sde}')
            if min_snr is not None and snr < min_snr:
                why.append(f'SNR {fmt(snr)} < MIN_SNR {min_snr}')
            log(f'    порог: MIN_SDE={min_sde} MIN_SNR={min_snr} -> ' + ('; '.join(why) if why else 'проходит'))
            if why:
                st['ev']['gate_fail'] = ' и '.join(('SDE' if w.startswith('SDE') else 'SNR') for w in why) + ' (' + '; '.join(why) + ')'
            st['ev']['refined'] = res
            return res
        mod.refine_peak = refine_peak

    def resolve_alias_period(time_arr, flux_arr, err_arr, period, epoch, duration, periods, powers):
        if orig['bls_spectrum'] is None:
            finalize()
            st['iter'] += 1
            st['ev'] = {}
            p_arr = np.asarray(getattr(periods, 'value', periods), dtype=float).ravel()
            pw_arr = np.asarray(getattr(powers, 'value', powers), dtype=float).ravel()
            log('')
            log(f"===== iter {st['iter']}: (версия без bls_spectrum; данные из аргументов resolve_alias_period) =====")
            top_block(p_arr, pw_arr)
        tol = c('ALIAS_TOL', 0.01)
        fr = c('ALIAS_POWER_FRACTION')
        dshort, dlong = c('ALIAS_DEPTH_SHORT'), c('ALIAS_DEPTH_LONG')
        p_arr = None if periods is None else np.asarray(getattr(periods, 'value', periods), dtype=float).ravel()
        pw_arr = None if powers is None else np.asarray(getattr(powers, 'value', powers), dtype=float).ravel()
        log(f'  resolve_alias_period: вход P={fmt(period)} epoch={fmt(epoch)} dur={fmt(duration)}')
        pn = orig['_power_near']
        pow_c = pn(p_arr, pw_arr, period, tol) if pn else np.nan
        log(f'    ALIAS_TOL={tol} ALIAS_POWER_FRACTION={fr} ALIAS_DEPTH_SHORT={dshort} ALIAS_DEPTH_LONG={dlong}')
        log(f'    мощность около P={fmt(period)}: {fmt(pow_c)}')
        for name, ratio in (('P/3', 1 / 3), ('P/2', 0.5), ('2P', 2.0)):
            pv = pn(p_arr, pw_arr, period * ratio, tol) if pn else np.nan
            rel = pv / pow_c if np.isfinite(pv) and np.isfinite(pow_c) and pow_c != 0 else np.nan
            verdict = 'нет данных' if not np.isfinite(rel) else ('>= порога' if rel >= fr else '< порога')
            log(f'    мощность около {name}={fmt(period * ratio)}: {fmt(pv)}  отношение к P={fmt(rel)}  (порог {fr}) {verdict}')
        trace = []
        saved = {}

        def mk_pn(f):
            def w(periods_, powers_, p, tol_):
                r = f(periods_, powers_, p, tol_)
                trace.append(('pow', float(p), r))
                return r
            return w

        def mk_bd(f):
            def w(t, fl, wt, per, ep, dur):
                r = f(t, fl, wt, per, ep, dur)
                trace.append(('depth', float(per), float(ep), float(r[0])))
                return r
            return w

        def mk_cd(f):
            def w(t, fl, wt, in_m, period_c, ep, k, d_c):
                r = f(t, fl, wt, in_m, period_c, ep, k, d_c)
                trace.append(('class', int(k), bool(r)))
                return r
            return w

        for name, mk in (('_power_near', mk_pn), ('_box_depth', mk_bd), ('_class_depths_ok', mk_cd)):
            if orig[name] is not None:
                saved[name] = orig[name]
                setattr(mod, name, mk(orig[name]))
        try:
            res = orig['resolve_alias_period'](time_arr, flux_arr, err_arr, period, epoch, duration, periods, powers)
        finally:
            for name, f in saved.items():
                setattr(mod, name, f)
        first_depth = [x for x in trace if x[0] == 'depth' and abs(x[1] - period) < 1e-9 * max(1.0, period)]
        d_c = first_depth[0][3] if first_depth else next((x[3] for x in trace if x[0] == 'depth'), np.nan)
        log(f'    глубина при P (d_c, box): {fmt(d_c)}')
        reasons = []
        if not trace:
            reasons.append(('все', 'd_c не конечна или <= 0, проверки множителей не выполнялись'))
        pows = [x for x in trace if x[0] == 'pow']
        for label, ratio, thr in (('P/3', 1 / 3, dshort), ('P/2', 0.5, dshort), ('2P', 2.0, dlong)):
            p_try = period * ratio
            calls_pow = [x for x in pows if abs(x[1] / p_try - 1.0) < 1e-9]
            calls_d = [x for x in trace if x[0] == 'depth' and abs(x[1] / p_try - 1.0) < 1e-9]
            calls_c = [x for x in trace if x[0] == 'class' and x[1] == (3 if label == 'P/3' else 2 if label == 'P/2' else -1)]
            if not calls_pow:
                reasons.append((label, f'пропущен: {label}={fmt(p_try)} <= длительности {fmt(duration)}' if ratio < 1 else 'не вычислялся'))
                continue
            pv = calls_pow[0][2]
            if not (np.isfinite(pv) and np.isfinite(pow_c) and pv >= fr * pow_c):
                reasons.append((label, f'мощность: {fmt(pv)} < {fr} * {fmt(pow_c)} = {fmt(fr * pow_c)} (отношение {fmt(pv / pow_c if np.isfinite(pv) and pow_c else np.nan)})'))
                continue
            if not calls_d:
                reasons.append((label, 'мощность прошла, но глубины не вычислялись'))
                continue
            for x in calls_d:
                off = (x[2] - epoch) / period
                tag = 'epoch' if abs(off) < 1e-9 else ('epoch+P' if abs(off - 1) < 1e-9 else f'epoch+{fmt(off)}P')
                log(f'    глубина при {label} ({tag}): {fmt(x[3])}  отношение к d_c={fmt(x[3] / d_c if d_c else np.nan)}  (порог {thr})')
            best = max((x[3] for x in calls_d if np.isfinite(x[3])), default=np.nan)
            if not (np.isfinite(best) and d_c and best >= thr * d_c):
                reasons.append((label, f'глубина: лучшая {fmt(best)} < {thr} * d_c = {fmt(thr * d_c if d_c else np.nan)} (отношение {fmt(best / d_c if d_c else np.nan)})'))
                continue
            if label != '2P':
                cok = [x[2] for x in calls_c]
                if cok and not all(cok):
                    reasons.append((label, 'поклассовые глубины (_class_depths_ok) не прошли'))
                    continue
            reasons.append((label, 'все условия выполнены -> множитель применён'))
        for n_, (label, why) in enumerate(reasons, 1):
            log(f"    проверка {label}: {why}" + ('   <- первое отклонение' if n_ == 1 and 'выполнены' not in why else ''))
        log(f"    итог resolve: factor={fmt(res['factor'])} period={fmt(res['period'])} epoch={fmt(res['epoch'])} depth={fmt(res['depth'])}")
        st['ev']['resolve'] = res
        st['ev']['final_period'] = res['period']
        return res
    mod.resolve_alias_period = resolve_alias_period

    if orig['is_alias_duplicate'] is not None:
        def is_alias_duplicate(period, found_periods, *a, **k):
            r = orig['is_alias_duplicate'](period, found_periods, *a, **k)
            log(f"  is_alias_duplicate(P={fmt(period)}, найденные={[fmt(x) for x in found_periods]}) -> {r}")
            st['ev']['dup'] = bool(r)
            return r
        mod.is_alias_duplicate = is_alias_duplicate

    if orig['count_observed_transits'] is not None:
        def count_observed_transits(*a, **k):
            r = orig['count_observed_transits'](*a, **k)
            log(f"  count_observed_transits(P={fmt(a[1])}) -> {r} (MIN_TRANSITS_OBSERVED={c('MIN_TRANSITS_OBSERVED')})")
            st['ev']['ntr'] = int(r)
            tw = orig_tw
            if tw is not None:
                t_, per_, ep_, dur_, cad_ = np.asarray(a[0], dtype=float), a[1], a[2], a[3], a[4]
                wins = tw(t_, per_, ep_, dur_)
                counts = [int(np.sum(m)) for _, m in wins]
                cov = c('TRANSIT_COVERAGE_MIN')
                exp_ = dur_ / cad_ if cad_ > 0 else np.nan
                log(f'    зонд: cadence_d={fmt(cad_)} dur={fmt(dur_)} epoch={fmt(ep_)} ожидаемо точек в транзите={fmt(exp_)} нужно (coverage {cov})={fmt(cov * exp_ if cov else np.nan)}')
                log(f'    зонд: окон transit_windows={len(wins)}, окон с точками={sum(1 for x in counts if x > 0)}, окон с покрытием>=нужного={sum(1 for x in counts if cov and x >= cov * exp_ and x > 0)}, максимум точек в окне={max(counts) if counts else 0}')
                log(f'    зонд: первые окна (k, середина, точек): ' + '; '.join(f'({k}, {fmt(ep_ + k * per_)}, {n})' for (k, _), n in list(zip(wins, counts))[:8]))
                rt = st.get('removed_t')
                if rt is not None and len(rt) > 0:
                    rf = st['removed_f']
                    inw = np.zeros(len(rt), dtype=bool)
                    mids = [ep_ + k_ * per_ for k_, _ in wins]
                    for mid_ in mids:
                        inw |= np.abs(rt - mid_) < dur_ / 2.0
                    log(f'    зонд: удалено clip_outliers всего {len(rt)}, из них внутри окон транзитов этого (P, epoch): {int(inw.sum())} ({fmt(100.0 * inw.sum() / len(rt))}%), доля времени в окнах {fmt(100.0 * dur_ / per_)}%')
                    if inw.any():
                        log(f'    зонд: медиана потока удалённых внутри окон={fmt(np.nanmedian(rf[inw]))}, вне окон={fmt(np.nanmedian(rf[~inw]) if (~inw).any() else np.nan)}')
                log(f'    зонд: медиана dt по t={fmt(np.nanmedian(np.diff(t_)))}, точек={len(t_)}')
            return r
        mod.count_observed_transits = count_observed_transits

    return st, finalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code-dir', default=str(REPO_ROOT))
    ap.add_argument('--out-dir', default=str(REPO_ROOT / 'outputs' / '_diag_kepler7'))
    ap.add_argument('--label', default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    code_dir = Path(args.code_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    cache_dir = out_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    rev = git_rev(code_dir)
    label = args.label or rev
    log = Tee(out_dir / f'run_{label}.txt')
    log(f'ДИАГНОСТИКА Kepler-7 (KIC {STAR}), код: {code_dir}, коммит {rev}')
    mod = load_module(code_dir)
    log(f"PIPELINE_VERSION (build_pipeline_version): {getattr(mod, 'build_pipeline_version', lambda *a: 'n/a')(False, 3.0, 4)}")
    st, finalize = install(mod, log)
    fn = mod.process_single_star
    params = inspect.signature(fn).parameters
    kwargs = dict(quick_mode=False, cache_dir=str(cache_dir), star_timeout=3600.0, max_sectors=0)
    if 'resolve_aliases' in params:
        kwargs['resolve_aliases'] = True
    if 'inverted_control' in params:
        kwargs['inverted_control'] = False
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    t0 = time.time()
    res = fn(STAR, 'Kepler', [], False, 1.0, 3.0, 4, **kwargs)
    finalize()
    feats, curves, tid, err, dl_size, timings = res
    log('')
    log(f'process_single_star завершён за {time.time() - t0:.0f} с, ошибка: {err}, скачано байт: {dl_size}')
    log('')
    log('ТАБЛИЦА ИТЕРАЦИЙ:')
    for it, out, fp in st['outcomes']:
        log(f'  iter {it}: {out}')
    log('')
    log('ПРИНЯТЫЕ СИГНАЛЫ (из возвращённых признаков):')
    rows = feats if isinstance(feats, list) else ([] if feats is None else [feats])
    for r in rows:
        if hasattr(r, 'items'):
            sel = {k: v for k, v in r.items() if any(s in k.lower() for s in ('period', 'sde', 'snr', 'alias', 'signal_id', 'sig_id', 'transit_count', 'n_transit'))}
            log('  ' + ', '.join(f'{k}={fmt(v) if isinstance(v, (int, float, np.floating)) else v}' for k, v in sel.items()))
    if not rows:
        log('  (нет)')
    log.close()


if __name__ == '__main__':
    main()
