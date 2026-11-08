import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    'ru': {
        'raw_tall': "Kepler-80 (KIC {sid}): Исходная фотометрия ({n_files} файлов, {n_quarters} уник. кварталов, базлайн {baseline:.1f} д)",
        'flat_tall': "Сглаженная и выровненная кривая блеска (Flattened, spline window=3.0d)",
        'bls_tall': "Периодограмма BLS (нормализованная мощность SDE = (P - μ)/σ)\nДиапазон поиска: 0.5–100.0 д | Отображаемый фрагмент: 0.5–12.0 д",
        'raw_wide': "Исходная фотометрия",
        'flat_wide': "Сглаженная кривая",
        'bls_wide': "Периодограмма BLS (SDE)",
        'time_axis': "Время (BJD - 2454833)",
        'bjd_axis': "BJD - 2454833",
        'flux_norm': "Нормированный поток",
        'flux_rel': "Относительный поток",
        'period_axis': "Период (дни)",
        'sde_axis': "Нормализованная мощность SDE",
        'sde_short': "SDE",
        'peak': "Пик: {p:.4f} д (SDE={sde:.1f})",
        'peak_short': "{p:.4f} д",
        'phot_legend': "Фотометрия (чужие транзиты маскированы)",
        'phot_short': "Фотометрия",
        'binned': "Биннированный профиль (~18 мин)",
        'binned_short': "Профиль",
        'expected': "Ожидаемый транзит (глубина {d:.0f} ppm, длительность {h:.1f} ч)",
        'expected_short': "Ожидаемый транзит",
        'sig_tall': "Сигнал {i}: P = {p:.4f} д | Глубина = {d:.0f} ppm | Транзитов: ~{n} | Вердикт: {dec} (p = {prob})",
        'sig_wide': "Сигнал {i}: P = {p:.4f} д, {d:.0f} ppm\np = {prob}, {dec}",
        'transit_axis': "Время от центра транзита (часы)",
        'transit_axis_wide': "Время от центра транзита (ч)",
        'no_signals': "В текущем тестовом режиме сигналы не обнаружены",
        'no_signal': "Нет сигнала",
        'not_scored': "НЕ_ОЦЕНИВАЛСЯ",
        'na': "N/A",
    },
    'en': {
        'raw_tall': "Kepler-80 (KIC {sid}): Raw photometry ({n_files} files, {n_quarters} unique quarters, baseline {baseline:.1f} d)",
        'flat_tall': "Detrended and flattened light curve (Flattened, spline window=3.0d)",
        'bls_tall': "BLS periodogram (normalized power SDE = (P - μ)/σ)\nSearch range: 0.5–100.0 d | Displayed range: 0.5–12.0 d",
        'raw_wide': "Raw photometry",
        'flat_wide': "Detrended light curve",
        'bls_wide': "BLS periodogram (SDE)",
        'time_axis': "Time (BJD - 2454833)",
        'bjd_axis': "BJD - 2454833",
        'flux_norm': "Normalized flux",
        'flux_rel': "Relative flux",
        'period_axis': "Period (days)",
        'sde_axis': "Normalized SDE power",
        'sde_short': "SDE",
        'peak': "Peak: {p:.4f} d (SDE={sde:.1f})",
        'peak_short': "{p:.4f} d",
        'phot_legend': "Photometry (other transits masked)",
        'phot_short': "Photometry",
        'binned': "Binned profile (~18 min)",
        'binned_short': "Binned",
        'expected': "Expected transit (depth {d:.0f} ppm, duration {h:.1f} h)",
        'expected_short': "Expected transit",
        'sig_tall': "Signal {i}: P = {p:.4f} d | Depth = {d:.0f} ppm | Transits: ~{n} | Verdict: {dec} (p = {prob})",
        'sig_wide': "Signal {i}: P = {p:.4f} d, {d:.0f} ppm\np = {prob}, {dec}",
        'transit_axis': "Time from transit centre (hours)",
        'transit_axis_wide': "Time from transit centre (h)",
        'no_signals': "No signals were detected in the current test mode",
        'no_signal': "No signal",
        'not_scored': "NOT_SCORED",
        'na': "N/A",
    },
}

RU_NOT_SCORED = 'НЕ_ОЦЕНИВАЛСЯ'


def match_verdicts(signals_df, scored_df, tol=0.01):
    p_means = []
    verdicts = []
    for _, row in signals_df.iterrows():
        p = float(row['bls_period_d'])
        p_mean = None
        verdict = None
        if scored_df is not None and len(scored_df) > 0:
            for _, cand in scored_df.iterrows():
                cp = float(cand['bls_period_d'])
                if cp > 0 and abs(p / cp - 1.0) < tol:
                    pm = cand.get('p_mean')
                    p_mean = float(pm) if pd.notnull(pm) else None
                    dv = cand.get('decision')
                    verdict = str(dv) if pd.notnull(dv) else None
                    break
        p_means.append(p_mean)
        verdicts.append(verdict)
    out = signals_df.copy()
    out['p_mean'] = p_means
    out['decision'] = verdicts
    return out


def _prob_text(p_mean, fmt, na):
    if p_mean is not None and pd.notnull(p_mean) and not np.isnan(float(p_mean)):
        return format(float(p_mean), fmt)
    return na


def _fold_signal(data, signals, idx, row):
    times_flat = data['times_flat']
    flux_flat = data['flux_flat']
    p_val = float(row.get('bls_period_d', 1.0))
    t0_val = float(row.get('bls_epoch', 0.0))
    dur_val = float(row.get('bls_duration_d', 0.1))
    dur_hours = dur_val * 24.0

    mask_other = np.zeros(len(times_flat), dtype=bool)
    for other_idx, (_, other_row) in enumerate(signals.iterrows()):
        if other_idx == idx:
            continue
        op = float(other_row.get('bls_period_d', 0))
        ot0 = float(other_row.get('bls_epoch', 0))
        odur = float(other_row.get('bls_duration_d', 0))
        if op > 0:
            oph = ((times_flat - ot0 + 0.5 * op) % op) - 0.5 * op
            mask_other |= (np.abs(oph) < 0.6 * odur)

    t_clean = times_flat[~mask_other]
    f_clean = flux_flat[~mask_other]

    phase_d = ((t_clean - t0_val + 0.5 * p_val) % p_val) - 0.5 * p_val
    hours_from_mid = phase_d * 24.0
    win_hours = max(4.0 * dur_hours, 6.0)
    m_win = np.abs(hours_from_mid) <= win_hours
    x_win = hours_from_mid[m_win]
    y_win = f_clean[m_win]

    if len(y_win) > 10:
        med_y = float(np.median(y_win))
        mad_y = float(np.median(np.abs(y_win - med_y)))
        sig_y = 1.4826 * mad_y if mad_y > 0 else float(np.std(y_win))
        if sig_y > 0:
            in_5sig = np.abs(y_win - med_y) <= 5.0 * sig_y
            x_win = x_win[in_5sig]
            y_win = y_win[in_5sig]

    bin_width = 0.3
    n_bins = int(np.ceil((2.0 * win_hours) / bin_width))
    bin_edges = np.linspace(-win_hours, win_hours, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_medians = []
    valid_centers = []
    for bi in range(len(bin_centers)):
        bm = (x_win >= bin_edges[bi]) & (x_win < bin_edges[bi + 1])
        if np.any(bm):
            bin_medians.append(float(np.nanmedian(y_win[bm])))
            valid_centers.append(bin_centers[bi])
    return {
        'x': x_win,
        'y': y_win,
        'centers': np.array(valid_centers),
        'medians': np.array(bin_medians),
        'win_hours': win_hours,
        'dur_hours': dur_hours,
    }


def _plot_fold(ax_f, fold, depth_ppm, labels, wide):
    depth_frac = depth_ppm * 1e-6
    dur_hours = fold['dur_hours']
    win_hours = fold['win_hours']
    if wide:
        ax_f.plot(fold['x'], fold['y'], 'k.', markersize=1.5, alpha=0.25, label=labels['phot_short'])
    else:
        ax_f.plot(fold['x'], fold['y'], 'k.', markersize=2, alpha=0.25, label=labels['phot_legend'])
    if len(fold['medians']) > 0:
        ax_f.plot(fold['centers'], fold['medians'], 'r-', linewidth=1.4 if wide else 1.8,
                  label=labels['binned_short'] if wide else labels['binned'])
        y_p1 = float(np.percentile(fold['medians'], 1))
        y_p99 = float(np.percentile(fold['medians'], 99))
        margin = max(0.0003, (y_p99 - y_p1) * 0.4)
        ax_f.set_ylim(y_p1 - margin, y_p99 + margin)
    t_half_dur = dur_hours / 2.0
    ax_f.plot([-t_half_dur, -t_half_dur, t_half_dur, t_half_dur],
              [1.0, 1.0 - depth_frac, 1.0 - depth_frac, 1.0],
              color='tab:cyan', linestyle='--', linewidth=1.5,
              label=labels['expected_short'] if wide else labels['expected'].format(d=depth_ppm, h=dur_hours))
    ax_f.set_xlim(-win_hours, win_hours)


def _verdict_text(value, labels, missing):
    if value is None or (not isinstance(value, str) and pd.isnull(value)):
        return missing
    text = str(value)
    if text == RU_NOT_SCORED:
        return labels['not_scored']
    return text


def _render_tall(data, out_path, labels):
    signals = data['signals']
    times_raw, flux_raw = data['times_raw'], data['flux_raw']
    times_flat, flux_flat = data['times_flat'], data['flux_flat']
    n_sigs = min(4, len(signals))
    total_panels = 3 + max(1, n_sigs)
    fig = plt.figure(figsize=(15, 3.8 * total_panels))
    gs = fig.add_gridspec(total_panels, 1)
    baseline_days = data['baseline_days']

    ax1 = fig.add_subplot(gs[0, 0])
    sub_step = max(1, len(times_raw) // 15000)
    ax1.plot(times_raw[::sub_step], flux_raw[::sub_step], 'k.', markersize=1, alpha=0.5)
    ax1.set_title(labels['raw_tall'].format(sid=data['showcase_id'], n_files=data['n_files'], n_quarters=data['n_quarters'], baseline=baseline_days))
    ax1.set_xlabel(labels['time_axis'])
    ax1.set_ylabel(labels['flux_norm'])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(times_flat[::sub_step], flux_flat[::sub_step], 'b.', markersize=1, alpha=0.5)
    ax2.set_title(labels['flat_tall'])
    ax2.set_xlabel(labels['time_axis'])
    ax2.set_ylabel(labels['flux_rel'])

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(data['bls_period'], data['sde_curve'], 'r-', linewidth=0.8, label='BLS SDE')
    ax3.set_xlim(0.5, 12.0)
    ax3.set_title(labels['bls_tall'])
    ax3.set_xlabel(labels['period_axis'])
    ax3.set_ylabel(labels['sde_axis'])
    if n_sigs > 0:
        for _, s_row in signals.head(n_sigs).iterrows():
            p_val = float(s_row.get('bls_period_d', 0))
            sde_sig = float(s_row.get('bls_sde', 0))
            ax3.axvline(p_val, color='g', linestyle='--', alpha=0.8, label=labels['peak'].format(p=p_val, sde=sde_sig))
    ax3.legend(loc='upper right', fontsize=8)

    if n_sigs > 0:
        for idx, (_, s_row) in enumerate(signals.head(n_sigs).iterrows()):
            ax_f = fig.add_subplot(gs[3 + idx, 0])
            p_val = float(s_row.get('bls_period_d', 1.0))
            depth_ppm = float(s_row.get('bls_depth_ppm', 0.0))
            prob_str = _prob_text(s_row.get('p_mean'), '.1%', labels['na'])
            dec_str = _verdict_text(s_row.get('decision', RU_NOT_SCORED), labels, labels['not_scored'])
            n_transits = int(round(baseline_days / p_val)) if p_val > 0 else 0
            fold = _fold_signal(data, signals, idx, s_row)
            _plot_fold(ax_f, fold, depth_ppm, labels, False)
            ax_f.set_title(labels['sig_tall'].format(i=idx + 1, p=p_val, d=depth_ppm, n=n_transits, dec=dec_str, prob=prob_str))
            ax_f.set_xlabel(labels['transit_axis'])
            ax_f.set_ylabel(labels['flux_rel'])
            ax_f.legend(loc='lower right', fontsize=8)
    else:
        ax_f = fig.add_subplot(gs[3, 0])
        ax_f.text(0.5, 0.5, labels['no_signals'], ha='center', va='center')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150)
    return fig


def _render_wide(data, out_path, labels):
    signals = data['signals']
    times_raw, flux_raw = data['times_raw'], data['flux_raw']
    times_flat, flux_flat = data['times_flat'], data['flux_flat']
    n_sigs = min(4, len(signals))
    fs_title, fs_label, fs_tick, fs_leg = 10.5, 10, 9.5, 8.5
    fig = plt.figure(figsize=(11, 6.6), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.04, wspace=0.04, hspace=0.06)
    gs = fig.add_gridspec(3, 4)
    sub_step = max(1, len(times_raw) // 15000)

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(times_raw[::sub_step], flux_raw[::sub_step], 'k.', markersize=1, alpha=0.5)
    ax1.set_title(labels['raw_wide'], fontsize=fs_title)
    ax1.set_xlabel(labels['bjd_axis'], fontsize=fs_label)
    ax1.set_ylabel(labels['flux_rel'], fontsize=fs_label)

    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.plot(times_flat[::sub_step], flux_flat[::sub_step], 'b.', markersize=1, alpha=0.5)
    ax2.set_title(labels['flat_wide'], fontsize=fs_title)
    ax2.set_xlabel(labels['bjd_axis'], fontsize=fs_label)
    ax2.set_ylabel(labels['flux_rel'], fontsize=fs_label)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(data['bls_period'], data['sde_curve'], 'r-', linewidth=0.8)
    ax3.set_xlim(0.5, 12.0)
    ax3.set_title(labels['bls_wide'], fontsize=fs_title)
    ax3.set_xlabel(labels['period_axis'], fontsize=fs_label)
    ax3.set_ylabel(labels['sde_short'], fontsize=fs_label)
    for _, s_row in signals.head(n_sigs).iterrows():
        p_val = float(s_row.get('bls_period_d', 0))
        ax3.axvline(p_val, color='g', linestyle='--', alpha=0.8, label=labels['peak_short'].format(p=p_val))
    if n_sigs > 0:
        ax3.legend(loc='upper right', fontsize=fs_leg, ncol=n_sigs)

    axes_all = [ax1, ax2, ax3]
    for idx in range(4):
        ax_f = fig.add_subplot(gs[2, idx])
        axes_all.append(ax_f)
        if idx < n_sigs:
            s_row = signals.iloc[idx]
            p_val = float(s_row.get('bls_period_d', 1.0))
            depth_ppm = float(s_row.get('bls_depth_ppm', 0.0))
            prob_str = _prob_text(s_row.get('p_mean'), '.2f', 'n/a')
            dec_str = _verdict_text(s_row.get('decision'), labels, 'n/a')
            fold = _fold_signal(data, signals, idx, s_row)
            _plot_fold(ax_f, fold, depth_ppm, labels, True)
            ax_f.set_title(labels['sig_wide'].format(i=idx + 1, p=p_val, d=depth_ppm, prob=prob_str, dec=dec_str), fontsize=fs_title)
            if idx == 0:
                ax_f.set_ylabel(labels['flux_rel'], fontsize=fs_label)
                ax_f.legend(loc='lower left', fontsize=fs_leg - 1, handlelength=1.2, borderpad=0.3, labelspacing=0.2)
        else:
            ax_f.text(0.5, 0.5, labels['no_signal'], ha='center', va='center', fontsize=fs_label)
            ax_f.set_xticks([])
            ax_f.set_yticks([])
        ax_f.set_xlabel(labels['transit_axis_wide'], fontsize=fs_label)
    for ax in axes_all:
        ax.tick_params(labelsize=fs_tick)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=300)
    return fig


def render_showcase(data, out_path, layout='tall', lang='ru', close=True):
    if layout not in ('tall', 'wide'):
        raise ValueError(f"layout must be 'tall' or 'wide', got {layout!r}")
    if lang not in LABELS:
        raise ValueError(f"lang must be 'ru' or 'en', got {lang!r}")
    labels = LABELS[lang]
    fig = _render_tall(data, out_path, labels) if layout == 'tall' else _render_wide(data, out_path, labels)
    if close:
        plt.close(fig)
    return fig
