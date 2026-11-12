import onyx_features
import onyx_report


def test_ablation_sets_add_three_diagnostic_configs_after_existing():
    feats = ['bls_depth_snr', 'star_teff', 'transit_count', 'gap_fraction', 'centroid_shift_total_sigma', 'centroid_shift_total_arcsec',
             'secondary_z', 'odd_even_z', 'secondary_depth_ppm']
    sets = onyx_report.ablation_feature_sets(feats)
    names = [n for n, _ in sets]
    assert len(sets) == 5
    assert names[0].startswith('Без звёздных признаков') and names[1] == 'Без transit_count и gap_fraction'
    assert names[2] == 'Без центроидных признаков (centroid_shift_total_sigma, centroid_shift_total_arcsec)'
    assert names[3] == 'Без вторичного затмения и чёт-нечёт (secondary_depth_ppm, secondary_z, odd_even_z)'
    assert names[4] == onyx_report.BOTH_DIAGNOSTIC_NAME
    assert sets[2][1] == ['bls_depth_snr', 'star_teff', 'transit_count', 'gap_fraction', 'secondary_z', 'odd_even_z', 'secondary_depth_ppm']
    assert sets[3][1] == ['bls_depth_snr', 'star_teff', 'transit_count', 'gap_fraction', 'centroid_shift_total_sigma', 'centroid_shift_total_arcsec']
    assert sets[4][1] == ['bls_depth_snr', 'star_teff', 'transit_count', 'gap_fraction']
    assert onyx_report.ablation_skipped_notes(feats) == []
    assert 'centroid_shift_col_sigma' not in names[2]
    assert set(onyx_report.CENTROID_COLS) == {'centroid_shift_col_sigma', 'centroid_shift_row_sigma', 'centroid_shift_total_sigma', 'centroid_shift_total_arcsec'}
    assert set(onyx_report.SECONDARY_COLS) == {'secondary_depth_ppm', 'secondary_snr', 'secondary_z', 'secondary_phase', 'odd_even_depth_diff_ppm', 'odd_even_sigma', 'odd_even_z'}


def test_ablation_sets_skip_empty_groups_with_notes():
    no_centroid = ['bls_depth_snr', 'transit_count', 'secondary_z']
    sets = onyx_report.ablation_feature_sets(no_centroid)
    assert [n for n, _ in sets][2:] == ['Без вторичного затмения и чёт-нечёт (secondary_z)']
    notes = onyx_report.ablation_skipped_notes(no_centroid)
    assert len(notes) == 2
    assert 'Без центроидных признаков' in notes[0] and 'пропущена' in notes[0]
    assert onyx_report.BOTH_DIAGNOSTIC_NAME in notes[1] and 'пропущена' in notes[1]
    neither = ['bls_depth_snr', 'transit_count']
    assert len(onyx_report.ablation_feature_sets(neither)) == 2
    assert len(onyx_report.ablation_skipped_notes(neither)) == 3
    no_secondary = ['bls_depth_snr', 'centroid_shift_row_sigma']
    assert [n for n, _ in onyx_report.ablation_feature_sets(no_secondary)][2:] == ['Без центроидных признаков (centroid_shift_row_sigma)']
    assert len(onyx_report.ablation_skipped_notes(no_secondary)) == 2


def _section(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


def test_report_ablation_rows_and_explanation(fixes_default_report):
    text = fixes_default_report['text']
    abl = _section(text, 'Абляции признаков', 'Двунаправленная перекрёстная проверка')
    rows = [ln for ln in abl.splitlines() if ln.startswith('| ')]
    names = [r.split('|')[1].strip() for r in rows[2:]]
    assert names[0].startswith('Основная модель')
    assert names[-3].startswith('Без центроидных признаков (centroid_shift_col_sigma, centroid_shift_row_sigma, centroid_shift_total_sigma, centroid_shift_total_arcsec)')
    assert names[-2].startswith('Без вторичного затмения и чёт-нечёт (secondary_depth_ppm, odd_even_depth_diff_ppm)')
    assert names[-1].startswith(onyx_report.BOTH_DIAGNOSTIC_NAME)
    assert 'secondary_snr' not in names[-2] and 'odd_even_sigma' not in names[-2]
    feature_cols = onyx_features.compute_feature_columns(fixes_default_report['ws']['features'], exclude_coverage=True)
    assert f"[признаков: {len(feature_cols) - 6}]" in names[-1]
    assert 'FALSE_POSITIVE' in abl and 'круговую зависимость' in abl
    assert 'пропущена' not in abl


def test_report_states_skipped_ablation_configs(fixes_wilson_report):
    abl = _section(fixes_wilson_report['text'], 'Абляции признаков', 'Двунаправленная перекрёстная проверка')
    assert '| Без вторичного затмения и чёт-нечёт (' in abl
    assert '| Без центроидных признаков' not in abl
    assert 'Конфигурация «Без центроидных признаков»: пропущена' in abl
    assert f"Конфигурация «{onyx_report.BOTH_DIAGNOSTIC_NAME}»: пропущена" in abl
