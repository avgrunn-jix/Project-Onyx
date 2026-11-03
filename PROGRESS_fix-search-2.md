# PROGRESS fix-search-2

## Выполнено
- П0: ветка fix-search-2 создана от main (49d286193dc61901a9638ac31e006c32bad18dbf). Незакоммичено на main: только неотслеживаемая папка paper_notes/ (не трогаю). pytest 9.2.0 установлен в venv; пакеты requirements.txt уже были установлены.- П1: CLIP_UPWARD_ONLY + clip_outliers() в 1_data_extractor.py, оба вызова remove_outliers заменены (инжекция и process_single_star). Проверено: py_compile, pytest tests/test_clip_outliers.py (3 passed). Хеш: см. git log (коммит «П1»).- П2: RESOLVE_ALIASES + resolve_alias_period(), is_alias_duplicate(), _box_depth(), _power_near(), _class_depths_ok() в 1_data_extractor.py; вызов в цикле BLS process_single_star перед признаками и маской; alias_corrections последним столбцом диагностики (4 места diag_row); лог флагов в main; append_to_csv_flush выравнивает строки по заголовку существующего файла, если в нём нет нового столбца. Проверено: py_compile, pytest (12 passed); scratch-сравнение старой версии (6b886c5) и новой при RESOLVE_ALIASES=False на синтетической звезде: признаки идентичны (max abs diff 0.0).

## Сейчас
- П3: compute_feature_columns в общем модуле. Следующий шаг: найти exclude_cols в 2_ml_pipeline.py и 3_demo_pipeline.py, создать onyx_features.py, тест эквивалентности.

## Решения
- outputs/ на этом компьютере есть; не читаю и не меняю.
- П2: скорректированный период подставляется для всего сигнала (bls_period_d, bls_epoch, transit_count, duty_cycle, все признаки по свёртке и маска); bls_depth_ppm при поправке = взвешенная box-глубина на новом периоде; bls_sde, bls_depth_snr, bls_power_peak остаются от пика BLS. Дубликат (кратность 1,2,3,1/2,1/3, допуск 1%) не записывается, но его прохождения маскируются, итерация расходуется. signal_index/signal_id = номер записанного сигнала (при False совпадает с прежним sig_idx). Поправки считаются для каждого применённого изменения периода, включая отброшенные потом как дубликат.
- Пороги: ALIAS_DEPTH_SHORT=0.8, ALIAS_DEPTH_LONG=1.25, ALIAS_CLASS_DEPTH=0.5 (каждый класс прохождений P/k глубиной >= 0.5 глубины кандидата — «когерентный ряд»), ALIAS_POWER_FRACTION=0.5, ALIAS_TOL=0.01. Если период-проверка вне сетки BLS (нет мощности), поправка не делается.

## Вопросы
