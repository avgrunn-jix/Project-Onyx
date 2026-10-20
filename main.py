import argparse
import subprocess
import sys
import logging
import os
import tempfile
import shutil

def get_effective_output(out_path, default_base, is_quick, max_sectors, is_smoke):
    base_dir = os.path.join('outputs', '_smoke_test') if is_smoke else ''
    if out_path:
        out = out_path
        if is_smoke and not '_smoke_test' in out:
            out = os.path.join(base_dir, os.path.basename(out))
    else:
        name = f"{default_base}_quick.csv" if is_quick else f"{default_base}.csv"
        out = os.path.join(base_dir, name) if base_dir else name

    if max_sectors > 0:
        if not f'_sec{max_sectors}' in out and not '_sec' in out:
            b, e = os.path.splitext(out)
            out = f"{b}_sec{max_sectors}{e if e else '.csv'}"
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission', default='Kepler', choices=['Kepler', 'TESS'])
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--keep-raw', action='store_true')
    parser.add_argument('--min-free-gb', type=float, default=5.0)
    parser.add_argument('--purge-lk-cache', action='store_true')
    parser.add_argument('--flatten-days', type=float, default=3.0)
    parser.add_argument('--max-signals', type=int, default=4)
    parser.add_argument('--mode', default='train', choices=['train', 'predict', 'score-file', 'retrain', 'inject', 'transfer', 'extract', 'demo'])
    parser.add_argument('--targets-file', default='')
    parser.add_argument('--input-file', default='')
    parser.add_argument('--output', default='')
    parser.add_argument('--train-csv', default='')
    parser.add_argument('--test-csv', default='')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--injection-stars', type=int, default=100)
    parser.add_argument('--injections-per-star', type=int, default=12)
    parser.add_argument('--max-sectors', type=int, default=0, help='[DEBUG ONLY] Limit the number of sectors/quarters downloaded. Warning: alters baseline, features, and detection results; not for production training.')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--star-timeout', type=float, default=1800.0)
    parser.add_argument('--cache-dir', default=tempfile.gettempdir())
    parser.add_argument('--features-file', default='', help='[Optional] Precomputed features CSV to train/retrain ML model directly without feature extraction.')
    parser.add_argument('--smoke-test', action='store_true', help='[TEST ONLY] Run pipeline in isolated smoke test mode, writing all outputs to outputs/_smoke_test/')
    parser.add_argument('--marked-stars', type=int, default=120)
    parser.add_argument('--unmarked-stars', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--showcase-star', default='4852528')
    parser.add_argument('--unmarked-file', default='')
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--max-download-workers', type=int, default=4)
    parser.add_argument('--target-precision', type=float, default=0.90)
    parser.add_argument('--min-depth-ppm', type=float, default=30.0)
    args = parser.parse_args()

    if args.smoke_test and args.min_free_gb == 5.0:
        args.min_free_gb = 1.0
    free_now_gb = shutil.disk_usage(os.path.abspath('.')).free / (1024**3)
    print(f"Свободное место на диске: {free_now_gb:.2f} ГБ (требуемый минимум: {args.min_free_gb:.2f} ГБ)", flush=True)
    if free_now_gb < args.min_free_gb:
        print(f"Ошибка: Недостаточно свободного места на диске ({free_now_gb:.2f} ГБ < {args.min_free_gb:.2f} ГБ). Освободите место или используйте параметр --min-free-gb.", flush=True)
        sys.exit(1)

    logical_cores = os.cpu_count() or 1
    cpu_msg = f"Detected CPU cores: {logical_cores} logical. Default workers: {args.workers}. Note: on systems with Hyper-Threading (SMT), the optimal number of worker processes for CPU-bound BLS computation is typically equal to the number of physical cores."
    print(cpu_msg, flush=True)

    base_outputs = os.path.join('outputs', '_smoke_test') if args.smoke_test else 'outputs'
    os.makedirs(base_outputs, exist_ok=True)
    os.makedirs(os.path.join(base_outputs, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(base_outputs, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_outputs, 'explanations'), exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info(f'Starting Project Onyx pipeline in {args.mode} mode')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    extractor_script = os.path.join(base_dir, '1_data_extractor.py')
    ml_script = os.path.join(base_dir, '2_ml_pipeline.py')

    if args.mode == 'train' or args.mode == 'retrain':
        if args.features_file:
            if not os.path.exists(args.features_file):
                logging.error(f'Features file {args.features_file} not found')
                sys.exit(1)
            logging.info(f'Skipping extraction, running ML training directly on {args.features_file}')
            cmd2 = [sys.executable, ml_script, '--mode', args.mode, '--input', args.features_file]
            if args.smoke_test: cmd2.append('--smoke-test')
            ret2 = subprocess.run(cmd2)
            if ret2.returncode != 0:
                logging.error('ML pipeline failed with code ' + str(ret2.returncode))
                sys.exit(ret2.returncode)
            return

        out_csv = get_effective_output(args.output, 'extracted_features', args.quick, args.max_sectors, args.smoke_test)
        if args.max_sectors > 0:
            logging.warning(f"WARNING: [DEBUG MODE] --max-sectors={args.max_sectors} is active. This alters baseline and features. Output files MUST NOT be used for primary model training!")
            print(f"WARNING: [DEBUG MODE] --max-sectors={args.max_sectors} is active. This alters baseline and features. Output files MUST NOT be used for primary model training!", flush=True)

        cmd1 = [sys.executable, extractor_script, '--mission', args.mission, '--limit', str(args.limit), '--workers', str(args.workers), '--min-free-gb', str(args.min_free_gb), '--flatten-days', str(args.flatten_days), '--max-signals', str(args.max_signals), '--output', out_csv, '--star-timeout', str(args.star_timeout), '--cache-dir', args.cache_dir, '--max-sectors', str(args.max_sectors)]
        if args.targets_file: cmd1.extend(['--targets-file', args.targets_file])
        if args.keep_raw: cmd1.append('--keep-raw')
        if args.purge_lk_cache: cmd1.append('--purge-lk-cache')
        if args.quick: cmd1.append('--quick')
        if args.no_resume: cmd1.append('--no-resume')
        if args.smoke_test: cmd1.append('--smoke-test')
        logging.info('Running extractor')
        ret1 = subprocess.run(cmd1)
        if ret1.returncode != 0:
            logging.error('Extractor failed with code ' + str(ret1.returncode))
            sys.exit(ret1.returncode)
        logging.info('Extractor finished. Running ML pipeline.')
        cmd2 = [sys.executable, ml_script, '--mode', args.mode, '--input', out_csv]
        if args.smoke_test: cmd2.append('--smoke-test')
        ret2 = subprocess.run(cmd2)
        if ret2.returncode != 0:
            logging.error('ML pipeline failed with code ' + str(ret2.returncode))
            sys.exit(ret2.returncode)

    elif args.mode == 'extract':
        default_base = os.path.join(base_outputs, 'tess_features') if args.mission == 'TESS' else 'extracted_features'
        out_csv = get_effective_output(args.output, default_base, args.quick, args.max_sectors, args.smoke_test)
        if args.max_sectors > 0:
            logging.warning(f"WARNING: [DEBUG MODE] --max-sectors={args.max_sectors} is active. This alters baseline and features. Output files MUST NOT be used for primary model training!")
            print(f"WARNING: [DEBUG MODE] --max-sectors={args.max_sectors} is active. This alters baseline and features. Output files MUST NOT be used for primary model training!", flush=True)
        cmd1 = [sys.executable, extractor_script, '--mission', args.mission, '--limit', str(args.limit), '--workers', str(args.workers), '--min-free-gb', str(args.min_free_gb), '--flatten-days', str(args.flatten_days), '--max-signals', str(args.max_signals), '--output', out_csv, '--star-timeout', str(args.star_timeout), '--cache-dir', args.cache_dir, '--max-sectors', str(args.max_sectors)]
        if args.targets_file: cmd1.extend(['--targets-file', args.targets_file])
        if args.keep_raw: cmd1.append('--keep-raw')
        if args.purge_lk_cache: cmd1.append('--purge-lk-cache')
        if args.quick: cmd1.append('--quick')
        if args.no_resume: cmd1.append('--no-resume')
        if args.smoke_test: cmd1.append('--smoke-test')
        logging.info('Running extractor in extract-only mode')
        ret1 = subprocess.run(cmd1)
        if ret1.returncode != 0:
            logging.error('Extractor failed with code ' + str(ret1.returncode))
            sys.exit(ret1.returncode)
        logging.info(f'Feature extraction completed. Output saved to {out_csv}')

    elif args.mode == 'predict':
        if not args.targets_file:
            logging.error('--targets-file required for predict mode')
            sys.exit(1)
        default_pred_features = os.path.join(base_outputs, 'predict_features')
        out_csv = get_effective_output('', default_pred_features, args.quick, args.max_sectors, False)
        cmd1 = [sys.executable, extractor_script, '--mission', args.mission, '--targets-file', args.targets_file, '--workers', str(args.workers), '--min-free-gb', str(args.min_free_gb), '--flatten-days', str(args.flatten_days), '--max-signals', str(args.max_signals), '--output', out_csv, '--star-timeout', str(args.star_timeout), '--cache-dir', args.cache_dir, '--max-sectors', str(args.max_sectors)]
        if args.keep_raw: cmd1.append('--keep-raw')
        if args.quick: cmd1.append('--quick')
        if args.no_resume: cmd1.append('--no-resume')
        if args.smoke_test: cmd1.append('--smoke-test')
        logging.info('Running extractor for prediction targets')
        ret1 = subprocess.run(cmd1)
        if ret1.returncode != 0:
            sys.exit(ret1.returncode)
        logging.info('Running ML prediction')
        out_pred = args.output if args.output else os.path.join(base_outputs, 'new_candidates.csv')
        if args.smoke_test and not '_smoke_test' in out_pred:
            out_pred = os.path.join(base_outputs, os.path.basename(out_pred))
        cmd2 = [sys.executable, ml_script, '--mode', 'predict', '--input', out_csv, '--output', out_pred]
        if args.smoke_test: cmd2.append('--smoke-test')
        subprocess.run(cmd2)

    elif args.mode == 'score-file':
        if not args.input_file:
            logging.error('--input-file required for score-file mode')
            sys.exit(1)
        out_score = args.output if args.output else os.path.join(base_outputs, 'scored_file.csv')
        if args.smoke_test and not '_smoke_test' in out_score:
            out_score = os.path.join(base_outputs, os.path.basename(out_score))
        cmd2 = [sys.executable, ml_script, '--mode', 'score-file', '--input', args.input_file, '--output', out_score]
        if args.smoke_test: cmd2.append('--smoke-test')
        subprocess.run(cmd2)

    elif args.mode == 'inject':
        cmd1 = [sys.executable, extractor_script, '--mission', args.mission, '--inject', '--injection-stars', str(args.injection_stars), '--injections-per-star', str(args.injections_per_star), '--workers', str(args.workers), '--star-timeout', str(args.star_timeout), '--cache-dir', args.cache_dir, '--max-sectors', str(args.max_sectors)]
        if args.targets_file: cmd1.extend(['--targets-file', args.targets_file])
        if args.output:
            inj_tgt = os.path.join(base_outputs, os.path.basename(args.output)) if (args.smoke_test and not '_smoke_test' in args.output) else args.output
            cmd1.extend(['--injection-output', inj_tgt])
        if args.quick: cmd1.append('--quick')
        if args.no_resume: cmd1.append('--no-resume')
        if args.smoke_test: cmd1.append('--smoke-test')
        logging.info('Running injection tests')
        subprocess.run(cmd1)

    elif args.mode == 'transfer':
        train_csv = args.train_csv if args.train_csv else (args.input_file if args.input_file else 'extracted_features.csv')
        test_csv = args.test_csv if args.test_csv else (args.output if args.output else os.path.join(base_outputs, 'tess_features.csv'))
        if args.smoke_test and not '_smoke_test' in test_csv:
            test_csv = os.path.join(base_outputs, os.path.basename(test_csv))
        cmd = [sys.executable, ml_script, '--mode', 'transfer', '--train-csv', train_csv, '--test-csv', test_csv]
        if args.smoke_test: cmd.append('--smoke-test')
        subprocess.run(cmd)

    elif args.mode == 'demo':
        demo_script = os.path.join(base_dir, '3_demo_pipeline.py')
        eff_min_free = 1.0 if args.smoke_test and args.min_free_gb == 5.0 else args.min_free_gb
        cmd = [sys.executable, demo_script, '--marked-stars', str(args.marked_stars), '--unmarked-stars', str(args.unmarked_stars), '--seed', str(args.seed), '--showcase-star', str(args.showcase_star), '--workers', str(args.workers), '--max-download-workers', str(args.max_download_workers), '--target-precision', str(args.target_precision), '--min-depth-ppm', str(args.min_depth_ppm), '--star-timeout', str(args.star_timeout), '--cache-dir', args.cache_dir, '--flatten-days', str(args.flatten_days), '--max-signals', str(args.max_signals), '--min-free-gb', str(eff_min_free)]
        if args.output_dir:
            cmd.extend(['--output-dir', args.output_dir])
        if args.unmarked_file:
            cmd.extend(['--unmarked-file', args.unmarked_file])
        if args.quick:
            cmd.append('--quick')
        if args.smoke_test:
            cmd.append('--smoke-test')
        logging.info('Running demonstration pipeline')
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            logging.error('Demo pipeline failed with code ' + str(ret.returncode))
            sys.exit(ret.returncode)

if __name__ == '__main__':
    main()
