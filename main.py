import argparse
import subprocess
import sys
import logging
import os
import tempfile

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
    parser.add_argument('--mode', default='train', choices=['train', 'predict', 'score-file', 'retrain', 'inject', 'transfer', 'extract'])
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
    args = parser.parse_args()

    logical_cores = os.cpu_count() or 1
    cpu_msg = f"Detected CPU cores: {logical_cores} logical. Default workers: {args.workers}. Note: on systems with Hyper-Threading (SMT), the optimal number of worker processes for CPU-bound BLS computation is typically equal to the number of physical cores."
    print(cpu_msg, flush=True)

    os.makedirs('outputs', exist_ok=True)
    os.makedirs(os.path.join('outputs', 'figures'), exist_ok=True)
    os.makedirs(os.path.join('outputs', 'models'), exist_ok=True)
    os.makedirs(os.path.join('outputs', 'explanations'), exist_ok=True)
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
            ret2 = subprocess.run(cmd2)
            if ret2.returncode != 0:
                logging.error('ML pipeline failed with code ' + str(ret2.returncode))
                sys.exit(ret2.returncode)
            return

        out_csv = args.output if args.output else ('extracted_features_quick.csv' if args.quick else 'extracted_features.csv')
        if args.max_sectors > 0:
            logging.warning(f"WARNING: [DEBUG MODE] --max-sectors={args.max_sectors} is active. This alters baseline and features. Output files MUST NOT be used for primary model training!")
            print(f"WARNING: [DEBUG MODE] --max-sectors={args.max_sectors} is active. This alters baseline and features. Output files MUST NOT be used for primary model training!", flush=True)
            if out_csv == 'extracted_features.csv':
                out_csv = f'extracted_features_sec{args.max_sectors}.csv'
            elif not f'_sec{args.max_sectors}' in out_csv and not '_sec' in out_csv:
                b_o, e_o = os.path.splitext(out_csv)
                out_csv = f"{b_o}_sec{args.max_sectors}{e_o if e_o else '.csv'}"
        cmd1 = [sys.executable, extractor_script, '--mission', args.mission, '--limit', str(args.limit), '--workers', str(args.workers), '--min-free-gb', str(args.min_free_gb), '--flatten-days', str(args.flatten_days), '--max-signals', str(args.max_signals), '--output', out_csv, '--star-timeout', str(args.star_timeout), '--cache-dir', args.cache_dir, '--max-sectors', str(args.max_sectors)]
        if args.targets_file: cmd1.extend(['--targets-file', args.targets_file])
        if args.keep_raw: cmd1.append('--keep-raw')
        if args.purge_lk_cache: cmd1.append('--purge-lk-cache')
        if args.quick: cmd1.append('--quick')
        if args.no_resume: cmd1.append('--no-resume')
        logging.info('Running extractor')
        ret1 = subprocess.run(cmd1)
        if ret1.returncode != 0:
            logging.error('Extractor failed with code ' + str(ret1.returncode))
            sys.exit(ret1.returncode)
        logging.info('Extractor finished. Running ML pipeline.')
        cmd2 = [sys.executable, ml_script, '--mode', args.mode, '--input', out_csv]
        ret2 = subprocess.run(cmd2)
        if ret2.returncode != 0:
            logging.error('ML pipeline failed with code ' + str(ret2.returncode))
            sys.exit(ret2.returncode)

    elif args.mode == 'extract':
        out_csv = args.output if args.output else (os.path.join('outputs', 'tess_features.csv') if args.mission == 'TESS' else ('extracted_features_quick.csv' if args.quick else 'extracted_features.csv'))
        if args.max_sectors > 0:
            logging.warning(f"WARNING: [DEBUG MODE] --max-sectors={args.max_sectors} is active. This alters baseline and features. Output files MUST NOT be used for primary model training!")
            print(f"WARNING: [DEBUG MODE] --max-sectors={args.max_sectors} is active. This alters baseline and features. Output files MUST NOT be used for primary model training!", flush=True)
            if out_csv == 'extracted_features.csv':
                out_csv = f'extracted_features_sec{args.max_sectors}.csv'
            elif not f'_sec{args.max_sectors}' in out_csv and not '_sec' in out_csv:
                b_o, e_o = os.path.splitext(out_csv)
                out_csv = f"{b_o}_sec{args.max_sectors}{e_o if e_o else '.csv'}"
        cmd1 = [sys.executable, extractor_script, '--mission', args.mission, '--limit', str(args.limit), '--workers', str(args.workers), '--min-free-gb', str(args.min_free_gb), '--flatten-days', str(args.flatten_days), '--max-signals', str(args.max_signals), '--output', out_csv, '--star-timeout', str(args.star_timeout), '--cache-dir', args.cache_dir, '--max-sectors', str(args.max_sectors)]
        if args.targets_file: cmd1.extend(['--targets-file', args.targets_file])
        if args.keep_raw: cmd1.append('--keep-raw')
        if args.purge_lk_cache: cmd1.append('--purge-lk-cache')
        if args.quick: cmd1.append('--quick')
        if args.no_resume: cmd1.append('--no-resume')
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
        out_csv = os.path.join('outputs', 'predict_features.csv')
        cmd1 = [sys.executable, extractor_script, '--mission', args.mission, '--targets-file', args.targets_file, '--workers', str(args.workers), '--min-free-gb', str(args.min_free_gb), '--flatten-days', str(args.flatten_days), '--max-signals', str(args.max_signals), '--output', out_csv, '--star-timeout', str(args.star_timeout), '--cache-dir', args.cache_dir, '--max-sectors', str(args.max_sectors)]
        if args.keep_raw: cmd1.append('--keep-raw')
        if args.quick: cmd1.append('--quick')
        if args.no_resume: cmd1.append('--no-resume')
        logging.info('Running extractor for prediction targets')
        ret1 = subprocess.run(cmd1)
        if ret1.returncode != 0:
            sys.exit(ret1.returncode)
        logging.info('Running ML prediction')
        out_pred = args.output if args.output else os.path.join('outputs', 'new_candidates.csv')
        cmd2 = [sys.executable, ml_script, '--mode', 'predict', '--input', out_csv, '--output', out_pred]
        subprocess.run(cmd2)

    elif args.mode == 'score-file':
        if not args.input_file:
            logging.error('--input-file required for score-file mode')
            sys.exit(1)
        out_score = args.output if args.output else os.path.join('outputs', 'scored_file.csv')
        cmd2 = [sys.executable, ml_script, '--mode', 'score-file', '--input', args.input_file, '--output', out_score]
        subprocess.run(cmd2)

    elif args.mode == 'inject':
        cmd1 = [sys.executable, extractor_script, '--mission', args.mission, '--inject', '--injection-stars', str(args.injection_stars), '--injections-per-star', str(args.injections_per_star), '--workers', str(args.workers), '--star-timeout', str(args.star_timeout), '--cache-dir', args.cache_dir, '--max-sectors', str(args.max_sectors)]
        if args.targets_file: cmd1.extend(['--targets-file', args.targets_file])
        if args.output: cmd1.extend(['--injection-output', args.output])
        if args.quick: cmd1.append('--quick')
        if args.no_resume: cmd1.append('--no-resume')
        logging.info('Running injection tests')
        subprocess.run(cmd1)

    elif args.mode == 'transfer':
        train_csv = args.train_csv if args.train_csv else (args.input_file if args.input_file else 'extracted_features.csv')
        test_csv = args.test_csv if args.test_csv else (args.output if args.output else os.path.join('outputs', 'tess_features.csv'))
        cmd = [sys.executable, ml_script, '--mode', 'transfer', '--train-csv', train_csv, '--test-csv', test_csv]
        subprocess.run(cmd)

if __name__ == '__main__':
    main()
