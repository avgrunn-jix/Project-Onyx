import argparse
import subprocess
import sys
import logging
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission', default='Kepler', choices=['Kepler', 'TESS'])
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--keep-raw', action='store_true')
    parser.add_argument('--min-free-gb', type=float, default=5.0)
    parser.add_argument('--purge-lk-cache', action='store_true')
    parser.add_argument('--flatten-days', type=float, default=3.0)
    parser.add_argument('--max-signals', type=int, default=4)
    parser.add_argument('--mode', default='train', choices=['train', 'predict', 'score-file', 'retrain', 'inject', 'transfer'])
    parser.add_argument('--targets-file', default='')
    parser.add_argument('--input-file', default='')
    parser.add_argument('--injection-stars', type=int, default=100)
    parser.add_argument('--injections-per-star', type=int, default=12)
    parser.add_argument('--max-sectors', type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info(f'Starting Project Onyx pipeline in {args.mode} mode')

    if args.mode == 'train' or args.mode == 'retrain':
        out_csv = 'extracted_features.csv'
        cmd1 = [sys.executable, '1_data_extractor.py', '--mission', args.mission, '--limit', str(args.limit), '--workers', str(args.workers), '--min-free-gb', str(args.min_free_gb), '--flatten-days', str(args.flatten_days), '--max-signals', str(args.max_signals), '--output', out_csv]
        if args.keep_raw: cmd1.append('--keep-raw')
        if args.purge_lk_cache: cmd1.append('--purge-lk-cache')
        logging.info('Running extractor')
        ret1 = subprocess.run(cmd1)
        if ret1.returncode != 0:
            logging.error('Extractor failed with code ' + str(ret1.returncode))
            sys.exit(ret1.returncode)
        logging.info('Extractor finished. Running ML pipeline.')
        cmd2 = [sys.executable, '2_ml_pipeline.py', '--mode', args.mode, '--input', out_csv]
        ret2 = subprocess.run(cmd2)
        if ret2.returncode != 0:
            logging.error('ML pipeline failed with code ' + str(ret2.returncode))
            sys.exit(ret2.returncode)

    elif args.mode == 'predict':
        if not args.targets_file:
            logging.error('--targets-file required for predict mode')
            sys.exit(1)
        out_csv = 'outputs/predict_features.csv'
        cmd1 = [sys.executable, '1_data_extractor.py', '--mission', args.mission, '--targets-file', args.targets_file, '--workers', str(args.workers), '--min-free-gb', str(args.min_free_gb), '--flatten-days', str(args.flatten_days), '--max-signals', str(args.max_signals), '--output', out_csv]
        if args.keep_raw: cmd1.append('--keep-raw')
        logging.info('Running extractor for prediction targets')
        ret1 = subprocess.run(cmd1)
        if ret1.returncode != 0:
            sys.exit(ret1.returncode)
        logging.info('Running ML prediction')
        cmd2 = [sys.executable, '2_ml_pipeline.py', '--mode', 'predict', '--input', out_csv]
        subprocess.run(cmd2)

    elif args.mode == 'score-file':
        if not args.input_file:
            logging.error('--input-file required for score-file mode')
            sys.exit(1)
        cmd2 = [sys.executable, '2_ml_pipeline.py', '--mode', 'score-file', '--input', args.input_file]
        subprocess.run(cmd2)

    elif args.mode == 'inject':
        cmd1 = [sys.executable, '1_data_extractor.py', '--mission', args.mission, '--inject', '--injection-stars', str(args.injection_stars), '--injections-per-star', str(args.injections_per_star), '--workers', str(args.workers)]
        logging.info('Running injection tests')
        subprocess.run(cmd1)

    elif args.mode == 'transfer':
        pass

if __name__ == '__main__':
    main()
