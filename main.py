import argparse
import logging
import os
import subprocess
import sys
import json

def setup_logger():
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler('logs/main.log')
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission', default='Kepler', choices=['Kepler', 'TESS'])
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    logger = setup_logger()
    logger.info('Starting Project Onyx pipeline')
    cmd1 = [
        sys.executable, '1_data_extractor.py',
        '--mission', args.mission,
        '--limit', str(args.limit),
        '--workers', str(args.workers)
    ]
    logger.info('Running extractor')
    res1 = subprocess.run(cmd1, check=False)
    if res1.returncode != 0:
        logger.error('Extractor failed with code ' + str(res1.returncode))
        sys.exit(1)
    if not os.path.exists('extracted_features.csv') or os.path.getsize('extracted_features.csv') == 0:
        logger.error('No features extracted. Exiting.')
        sys.exit(1)
    logger.info('Extractor finished. Running ML pipeline.')
    cmd2 = [sys.executable, '2_ml_pipeline.py', '--input', 'extracted_features.csv']
    res2 = subprocess.run(cmd2, check=False)
    if res2.returncode != 0:
        logger.error('ML pipeline failed with code ' + str(res2.returncode))
        sys.exit(1)
    if os.path.exists('outputs/metrics.json'):
        with open('outputs/metrics.json', 'r') as f:
            metrics = json.load(f)
        logger.info('Pipeline finished successfully. Metrics:')
        for k, v in metrics.items():
            logger.info(k + ': ' + str(round(v, 4)))
    else:
        logger.warning('Pipeline finished but no metrics.json found.')
    with open('extracted_features.csv', 'r') as f:
        count = sum(1 for _ in f) - 1
    logger.info('Total stars processed and saved: ' + str(count))

if __name__ == '__main__':
    main()
