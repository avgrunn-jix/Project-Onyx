from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def test_labels_file_has_koi_depth_column():
    path = ROOT / 'data' / 'labels' / 'kepler_labels.csv'
    df = pd.read_csv(path, dtype={'target_id': str})
    assert list(df.columns) == ['target_id', 'label', 'catalog_period', 'koi_depth', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    depth = pd.to_numeric(df['koi_depth'], errors='coerce')
    assert depth.notna().mean() > 0.9
    assert (depth.dropna() >= 0).all()
    assert set(df['label']) <= {'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'}
