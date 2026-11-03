import numpy as np
import pandas as pd


def make_synthetic_catalog(n_marked_pool=80, n_unmarked_pool=200, showcase_star="4852528", seed=7):
    marked_ids = [showcase_star] + [str(2000000 + i * 1103) for i in range(n_marked_pool - 1)]
    rows = []
    dispositions = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]
    for idx, tid in enumerate(marked_ids):
        lbl = "CONFIRMED" if tid == showcase_star else dispositions[idx % len(dispositions)]
        n_sig = 3 if idx % 4 == 0 else 1
        for s in range(n_sig):
            rows.append({
                "target_id": tid,
                "label": lbl,
                "catalog_period": 2.0 + s * 3.5,
            })
    labels_df = pd.DataFrame(rows)
    marked_info = {
        tid: (float(11.5 + (i % 9) * 0.4), int(12 + (i % 6)))
        for i, tid in enumerate(marked_ids)
    }
    pool_rows = []
    for j in range(n_unmarked_pool):
        uid = str(1500000 + j * 701)
        if uid in marked_info:
            continue
        q_cnt = 12 + (j % 6)
        pool_rows.append({
            "kepid": uid,
            "kepmag": 11.3 + (j % 10) * 0.4,
            "st_quarters": "1" * q_cnt + "0" * (17 - q_cnt),
            "dataspan": 1350.0,
        })
    pool_df = pd.DataFrame(pool_rows)
    return labels_df, marked_info, pool_df


def test_split_stars_pairwise_disjoint_and_balanced(demo):
    showcase = "4852528"
    labels_df, marked_info, pool_df = make_synthetic_catalog(showcase_star=showcase)
    marked_ids, star_prim = demo.get_marked_stars(labels_df, 40, seed=42, showcase_star=showcase)
    unmarked_ids, _ = demo.match_unmarked_stars(marked_ids, marked_info, pool_df, 40, 42, labels_df=labels_df)
    tr_m, ctrl_m, tr_u, ctrl_u = demo.split_stars(marked_ids, unmarked_ids, star_prim, showcase, seed=42)

    assert len(tr_m) == len(ctrl_m) == len(tr_u) == len(ctrl_u) == 20
    sets = [set(tr_m), set(ctrl_m), set(tr_u), set(ctrl_u)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j])

    train_stars = set(tr_m) | set(tr_u)
    ctrl_stars = set(ctrl_m) | set(ctrl_u)
    assert train_stars.isdisjoint(ctrl_stars)
    assert showcase in ctrl_m
    assert showcase not in train_stars


def test_multi_signal_stars_never_split_across_train_and_control(demo):
    showcase = "4852528"
    labels_df, marked_info, pool_df = make_synthetic_catalog(showcase_star=showcase)
    marked_ids, star_prim = demo.get_marked_stars(labels_df, 30, seed=99, showcase_star=showcase)
    unmarked_ids, _ = demo.match_unmarked_stars(marked_ids, marked_info, pool_df, 30, 99, labels_df=labels_df)
    tr_m, ctrl_m, tr_u, ctrl_u = demo.split_stars(marked_ids, unmarked_ids, star_prim, showcase, seed=99)

    all_stars = tr_m + ctrl_m + tr_u + ctrl_u
    sig_rows = []
    for sid in all_stars:
        is_marked = sid in set(marked_ids)
        for sig_idx in range(3):
            sig_rows.append({
                "signal_id": f"{sid}_{sig_idx}",
                "target_id": sid,
                "signal_index": sig_idx,
                "label": star_prim.get(sid, "UNMATCHED") if is_marked else "UNMATCHED",
                "match_status": "MATCHED" if is_marked else "UNMATCHED",
                "target_bin": 1 if star_prim.get(sid) in ("CONFIRMED", "CANDIDATE") else 0,
                "bls_period_d": 2.0 + sig_idx,
                "bls_depth_snr": 12.0 + sig_idx,
            })
    df_features = pd.DataFrame(sig_rows)

    train_stars_set = set(tr_m + tr_u)
    ctrl_stars_set = set(ctrl_m + ctrl_u)
    df_train_all = df_features[df_features["target_id"].isin(train_stars_set)]
    df_ctrl_all = df_features[df_features["target_id"].isin(ctrl_stars_set)]

    train_targets = set(df_train_all["target_id"].unique())
    ctrl_targets = set(df_ctrl_all["target_id"].unique())
    assert train_targets.isdisjoint(ctrl_targets)
    assert len(df_train_all) + len(df_ctrl_all) == len(df_features)

    for tid, grp in df_features.groupby("target_id"):
        in_tr = grp["signal_id"].isin(set(df_train_all["signal_id"])).all()
        in_ctrl = grp["signal_id"].isin(set(df_ctrl_all["signal_id"])).all()
        assert in_tr ^ in_ctrl

    df_ctrl_m_for_b = df_ctrl_all[(df_ctrl_all["target_id"].isin(set(ctrl_m))) & (df_ctrl_all["target_id"] != showcase)]
    assert showcase not in set(df_ctrl_m_for_b["target_id"])
    assert set(df_ctrl_m_for_b["target_id"]).isdisjoint(train_targets)


def test_train_and_eval_pipeline_group_cv_keeps_star_signals_together(demo, monkeypatch):
    rng = np.random.RandomState(123)
    train_stars = [f"tr_star_{i}" for i in range(10)]
    ctrl_stars = [f"ctrl_star_{i}" for i in range(6)]
    tr_rows = []
    for i, sid in enumerate(train_stars):
        cls = 1 if i % 2 == 0 else 0
        for k in range(3):
            tr_rows.append({
                "signal_id": f"{sid}_{k}",
                "target_id": sid,
                "target_bin": cls,
                "f1": float(rng.normal(cls * 2.0, 0.5) + 2.0),
                "f2": float(rng.normal(0.0, 1.0) + 2.0),
            })
    ctrl_rows = []
    for i, sid in enumerate(ctrl_stars):
        cls = 1 if i % 2 == 0 else 0
        for k in range(2):
            ctrl_rows.append({
                "signal_id": f"{sid}_{k}",
                "target_id": sid,
                "target_bin": cls,
                "f1": float(rng.normal(cls * 2.0, 0.5) + 2.0),
                "f2": float(rng.normal(0.0, 1.0) + 2.0),
            })
    df_tr = pd.DataFrame(tr_rows)
    df_ctrl = pd.DataFrame(ctrl_rows)

    recorded_folds = []
    orig_gkf = demo.GroupKFold

    class SpyGroupKFold:
        def __init__(self, n_splits=5):
            self.inner = orig_gkf(n_splits=n_splits)

        def split(self, X, y=None, groups=None):
            assert groups is not None
            real_star_ids = df_tr["target_id"].values
            for tr_idx, val_idx in self.inner.split(X, y, groups=groups):
                tr_g = set(real_star_ids[tr_idx])
                val_g = set(real_star_ids[val_idx])
                recorded_folds.append((tr_g, val_g))
                yield tr_idx, val_idx

    monkeypatch.setattr(demo, "GroupKFold", SpyGroupKFold)
    df_res, model, _, _, _ = demo.train_and_eval_pipeline(df_tr, df_ctrl, ["f1", "f2"], seed=42)
    assert model is not None
    assert len(df_res) == len(df_ctrl)
    assert len(recorded_folds) == 5
    for tr_g, val_g in recorded_folds:
        assert tr_g.isdisjoint(val_g)
        assert tr_g.isdisjoint(set(ctrl_stars))
        assert val_g.isdisjoint(set(ctrl_stars))
