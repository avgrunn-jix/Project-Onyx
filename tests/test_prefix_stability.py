import pandas as pd


def build_synthetic_inputs(n_marked_pool=150, n_unmarked_pool=400, showcase="4852528"):
    marked_all = [showcase] + [str(3000000 + i * 997) for i in range(n_marked_pool - 1)]
    dispositions = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]
    label_rows = []
    for i, tid in enumerate(marked_all):
        lbl = "CONFIRMED" if tid == showcase else dispositions[i % 3]
        label_rows.append({"target_id": tid, "label": lbl, "catalog_period": 3.0 + (i % 15)})
    labels_df = pd.DataFrame(label_rows)

    marked_info = {
        tid: (float(11.2 + (i % 11) * 0.35), int(11 + (i % 7)))
        for i, tid in enumerate(marked_all)
    }

    pool_rows = []
    for j in range(n_unmarked_pool):
        uid = str(2500000 + j * 613)
        if uid in marked_info:
            continue
        q_cnt = 11 + (j % 7)
        pool_rows.append({
            "kepid": uid,
            "kepmag": 11.0 + (j % 12) * 0.35,
            "st_quarters": "1" * q_cnt + "0" * (17 - q_cnt),
            "dataspan": 1400.0,
        })
    pool_df = pd.DataFrame(pool_rows)
    return labels_df, marked_info, pool_df


def test_marked_stars_prefix_stability(demo):
    showcase = "4852528"
    labels_df, _, _ = build_synthetic_inputs(showcase=showcase)
    sizes = [4, 10, 24, 50, 120]
    full_marked, _ = demo.get_marked_stars(labels_df, sizes[-1], seed=42, showcase_star=showcase)
    assert full_marked[1] == showcase
    for n in sizes:
        sub_marked, _ = demo.get_marked_stars(labels_df, n, seed=42, showcase_star=showcase)
        assert len(sub_marked) == n
        assert sub_marked == full_marked[:n]


def test_unmarked_matching_prefix_stability(demo):
    showcase = "4852528"
    labels_df, marked_info, pool_df = build_synthetic_inputs(showcase=showcase)
    sizes = [6, 16, 36, 80]
    marked_max, _ = demo.get_marked_stars(labels_df, sizes[-1], seed=42, showcase_star=showcase)
    unmarked_max, _ = demo.match_unmarked_stars(marked_max, marked_info, pool_df, sizes[-1], 42, labels_df=labels_df)
    for n in sizes:
        marked_n, _ = demo.get_marked_stars(labels_df, n, seed=42, showcase_star=showcase)
        unmarked_n, _ = demo.match_unmarked_stars(marked_n, marked_info, pool_df, n, 42, labels_df=labels_df)
        assert len(unmarked_n) == n
        assert unmarked_n == unmarked_max[:n]


def test_split_stars_prefix_stability(demo):
    showcase = "4852528"
    labels_df, marked_info, pool_df = build_synthetic_inputs(showcase=showcase)
    sizes = [10, 20, 40, 80]
    results = {}
    for n in sizes:
        m_ids, star_prim = demo.get_marked_stars(labels_df, n, seed=42, showcase_star=showcase)
        u_ids, _ = demo.match_unmarked_stars(m_ids, marked_info, pool_df, n, 42, labels_df=labels_df)
        results[n] = demo.split_stars(m_ids, u_ids, star_prim, showcase, seed=42)

    tr_m_max, ctrl_m_max, tr_u_max, ctrl_u_max = results[sizes[-1]]
    assert ctrl_m_max[0] == showcase
    for n in sizes:
        k = n // 2
        tr_m, ctrl_m, tr_u, ctrl_u = results[n]
        assert tr_m == tr_m_max[:k]
        assert ctrl_m == ctrl_m_max[:k]
        assert tr_u == tr_u_max[:k]
        assert ctrl_u == ctrl_u_max[:k]
