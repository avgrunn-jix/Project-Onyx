import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

import e2_compare
import exp_common as ec


def test_risk_when_only_least_confident_is_wrong():
    y = np.array([1, 1, 1, 1])
    p = np.array([0.9, 0.8, 0.7, 0.4])
    assert ec.selective_risk(y, p, 0.5) == 0.0
    assert ec.selective_risk(y, p, 0.7) == 0.0
    assert ec.selective_risk(y, p, 0.9) == pytest.approx(0.25)
    assert ec.aurc(y, p) == pytest.approx(0.0625)
    assert ec.augrc(y, p) == pytest.approx(0.0625)


def test_risk_when_most_confident_is_wrong():
    y = np.array([1, 1, 1, 1])
    p = np.array([0.1, 0.8, 0.7, 0.6])
    assert ec.selective_risk(y, p, 0.5) == pytest.approx(0.5)
    assert ec.aurc(y, p) == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)
    assert ec.augrc(y, p) == pytest.approx(0.25)
    cov, risk = ec.risk_coverage_curve(y, p)
    assert list(cov) == [0.25, 0.5, 0.75, 1.0]
    assert risk == pytest.approx([1, 1 / 2, 1 / 3, 1 / 4])


def test_perfect_and_all_wrong():
    y = np.array([0, 1, 0, 1, 1])
    assert ec.aurc(y, y * 0.8 + 0.1) == 0.0
    assert ec.augrc(y, y * 0.8 + 0.1) == 0.0
    assert ec.aurc(y, 0.9 - y * 0.8) == pytest.approx(1.0)
    assert ec.augrc(y, 0.9 - y * 0.8) == pytest.approx(np.mean(np.arange(1, 6) / 5))


def test_confidence_ties_are_deterministic():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.7, 0.3, 0.3, 0.7])
    assert list(ec.confidence_order(p)) == [0, 1, 2, 3]
    assert ec.selective_risk(y, p, 0.5) == 0.0
    assert ec.selective_risk(y, p, 0.75) == pytest.approx(1 / 3)
    assert list(ec.confidence_order(p, tiebreak=[3, 2, 1, 0])) == [3, 2, 1, 0]


def test_coverage_rounds_up():
    assert ec.accepted_count(10, 0.7) == 7
    assert ec.accepted_count(299, 0.7) == 210
    assert ec.accepted_count(3, 0.5) == 2
    assert ec.accepted_count(5, 0.0) == 1


def test_all_metrics_matches_single_functions():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    p = np.clip(y * 0.3 + rng.rand(200) * 0.7, 0, 1)
    m = ec.all_metrics(y, p)
    assert m['aurc'] == pytest.approx(ec.aurc(y, p))
    assert m['augrc'] == pytest.approx(ec.augrc(y, p))
    for c in ec.COVERAGES:
        assert m[f"risk@{c:.2f}"] == pytest.approx(ec.selective_risk(y, p, c))
    assert m['brier'] == pytest.approx(np.mean((p - y) ** 2))


def test_star_bootstrap_keeps_star_signals_together():
    groups = np.array(['a', 'b', 'b', 'c', 'c', 'c', 'd'])
    sizes = {'a': 1, 'b': 2, 'c': 3, 'd': 1}
    for idx in ec.star_bootstrap_indices(groups, 200, 5):
        picked = groups[idx]
        n_stars = 0
        for g, size in sizes.items():
            cnt = int(np.sum(picked == g))
            assert cnt % size == 0
            n_stars += cnt // size
            members = np.where(groups == g)[0]
            counts = [int(np.sum(idx == i)) for i in members]
            assert len(set(counts)) == 1
        assert n_stars == 4


def test_star_bootstrap_reproducible():
    groups = np.repeat(np.arange(20), 2).astype(str)
    a = [list(i) for i in ec.star_bootstrap_indices(groups, 10, 42)]
    b = [list(i) for i in ec.star_bootstrap_indices(groups, 10, 42)]
    assert a == b


def test_paired_bootstrap_uses_same_samples():
    rng = np.random.RandomState(1)
    groups = np.repeat(np.arange(30), 2).astype(str)
    y = rng.randint(0, 2, 60)
    p = rng.rand(60)
    draws = ec.bootstrap_metrics(y, {'a': [p], 'b': [p.copy()]}, groups, n_boot=50, seed=3)
    assert np.allclose(draws['a']['aurc'], draws['b']['aurc'])
    assert e2_compare.paired_ci(draws, 'a', 'b', 'aurc') == (0.0, 0.0)


def toy_summary():
    return {
        'lr': {'risk@0.70': 0.05, 'aurc': 0.04, 'fold_auc_std': 0.02, 'fit_median_s': 0.01},
        'rf': {'risk@0.70': 0.03, 'aurc': 0.03, 'fold_auc_std': 0.02, 'fit_median_s': 0.5},
        'xgb': {'risk@0.70': 0.04, 'aurc': 0.05, 'fold_auc_std': 0.03, 'fit_median_s': 0.6},
    }


def test_pareto_front():
    assert e2_compare.pareto_front(toy_summary()) == ['lr', 'rf']


def test_decision_prefers_simpler_when_indistinguishable():
    s = toy_summary()
    front = e2_compare.pareto_front(s)
    near = {'lr': {'risk@0.70': np.linspace(-0.05, 0.1, 101)}, 'rf': {'risk@0.70': np.linspace(0.0, 0.08, 101)}}
    leader, tied, chosen = e2_compare.decide(s, near, front)
    assert leader == 'rf' and set(tied) == {'lr', 'rf'} and chosen == 'lr'
    far = {'lr': {'risk@0.70': np.full(101, 0.05)}, 'rf': {'risk@0.70': np.full(101, 0.03)}}
    leader, tied, chosen = e2_compare.decide(s, far, front)
    assert tied == ['rf'] and chosen == 'rf'


def test_trial_budget_is_twelve():
    for f in e2_compare.TUNED:
        assert len(e2_compare.trial_configs(f)) == 12
    assert e2_compare.trial_configs('xgb_ens') == []
    assert e2_compare.trial_configs('rf') == e2_compare.trial_configs('rf')
