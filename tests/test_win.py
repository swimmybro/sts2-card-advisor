import json

import numpy as np
import pandas as pd
import pytest

from sts2_advisor.features import OFFER_COLUMNS, SKIP
from sts2_advisor.train import run as run_pick_model
from sts2_advisor.win import (
    adjust_win_chances, auc, calibration, fit_temperature, near_best, outcome_rows,
    pick_probabilities, run,
)
from test_train import config as pick_config, rows


def popular_vs_good_offers(n_runs=400, seed=0):
    """Players always take POPULAR when offered, otherwise a random option.
    Runs are won only if the player ever took GOOD."""
    rng = np.random.default_rng(seed)
    pool = ["POPULAR", "GOOD", "A", "B", "C"]
    out = []
    for r in range(n_runs):
        offers = [list(rng.choice(pool, size=3, replace=False)) for _ in range(5)]
        picks = [("POPULAR" if "POPULAR" in o else rng.choice(o + [SKIP])) for o in offers]
        won = "GOOD" in picks
        source = "mine" if r < 20 else "codex"
        for floor, (cands, pick) in enumerate(zip(offers, picks), start=1):
            out += rows(f"r{r}:{floor}", cands + [SKIP], pick, source=source, floor=floor,
                        act=1, ascension=5, n_offered=3, hp_frac=0.5, run_won=won)
    return pd.DataFrame(out, columns=OFFER_COLUMNS)


def win_config(tmp_path):
    cfg = pick_config(tmp_path)
    cfg["model"] = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 15,
                    "min_data_in_leaf": 5, "num_boost_round": 200,
                    "early_stopping_rounds": 20}
    cfg["output"] = {"model_dir": str(tmp_path / "win")}
    return cfg


def test_auc_known_values():
    assert auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert np.isnan(auc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_calibration_bins():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.05, 0.15, 0.15, 0.95, 0.91])
    bins = {(b["lo"], b["hi"]): b for b in calibration(y, p, n_bins=10)}
    assert bins[(0.1, 0.2)]["n"] == 2 and bins[(0.1, 0.2)]["actual"] == 0.5
    assert bins[(0.9, 1.0)]["predicted"] == pytest.approx(0.93)
    assert sum(b["n"] for b in bins.values()) == 5


def test_outcome_rows_keep_picked_and_finished_only():
    df = pd.DataFrame(rows("a:1", ["X", "Y", SKIP], "Y", run_won=True)
                      + rows("b:1", ["X", SKIP], "X", run_won=None))
    out = outcome_rows(df)
    assert out["offer_id"].tolist() == ["a:1"] and out["candidate"].tolist() == ["Y"]


def test_pick_probabilities_per_reward():
    ids = np.array([0, 0, 1, 1, 1])
    p = pick_probabilities(ids, np.array([1.0, 1.0, 0.0, 0.0, np.log(2)]))
    assert p.tolist() == pytest.approx([0.5, 0.5, 0.25, 0.25, 0.5])
    sharp = pick_probabilities(ids, np.array([2.0, 0.0, 0, 0, 0]), temperature=0.1)
    assert sharp[0] > 0.999


def test_fit_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    n_offers, k, true_t = 3000, 4, 2.0
    scores = rng.normal(0, 3, size=(n_offers, k))
    probs = np.exp(scores / true_t)
    probs /= probs.sum(axis=1, keepdims=True)
    picks = np.array([rng.choice(k, p=row) for row in probs])
    df = pd.DataFrame({"offer_id": np.repeat(np.arange(n_offers), k),
                       "picked": (np.tile(np.arange(k), n_offers) == np.repeat(picks, k))})
    assert fit_temperature(df, scores.ravel()) == pytest.approx(true_t, rel=0.15)


def test_adjust_win_chances_shrinks_rare_options():
    ids = np.array([0, 0])
    win, p = np.array([0.5, 0.9]), np.array([0.95, 0.05])
    adjusted, baseline = adjust_win_chances(ids, win, p, shrink_k=0.05)
    assert baseline.tolist() == pytest.approx([0.52, 0.52])  # 0.95*0.5 + 0.05*0.9
    # common option keeps 95% of its gap; rare one only half
    assert adjusted.tolist() == pytest.approx([0.52 - 0.95 * 0.02, 0.52 + 0.5 * 0.38])


def test_near_best_marks_ties():
    ids = np.array([0, 0, 0, 1, 1])
    vals = np.array([0.40, 0.395, 0.30, 0.6, 0.5])
    assert near_best(ids, vals, margin=0.01).tolist() == [True, True, False, True, False]


def test_win_model_prefers_what_wins_over_what_is_popular(tmp_path):
    popular_vs_good_offers().to_parquet(tmp_path / "offers.parquet")
    run_pick_model(pick_config(tmp_path))  # pick model -> tmp_path/model
    metrics = run(win_config(tmp_path), pick_model=tmp_path / "model")

    test = metrics["test"]
    # A run's other rewards share its label, so single picks can't predict it well;
    # what matters is that GOOD is ranked above POPULAR.
    assert test["outcome"]["auc"] > 0.6
    r = test["ranking"]
    assert "GOOD" in r["favoured_over_players"]
    assert "POPULAR" in r["less_than_players"]
    assert r["agrees_with_pick_model"] < 0.9  # the two models disagree where it matters
    assert (tmp_path / "win" / "model.txt").exists()
    assert "mine" in metrics

    # Reliability check: settings saved for the API, before/after metrics reported
    rel = json.loads((tmp_path / "win" / "reliability.json").read_text())
    assert rel["temperature"] > 0 and rel["shrink_k"] == 0.05
    a = test["ranking_adjusted"]
    assert a["median_spread"] <= r["median_spread"]
    assert 0.0 <= a["tie_rate"] <= 1.0 and 0.0 <= a["raw_top_was_low_confidence"] <= 1.0


def test_no_reliability_without_pick_model(tmp_path):
    popular_vs_good_offers(n_runs=100).to_parquet(tmp_path / "offers.parquet")
    metrics = run(win_config(tmp_path))
    assert "ranking_adjusted" not in metrics["test"]
    assert not (tmp_path / "win" / "reliability.json").exists()
