import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from sts2_advisor.features import OFFER_COLUMNS, SKIP
from sts2_advisor.train import (
    apply_train_filter, compare_top_picks, evaluate, fit_categories, main,
    pick_rate_scores, predict, run, split_data, to_matrix,
)


def rows(offer_id, cands, picked, **ctx):
    base = dict({c: None for c in OFFER_COLUMNS}, offer_id=offer_id,
                run_id=offer_id.split(":")[0], source="codex", character="IRONCLAD",
                group=offer_id.split(":")[0])
    base.update(ctx)
    return [dict(base, candidate=c, is_skip=c == SKIP, picked=c == picked) for c in cands]


def test_evaluate_hand_checked():
    df = pd.DataFrame(rows("a:1", ["X", "Y", SKIP], "Y") + rows("b:1", ["X", "Z", SKIP], SKIP))
    # offer a: Y ranked first (hit). offer b: X first, SKIP second (miss, rank 2).
    m = evaluate(df, np.array([0.1, 0.9, 0.0, 0.9, 0.1, 0.5]))
    assert m["top1"] == 0.5 and m["mrr"] == pytest.approx(0.75)
    assert m["skip_precision"] == 0.0 and m["skip_recall"] == 0.0 and m["n_offers"] == 2

    m = evaluate(df, np.array([0.1, 0.9, 0.0, 0.0, 0.1, 0.5]))
    assert m["top1"] == 1.0 and m["skip_precision"] == 1.0 and m["skip_recall"] == 1.0
    assert np.isnan(m["win_if_followed"])  # no run outcomes in this data


def test_evaluate_win_rates_by_agreement():
    df = pd.DataFrame(rows("a:1", ["X", "Y"], "X", run_won=True)
                      + rows("b:1", ["X", "Y"], "X", run_won=False)
                      + rows("c:1", ["X", "Y"], "Y", run_won=False)
                      + rows("d:1", ["X", "Y"], "Y", run_won=None))  # abandoned: ignored
    m = evaluate(df, np.array([1, 0] * 4))  # always suggests X
    assert m["win_if_followed"] == 0.5 and m["win_if_not"] == 0.0


def test_apply_train_filter():
    df = pd.DataFrame(rows("a:1", ["X"], "X", run_won=True, ascension=2)
                      + rows("b:1", ["X"], "X", run_won=False, ascension=9)
                      + rows("c:1", ["X"], "X", run_won=None, ascension=9)
                      + rows("d:1", ["X"], "X", run_won=True, ascension=9))
    assert apply_train_filter(df, {})["offer_id"].tolist() == ["a:1", "b:1", "c:1", "d:1"]
    assert apply_train_filter(df, {"won_only": True})["offer_id"].tolist() == ["a:1", "d:1"]
    assert apply_train_filter(df, {"won_only": True, "min_ascension": 5}
                              )["offer_id"].tolist() == ["d:1"]


def test_relics_become_flag_columns():
    train = pd.DataFrame(rows("a:1", ["X"], "X", relics_held="ANCHOR|VAJRA")
                         + rows("b:1", ["X"], "X", relics_held="ANCHOR")
                         + rows("c:1", ["X"], "X", relics_held=None))
    cats = fit_categories(train)
    assert cats["relics"] == ["ANCHOR", "VAJRA"]
    test = pd.DataFrame(rows("d:1", ["X"], "X", relics_held="VAJRA|NEW_RELIC")
                        + rows("e:1", ["X"], "X", relics_held=""))
    X = to_matrix(test, cats)
    assert X[["relic_ANCHOR", "relic_VAJRA"]].values.tolist() == [[0, 1], [0, 0]]
    assert "relic_NEW_RELIC" not in X  # unseen in training: no column


def test_relic_columns_capped(monkeypatch):
    monkeypatch.setattr("sts2_advisor.train.MAX_RELIC_FEATURES", 2)
    held = ["A|B|C", "A|B", "A"]  # A x3, B x2, C x1
    train = pd.DataFrame(sum((rows(f"{i}:1", ["X"], "X", relics_held=h)
                              for i, h in enumerate(held)), []))
    assert fit_categories(train)["relics"] == ["A", "B"]


def test_compare_top_picks():
    df = pd.DataFrame(rows("a:1", ["X", "Y"], "X") + rows("b:1", ["X", "Y"], "X"))
    c = compare_top_picks(df, np.array([0, 1, 1, 0]), np.array([1, 0, 1, 0]))
    assert c["disagree"] == 0.5
    assert c["more_often"] == {"Y": 0.5} and c["less_often"] == {"X": -0.5}


def test_pick_rate_scores_with_character_fallback():
    train = pd.DataFrame(
        rows("a:1", ["X", "Y"], "X") + rows("b:1", ["X", "Y"], "Y", character="SILENT")
        + rows("c:1", ["X", "Y"], "Y", character="SILENT"))
    test = pd.DataFrame(rows("d:1", ["X", "Y", "NEW"], "X", character="SILENT")
                        + rows("e:1", ["X"], "X", character="DEFECT"))
    assert pick_rate_scores(train, test).tolist() == pytest.approx([1 / 3, 2 / 3, 0, 1 / 3])
    # per character: SILENT never picked X; DEFECT unseen -> global rate
    assert pick_rate_scores(train, test, ("character",)).tolist() == \
        pytest.approx([0.0, 1.0, 0.0, 1 / 3])


def synthetic_offers(n_runs=300, seed=0):
    """GOOD is always taken; with no GOOD, skip at low HP, else take the cheapest card."""
    rng = np.random.default_rng(seed)
    pool = ["GOOD", "A", "B", "C", "D", "E"]
    cost = {"GOOD": 1, "A": 0, "B": 1, "C": 2, "D": 2, "E": 3}
    out = []
    for r in range(n_runs):
        source = "mine" if r < 20 else "codex"
        won = bool(r % 2)
        for floor in range(1, 6):
            cands = list(rng.choice(pool, size=3, replace=False))
            hp = float(rng.random())
            if "GOOD" in cands:
                pick = "GOOD"
            elif hp < 0.3:
                pick = SKIP
            else:
                pick = min(cands, key=cost.get)
            for row in rows(f"r{r}:{floor}", cands + [SKIP], pick, source=source,
                            hp_frac=hp, floor=floor, act=1, ascension=5, n_offered=3,
                            run_won=won):
                row["cand_cost"] = cost.get(row["candidate"])
                out.append(row)
    return pd.DataFrame(out, columns=OFFER_COLUMNS)


def config(tmp_path, mlflow=False):
    return {
        "data": {"offers": str(tmp_path / "offers.parquet")},
        "split": {"test_frac": 0.2, "valid_frac": 0.2, "seed": 0, "hold_out_mine": True},
        "model": {"objective": "lambdarank", "learning_rate": 0.1, "num_leaves": 15,
                  "min_data_in_leaf": 5, "num_boost_round": 200,
                  "early_stopping_rounds": 20},
        "output": {"model_dir": str(tmp_path / "model")},
        "mlflow": {"enabled": mlflow, "db_path": str(tmp_path / "mlruns" / "mlflow.db"),
                   "artifact_dir": str(tmp_path / "mlruns" / "artifacts"),
                   "experiment": "test"},
    }


def test_run_learns_planted_pattern(tmp_path):
    synthetic_offers().to_parquet(tmp_path / "offers.parquet")
    metrics = run(config(tmp_path))

    assert metrics["n_offers"]["mine"] == 100  # 20 runs x 5 rewards, never trained on
    test = metrics["test"]
    assert test["model"]["top1"] > 0.9
    assert test["model"]["top1"] > test["pick_rate"]["top1"] + 0.1
    assert test["random"]["top1"] < 0.4
    assert metrics["mine"]["model"]["top1"] > 0.9
    for f in ("model.txt", "categories.json", "metrics.json"):
        assert (tmp_path / "model" / f).exists()


def test_train_filter_leaves_test_sets_unchanged(tmp_path):
    df = synthetic_offers(n_runs=100)
    cfg = config(tmp_path)
    base = split_data(df, cfg)
    filtered = split_data(df, dict(cfg, train_filter={"won_only": True}))
    assert filtered["train"]["run_won"].all()
    assert len(filtered["train"]) < len(base["train"])
    for split in ("test", "mine"):
        assert filtered[split]["offer_id"].tolist() == base[split]["offer_id"].tolist()


def test_run_compares_with_saved_model(tmp_path):
    synthetic_offers().to_parquet(tmp_path / "offers.parquet")
    run(config(tmp_path))  # saves the baseline model to tmp_path/model
    cfg = config(tmp_path)
    cfg["train_filter"] = {"won_only": True}
    cfg["output"]["model_dir"] = str(tmp_path / "winners")
    metrics = run(cfg, compare_with=tmp_path / "model")

    assert metrics["n_offers"]["train"] < metrics["n_offers"]["test"] * 4
    assert set(metrics) >= {"test", "test_won", "mine", "comparison"}
    assert 0.0 <= metrics["comparison"]["disagree"] <= 1.0
    assert metrics["test_won"]["model"]["n_offers"] < metrics["test"]["model"]["n_offers"]


def test_predict_works_for_model_with_fewer_features():
    """A model trained before a feature column existed still scores new data."""
    df = synthetic_offers(n_runs=60)
    cats = fit_categories(df)
    X = to_matrix(df, cats).drop(columns=["cand_synergy"])
    older = lgb.train({"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 5},
                      lgb.Dataset(X, label=df["picked"].astype(int),
                                  group=df.groupby("offer_id", sort=False).size().to_numpy()),
                      num_boost_round=5)
    assert "cand_synergy" not in older.feature_name()
    assert len(predict(older, cats, df)) == len(df)


def test_main_logs_to_mlflow(tmp_path, capsys):
    synthetic_offers(n_runs=80).to_parquet(tmp_path / "offers.parquet")
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(json.dumps(config(tmp_path, mlflow=True)))  # JSON is valid YAML
    main(["--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert "Community test set:" in out and "Your runs (held out):" in out
    assert (tmp_path / "mlruns" / "mlflow.db").exists()
    assert any((tmp_path / "mlruns" / "artifacts").rglob("model.txt"))
