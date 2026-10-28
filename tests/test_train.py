import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from sts2_advisor.features import OFFER_COLUMNS, SKIP, decks_path
from sts2_advisor.train import (
    DeckStore, apply_train_filter, compare_top_picks, evaluate, fit_categories, load_offers,
    main, pick_rate_scores, predict, run, split_data, to_matrix,
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


def test_deck_card_columns():
    train = pd.DataFrame(rows("a:1", ["X", SKIP], "X", deck_cards="A|A|B")
                         + rows("b:1", ["X", SKIP], "X", deck_cards="A|C")
                         + rows("c:1", ["X", SKIP], "X", deck_cards=""))
    assert "deck_cards" not in fit_categories(train)  # off unless asked for
    cats = fit_categories(train, n_deck_cards=2)
    assert cats["deck_cards"] == ["A", "B"]  # A in 4 rows, B and C in 2; ties by name
    X = to_matrix(train, cats)
    assert X[["deck_has_A", "deck_has_B"]].values.tolist() == [[2, 1], [2, 1], [1, 0],
                                                                [1, 0], [0, 0], [0, 0]]
    # dictionary-encoded input (as load_offers(with_decks=True) gives) works the same
    as_cat = train.assign(deck_cards=train["deck_cards"].astype("category"))
    assert to_matrix(as_cat.iloc[2:4], cats)[["deck_has_A"]].values.tolist() == [[1], [1]]


def test_deck_store():
    store = DeckStore(["A|A|B", "", "C|B"])
    assert store.counts(np.array([0, 1, 2, 0]), ["A", "B"]).tolist() == [
        [2, 1], [0, 0], [0, 1], [2, 1]]
    assert store.frequency(np.array([0, 2])).to_dict() == {"A": 2, "B": 2, "C": 1}
    assert store.strings([2, 1]) == ["C|B", ""]


def test_load_offers_with_deck_side_file(tmp_path):
    df = synthetic_offers(n_runs=40)
    df.to_parquet(tmp_path / "offers.parquet")
    offer_ids = df.loc[df["is_skip"], "offer_id"]
    decks = pd.DataFrame({"offer_id": offer_ids,
                          "deck_cards": ["A|B" if i % 2 else "B" for i in range(len(offer_ids))]})
    decks.to_parquet(decks_path(tmp_path / "offers.parquet"))

    assert "decks" not in load_offers(tmp_path / "offers.parquet").attrs
    loaded = load_offers(tmp_path / "offers.parquet", with_decks=True)
    parts = split_data(loaded, config(tmp_path))
    for name in ("train", "test", "mine"):
        assert parts[name].attrs["decks"] is loaded.attrs["decks"]  # survives splitting

    cats = fit_categories(parts["train"], n_deck_cards=2)
    assert cats["deck_cards"] == ["A", "B"]
    from_store = to_matrix(parts["test"], cats)[["deck_has_A", "deck_has_B"]]
    # the same rows with deck strings attached give the same columns
    strings = dict(zip(decks["offer_id"], decks["deck_cards"]))
    original_ids = pd.Series(sorted(offer_ids)).to_numpy()[parts["test"]["offer_id"]]
    as_text = parts["test"].assign(deck_cards=[strings[o] for o in original_ids])
    assert (to_matrix(as_text, cats)[from_store.columns].values == from_store.values).all()


def test_deck_columns_need_decks(tmp_path):
    rows_without = pd.DataFrame(rows("a:1", ["X", SKIP], "X"))
    with pytest.raises(ValueError, match="uses deck cards"):
        to_matrix(rows_without, {**fit_categories(rows_without), "deck_cards": ["A"]})


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
