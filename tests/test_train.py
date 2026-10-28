import json

import numpy as np
import pandas as pd
import pytest

from sts2_advisor.features import OFFER_COLUMNS, SKIP
from sts2_advisor.train import evaluate, main, pick_rate_scores, run


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
                            hp_frac=hp, floor=floor, act=1, ascension=5, n_offered=3):
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


def test_main_logs_to_mlflow(tmp_path, capsys):
    synthetic_offers(n_runs=80).to_parquet(tmp_path / "offers.parquet")
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(json.dumps(config(tmp_path, mlflow=True)))  # JSON is valid YAML
    main(["--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert "Community test set:" in out and "Your runs (held out):" in out
    assert (tmp_path / "mlruns" / "mlflow.db").exists()
    assert any((tmp_path / "mlruns" / "artifacts").rglob("model.txt"))
