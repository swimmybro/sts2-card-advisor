import json
import os

import numpy as np
import pytest
import yaml

from sts2_advisor import pipeline
from sts2_advisor.pipeline import decide, promote, run_pipeline
from test_features import card, make_run, offer, point
from test_train import config as pick_config
from test_win import win_config

POOL = ["GOOD", "A", "B", "C", "D"]
TOL = {"pick_top1": 0.002, "win_auc": 0.003}


def test_decide():
    cur = {"pick_top1": 0.50, "win_auc": 0.77}
    assert decide(None, cur, TOL)["promote"] is True
    assert decide(cur, {"pick_top1": 0.499, "win_auc": 0.769}, TOL)["promote"] is True  # within tol
    worse = decide(cur, {"pick_top1": 0.48, "win_auc": 0.78}, TOL)
    assert worse["promote"] is False and "pick top-1" in worse["reason"]
    worse = decide(cur, {"pick_top1": 0.52, "win_auc": 0.70}, TOL)
    assert worse["promote"] is False and "win AUC" in worse["reason"]
    assert decide(cur, {"pick_top1": 0.1, "win_auc": 0.1}, TOL, force=True)["promote"] is True


def _dies_abruptly():
    os._exit(9)  # like being killed for using too much memory


def test_killed_step_gives_clear_error():
    assert pipeline._in_fresh_process("adding", sum, [1, 2]) == 3
    with pytest.raises(RuntimeError, match="crashing was killed.*out of memory"):
        pipeline._in_fresh_process("crashing", _dies_abruptly)


def test_promote_archives_current_pair(tmp_path):
    current, staging, archive = tmp_path / "models", tmp_path / "staging", tmp_path / "archive"
    for d, tag in ((current, "old"), (staging, "new")):
        for name in ("card_pick", "win_prob"):
            (d / name).mkdir(parents=True)
            (d / name / "model.txt").write_text(tag)
    (staging / "win_prob" / "reliability.json").write_text(
        json.dumps({"pick_model_dir": str(staging / "card_pick")}))

    archived = promote(staging, current, archive)
    assert (current / "card_pick" / "model.txt").read_text() == "new"
    assert (archived / "card_pick" / "model.txt").read_text() == "old"
    assert (archived / "win_prob" / "model.txt").read_text() == "old"
    assert not (staging / "card_pick").exists()
    rel = json.loads((current / "win_prob" / "reliability.json").read_text())
    assert rel["pick_model_dir"] == str(current / "card_pick")


def test_promote_without_current_models(tmp_path):
    staging = tmp_path / "staging"
    for name in ("card_pick", "win_prob"):
        (staging / name).mkdir(parents=True)
    assert promote(staging, tmp_path / "models", tmp_path / "archive") is None
    assert (tmp_path / "models" / "win_prob").is_dir()


def synthetic_raw_runs(raw, n_runs=150, seed=0):
    """Game-run files: players take GOOD when offered, and taking it wins the run."""
    rng = np.random.default_rng(seed)
    raw.mkdir(parents=True)
    for r in range(n_runs):
        history, gained = [], []
        for _ in range(6):
            cands = list(rng.choice(POOL, size=3, replace=False))
            pick = "GOOD" if "GOOD" in cands else (cands[0] if rng.random() < 0.6 else None)
            history.append(point("monster", card_choices=offer(*cands, picked=pick),
                                 cards_gained=[card(pick)] if pick else []))
            gained += [pick] if pick else []
        run = make_run([history], ["STRIKE"] * 5 + ["DEFEND"] * 4 + gained,
                       win="GOOD" in gained, seed=f"s{r}", start_time=1000 + r)
        (raw / f"{1000 + r}.run").write_text(json.dumps(run))


@pytest.fixture
def setup(tmp_path):
    synthetic_raw_runs(tmp_path / "raw")
    pick, win = pick_config(tmp_path), win_config(tmp_path)
    pick["split"]["hold_out_mine"] = win["split"]["hold_out_mine"] = False  # all runs are .run files
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(pick))
    (tmp_path / "win.yaml").write_text(yaml.safe_dump(win))
    (tmp_path / "cards.csv").write_text("id,type,rarity,color,cost,is_x_cost,traits\n" + "".join(
        f"{c},Attack,Common,ironclad,1,False,\n" for c in POOL + ["STRIKE", "DEFEND"]))
    (tmp_path / "allow.txt").write_text("\n".join(POOL + ["STRIKE", "DEFEND"]))
    cfg = {"inputs": [str(tmp_path / "raw")], "offers": str(tmp_path / "offers.parquet"),
           "train_config": str(tmp_path / "train.yaml"), "win_config": str(tmp_path / "win.yaml"),
           "models": {"current": str(tmp_path / "models"), "staging": str(tmp_path / "staging"),
                      "archive": str(tmp_path / "archive")},
           "tolerance": TOL}
    kw = {"card_meta": tmp_path / "cards.csv", "card_allowlist": tmp_path / "allow.txt"}
    return tmp_path, cfg, kw


def test_pipeline_end_to_end(setup):
    tmp, cfg, kw = setup
    first = run_pipeline(cfg, **kw)
    assert first["steps"]["features"]["runs"] == 150
    assert first["current"] is None and first["decision"]["promote"] is True
    assert first["decision"]["archived_to"] is None
    assert (tmp / "models" / "card_pick" / "model.txt").exists()
    assert (tmp / "models" / "win_prob" / "reliability.json").exists()
    assert first["candidate"]["pick_top1"] > 0.5  # GOOD is learnable

    second = run_pipeline(cfg, skip_features=True, **kw)
    assert "features" not in second["steps"]
    assert second["current"] is not None
    # same data and seed: the new pair matches the current one, so it's promoted
    assert second["candidate"]["pick_top1"] == pytest.approx(second["current"]["pick_top1"])
    assert second["decision"]["promote"] is True
    assert len(list((tmp / "archive").iterdir())) == 1


def test_pipeline_keeps_better_current_models(setup, monkeypatch):
    tmp, cfg, kw = setup
    run_pipeline(cfg, **kw)
    # Pretend the current models score far better than anything new can.
    real_score = pipeline.score_pair

    def score(parts, pick_dir, win_dir, meta=None):
        s = real_score(parts, pick_dir, win_dir, meta)
        if s and "staging" not in str(pick_dir):
            s = dict(s, pick_top1=1.0)
        return s

    monkeypatch.setattr(pipeline, "score_pair", score)
    result = run_pipeline(cfg, skip_features=True, **kw)
    assert result["decision"]["promote"] is False
    assert (tmp / "staging" / "card_pick" / "model.txt").exists()  # left for inspection
    assert not (tmp / "archive").exists()

    forced = run_pipeline(cfg, skip_features=True, force=True, **kw)
    assert forced["decision"]["reason"] == "forced"
    assert len(list((tmp / "archive").iterdir())) == 1
