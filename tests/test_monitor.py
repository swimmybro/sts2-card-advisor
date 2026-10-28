import gzip
import json

import pandas as pd
import pytest

from make_demo_models import main as make_demo_models
from sts2_advisor.advise import Advisor
from sts2_advisor.monitor import _calibration_gap, main, run_checks
from test_features import card, make_run, offer, point

ALLOWED = {"STRIKE", "DEFEND", "ANGER", "INFLAME", "HEADBUTT"}
THRESHOLDS = {"min_runs": 5, "min_rewards": 5, "new_card_run_share": 0.2,
              "pick_rate_shift": 0.3, "min_card_offers": 5, "shifted_cards": 1,
              "skip_rate_shift": 0.5, "top1_drop": 0.03, "calibration_gap": 0.05,
              "auc_drop": 0.05}


@pytest.fixture(scope="module")
def advisor(tmp_path_factory):
    models = tmp_path_factory.mktemp("models")
    make_demo_models(models)
    return Advisor(models / "card_pick", None, models / "win_prob")


def reference(tmp_path, anger_pick_rate=0.1, n=100, build="v1", source="codex"):
    """Reference offers: ANGER offered n times and picked at the given rate."""
    rows = []
    for i in range(n):
        picked = i < anger_pick_rate * n
        rows += [{"run_id": f"ref{i}", "candidate": "ANGER", "picked": picked,
                  "is_skip": False, "source": source, "build_id": build},
                 {"run_id": f"ref{i}", "candidate": "SKIP", "picked": not picked,
                  "is_skip": True, "source": source, "build_id": build}]
    if source != "codex":  # the models' training versions always come from codex rows
        rows.append({"run_id": "c", "candidate": "SKIP", "picked": True, "is_skip": True,
                     "source": "codex", "build_id": build})
    path = tmp_path / "offers.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return {"reference": {"offers": str(path)}, "thresholds": dict(THRESHOLDS)}


def recent_run(i, extra_card=None, build="v1", win=False):
    """One reward where the player takes ANGER; optionally a card in the final deck."""
    deck = ["STRIKE", "DEFEND", "ANGER"] + ([extra_card] if extra_card else [])
    gained = [card("ANGER")] + ([card(extra_card)] if extra_card else [])
    history = [[point("monster", card_choices=offer("ANGER", "INFLAME", picked="ANGER"),
                       cards_gained=gained)]]
    return dict(make_run(history, deck, win=win, seed=f"s{i}", start_time=1000 + i),
                build_id=build, run_hash=f"h{i}")


def write_export(path, runs):
    with gzip.open(path, "wt") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    return path


def checks(result):
    return {c["name"]: c for c in result["checks"]}


def test_no_drift(tmp_path, advisor):
    cfg = reference(tmp_path, anger_pick_rate=0.9)
    runs = write_export(tmp_path / "r.jsonl.gz", [recent_run(i) for i in range(10)])
    c = checks(run_checks([runs], cfg, advisor, ALLOWED))
    assert c["New cards"]["status"] == "ok"
    assert c["Game version"]["status"] == "ok"
    assert c["Pick rates"]["status"] == "ok"


def test_new_card_and_version_and_pick_rate_drift(tmp_path, advisor):
    cfg = reference(tmp_path, anger_pick_rate=0.1)
    runs = [recent_run(i, build="v2") for i in range(10)]
    runs += [recent_run(10 + i, extra_card="NEW_CARD", build="v2") for i in range(5)]
    runs.append(recent_run(99, extra_card="SOMEMOD-ZAP", build="v2"))  # one run: a mod
    result = run_checks([write_export(tmp_path / "r.jsonl.gz", runs)], cfg, advisor, ALLOWED)
    c = checks(result)
    assert c["New cards"]["status"] == "drift"
    assert list(c["New cards"]["details"]["likely_new_cards"]) == ["NEW_CARD"]
    assert c["Game version"]["status"] == "drift"
    assert c["Game version"]["details"]["new_builds"] == {"v2": 16}
    assert c["Pick rates"]["status"] == "drift"  # ANGER: 10% -> 100%
    assert "ANGER" in c["Pick rates"]["details"]["shifted"]
    assert result["drift"] and result["verdict"].startswith("Retrain recommended")


def test_too_little_data(tmp_path, advisor):
    cfg = reference(tmp_path)
    runs = write_export(tmp_path / "r.jsonl.gz", [recent_run(i) for i in range(2)])
    result = run_checks([runs], cfg, advisor, ALLOWED)
    assert checks(result)["Recent runs"]["status"] == "low data"
    assert result["verdict"] == "Too little recent data to judge drift."


def test_your_runs_compared_with_your_earlier_runs(tmp_path, advisor):
    cfg = reference(tmp_path, anger_pick_rate=0.1, source="mine")
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(10):
        (raw / f"{1000 + i}.run").write_text(json.dumps(recent_run(i)))
    result = run_checks([raw], cfg, advisor, ALLOWED)
    assert result["population"] == "mine"
    c = checks(result)
    # a change in your own picks is reported, but isn't drift the models can fix
    assert c["Pick rates (yours)"]["status"] == "info"
    assert "your picks changed" in c["Pick rates (yours)"]["message"]


def test_since_filters_by_start_time(tmp_path, advisor):
    cfg = reference(tmp_path)
    runs = write_export(tmp_path / "r.jsonl.gz", [recent_run(i) for i in range(10)])
    result = run_checks([runs], cfg, advisor, ALLOWED, since=1005)
    assert checks(result)["Recent runs"]["details"]["n_runs"] == 5


def test_calibration_gap():
    bins = [{"n": 3, "predicted": 0.5, "actual": 0.2}, {"n": 1, "predicted": 0.9, "actual": 0.9}]
    assert _calibration_gap(bins) == pytest.approx(0.9 / 4)
    assert _calibration_gap(None) is None


def test_cli_json_and_fail_on_drift(tmp_path, advisor, monkeypatch, capsys):
    cfg = reference(tmp_path, anger_pick_rate=0.1)
    cfg_path = tmp_path / "monitor.yaml"
    cfg_path.write_text(json.dumps(cfg))
    allow = tmp_path / "cards.txt"
    allow.write_text("\n".join(ALLOWED))
    runs = write_export(tmp_path / "r.jsonl.gz", [recent_run(i, build="v2") for i in range(10)])
    monkeypatch.setenv("MODEL_DIR", str(advisor.model_dir))
    monkeypatch.setenv("WIN_MODEL_DIR", str(advisor.win_model_dir))
    args = [str(runs), "--config", str(cfg_path), "--card-allowlist", str(allow),
            "--json", str(tmp_path / "out.json")]

    main(args)  # drift, but no --fail-on-drift: returns normally
    assert "Retrain recommended" in capsys.readouterr().out
    assert json.loads((tmp_path / "out.json").read_text())["drift"] is True
    with pytest.raises(SystemExit) as e:
        main(args + ["--fail-on-drift"])
    assert e.value.code == 1
