import json

import pytest

from make_demo_models import main as make_demo_models
from sts2_advisor.advise import Advisor
from sts2_advisor.watch import find_history_dir, format_review, main, review_run, sync
from test_features import STANDARD_FINAL, STANDARD_HISTORY, make_run


@pytest.fixture(scope="module")
def advisor(tmp_path_factory):
    models = tmp_path_factory.mktemp("models")
    make_demo_models(models)
    return Advisor(models / "card_pick", None, models / "win_prob")


def write_run(path, **kw):
    path.write_text(json.dumps(make_run(STANDARD_HISTORY, STANDARD_FINAL, **kw)))
    return path


def test_sync_copies_new_runs_only(tmp_path):
    history, raw = tmp_path / "history", tmp_path / "raw"
    history.mkdir()
    write_run(history / "100.run")
    write_run(history / "200.run")
    (history / "100.run.backup").write_text("{}")
    (history / "300.run").write_text('{"players": [')  # still being written

    assert [p.name for p in sync(history, raw)] == ["100.run", "200.run"]
    assert sorted(p.name for p in raw.iterdir()) == ["100.run", "200.run"]
    assert sync(history, raw) == []  # nothing new the second time

    write_run(history / "300.run")  # finished writing
    assert [p.name for p in sync(history, raw)] == ["300.run"]


def test_sync_leaves_existing_files_alone(tmp_path):
    history, raw = tmp_path / "history", tmp_path / "raw"
    history.mkdir()
    raw.mkdir()
    write_run(history / "100.run")
    (raw / "100.run").write_text("mine")
    assert sync(history, raw) == []
    assert (raw / "100.run").read_text() == "mine"


def test_find_history_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("STS2_HISTORY_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        find_history_dir()
    found = (tmp_path / "Library/Application Support/SlayTheSpire2/steam/123"
             / "profile1/saves/history")
    found.mkdir(parents=True)
    assert find_history_dir() == found
    monkeypatch.setenv("STS2_HISTORY_DIR", str(tmp_path / "elsewhere"))
    assert find_history_dir() == tmp_path / "elsewhere"
    assert find_history_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_review_run(tmp_path, advisor):
    review = review_run(write_run(tmp_path / "42.run", killed_by_encounter="ENCOUNTER.QUEEN_BOSS"),
                        advisor)
    assert (review.run_id, review.character, review.outcome) == ("42", "IRONCLAD", "lost to QUEEN_BOSS")
    r = review.rewards
    assert r["floor"].tolist() == [1, 4]  # the shop on floor 2 isn't a card reward
    assert r["yours"].tolist() == ["ANGER", "SKIP"]
    assert r["offered"].iloc[1] == ["INFLAME+", "HEADBUTT"]
    assert {"pick_top", "win_top", "your_win", "best_win", "gap", "tied"} <= set(r)
    assert (r["gap"] >= 0).all()
    assert ((r["gap"] == 0) == r["agrees_win"]).all()

    text = format_review(review)
    assert text.startswith("Run 42: IRONCLAD A3, lost to QUEEN_BOSS after 4 floors")
    assert "you: ANGER" in text and "best:" in text and "popular:" in text
    assert "Took the best option (ties count) on" in text
    assert text.index("Took the best option") < text.index("Took the popular option")


def test_review_coop_and_abandoned(tmp_path, advisor):
    run = make_run(STANDARD_HISTORY, STANDARD_FINAL, abandoned=True)
    run["players"].append(dict(run["players"][0], id=2))
    (tmp_path / "7.run").write_text(json.dumps(run))
    review = review_run(tmp_path / "7.run", advisor)
    assert review.outcome == "abandoned" and "co-op" in review.note
    assert "co-op" in format_review(review)


def test_review_latest_from_cli(tmp_path, advisor, monkeypatch, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_run(raw / "100.run")
    write_run(raw / "200.run", win=True)
    monkeypatch.setenv("MODEL_DIR", str(advisor.model_dir))
    monkeypatch.setenv("WIN_MODEL_DIR", str(advisor.win_model_dir))
    main(["--review", "latest", "--raw-dir", str(raw)])
    assert capsys.readouterr().out.startswith("Run 200: IRONCLAD A3, won")


def test_once_syncs_and_reviews(tmp_path, advisor, monkeypatch, capsys):
    history, raw = tmp_path / "history", tmp_path / "raw"
    history.mkdir()
    write_run(history / "100.run")
    monkeypatch.setenv("MODEL_DIR", str(advisor.model_dir))
    monkeypatch.setenv("WIN_MODEL_DIR", str(advisor.win_model_dir))
    main(["--once", "--history", str(history), "--raw-dir", str(raw)])
    out = capsys.readouterr().out
    assert "New run copied: 100.run" in out and "Run 100: IRONCLAD" in out
    assert (raw / "100.run").exists()
