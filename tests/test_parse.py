import json

import pytest

from sts2_advisor.parse import (
    RunParseError, load_run, main, parse_dir, parse_run, summarize,
)


def _offer(*cards, picked=None):
    return [{"card": {"id": f"CARD.{c}"}, "was_picked": c == picked} for c in cards]


def _point(kind, stats):
    return {"map_point_type": kind, "player_stats": stats, "rooms": []}


def make_run(win=False, players=None, history=None):
    players = players or [{
        "id": 1, "character": "CHARACTER.IRONCLAD",
        "deck": [{"id": "CARD.STRIKE_IRONCLAD"}, {"id": "CARD.BASH"}],
        "relics": [{"id": "RELIC.BURNING_BLOOD"}],
    }]
    history = history or [
        [
            _point("ancient", [{"player_id": 1}]),
            _point("monster", [{"player_id": 1,
                                "card_choices": _offer("A", "B", "C", picked="B")}]),
        ],
        [
            _point("monster", [{"player_id": 1,
                                "card_choices": _offer("D", "E", "F")}]),
        ],
    ]
    return {
        "acts": ["ACT.OVERGROWTH", "ACT.HIVE"], "ascension": 5,
        "build_id": "v0.111.0", "game_mode": "standard",
        "killed_by_encounter": "ENCOUNTER.QUEEN_BOSS",
        "killed_by_event": "NONE.NONE", "map_point_history": history,
        "players": players, "run_time": 100, "schema_version": 10,
        "seed": "ABC", "start_time": 1, "was_abandoned": False, "win": win,
    }


def test_parse_run_rows():
    runs, choices = parse_run(make_run(), "r1")
    assert len(runs) == 1
    r = runs[0]
    assert r["character"] == "IRONCLAD"
    assert r["floors_reached"] == 3 and r["acts_reached"] == 2
    assert r["deck"] == "STRIKE_IRONCLAD|BASH" and r["deck_size"] == 2
    assert r["killed_by_encounter"] == "QUEEN_BOSS"
    assert r["win"] is False


def test_parse_run_card_choices():
    _, choices = parse_run(make_run(win=True), "r1")
    assert len(choices) == 6
    first = [c for c in choices if c["floor"] == 2]
    assert [c["card_id"] for c in first if c["was_picked"]] == ["B"]
    assert not any(c["offer_skipped"] for c in first)
    second = [c for c in choices if c["floor"] == 3]
    assert all(c["act"] == 2 and c["offer_skipped"] for c in second)
    assert all(c["win"] and c["n_offered"] == 3 for c in choices)


def test_multiplayer_choices_attributed_per_player():
    players = [
        {"id": 10, "character": "CHARACTER.SILENT", "deck": [], "relics": []},
        {"id": 20, "character": "CHARACTER.DEFECT", "deck": [], "relics": []},
    ]
    history = [[_point("monster", [
        {"player_id": 10, "card_choices": _offer("A", "B", picked="A")},
        {"player_id": 20, "card_choices": _offer("C", "D", picked="D")},
    ])]]
    runs, choices = parse_run(make_run(players=players, history=history), "mp")
    assert {r["n_players"] for r in runs} == {2}
    picks = {c["character"]: c["card_id"] for c in choices if c["was_picked"]}
    assert picks == {"SILENT": "A", "DEFECT": "D"}


def test_load_run_rejects_bad_files(tmp_path):
    bad_json = tmp_path / "a.run"
    bad_json.write_text("{not json")
    with pytest.raises(RunParseError):
        load_run(bad_json)
    missing = tmp_path / "b.run"
    missing.write_text(json.dumps({"players": []}))
    with pytest.raises(RunParseError):
        load_run(missing)


def test_parse_dir_skips_bad_and_summarizes(tmp_path):
    (tmp_path / "1.run").write_text(json.dumps(make_run(win=True)))
    (tmp_path / "2.run").write_text(json.dumps(make_run()))
    (tmp_path / "3.run").write_text("garbage")
    (tmp_path / "notes.txt").write_text("ignored")
    runs, choices, errors, n_mp = parse_dir(tmp_path)
    assert list(runs["run_id"]) == ["1", "2"]
    assert len(choices) == 12
    assert len(errors) == 1 and errors[0].startswith("3.run")
    assert n_mp == 0
    text = summarize(runs, choices, errors, n_mp)
    assert "Files parsed:      2  (skipped: 1, multiplayer excluded: 0)" in text
    assert "1/2 (50.0%)" in text


def test_parse_dir_excludes_multiplayer_by_default(tmp_path):
    players = [
        {"id": 10, "character": "CHARACTER.SILENT", "deck": [], "relics": []},
        {"id": 20, "character": "CHARACTER.DEFECT", "deck": [], "relics": []},
    ]
    (tmp_path / "solo.run").write_text(json.dumps(make_run()))
    (tmp_path / "coop.run").write_text(json.dumps(make_run(players=players)))

    runs, choices, errors, n_mp = parse_dir(tmp_path)
    assert list(runs["run_id"]) == ["solo"] and n_mp == 1
    assert set(choices["run_id"]) == {"solo"}

    runs, _, _, n_mp = parse_dir(tmp_path, include_multiplayer=True)
    assert set(runs["run_id"]) == {"solo", "coop"} and n_mp == 0


def test_main_writes_csvs(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "1.run").write_text(json.dumps(make_run()))
    out = tmp_path / "out"
    main([str(raw), "--out", str(out)])
    assert (out / "runs.csv").exists() and (out / "card_choices.csv").exists()
    assert "Files parsed" in capsys.readouterr().out
