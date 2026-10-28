import gzip
import itertools
import json
from pathlib import Path

import pandas as pd
import pytest

from sts2_advisor.parse import (
    RunParseError, load_card_allowlist, load_run, main, parse_dir, parse_run,
    summarize,
)

_start_times = itertools.count(1)


def _offer(*cards, picked=None):
    return [{"card": {"id": f"CARD.{c}"}, "was_picked": c == picked} for c in cards]


def _point(kind, stats):
    return {"map_point_type": kind, "player_stats": stats, "rooms": []}


MP_PLAYERS = [
    {"id": 10, "character": "CHARACTER.SILENT", "deck": [], "relics": []},
    {"id": 20, "character": "CHARACTER.DEFECT", "deck": [], "relics": []},
]


def make_run(win=False, players=None, history=None, seed="ABC", start_time=None):
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
        "seed": seed,
        "start_time": next(_start_times) if start_time is None else start_time,
        "was_abandoned": False, "win": win,
    }


def write_export(path, runs):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")


def test_parse_run_rows():
    runs, choices = parse_run(make_run(), "r1")
    assert len(runs) == 1
    r = runs[0]
    assert r["character"] == "IRONCLAD" and r["source"] == "mine"
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
    history = [[_point("monster", [
        {"player_id": 10, "card_choices": _offer("A", "B", picked="A")},
        {"player_id": 20, "card_choices": _offer("C", "D", picked="D")},
    ])]]
    runs, choices = parse_run(make_run(players=MP_PLAYERS, history=history), "mp")
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
    runs, choices, stats = parse_dir(tmp_path)
    assert list(runs["run_id"]) == ["1", "2"]
    assert len(choices) == 12
    assert stats.n_errors == 1 and stats.errors[0].startswith("3.run")
    text = summarize(runs, choices, stats)
    assert "Runs parsed:       2  (mine: 2)" in text
    assert "1 unreadable, 0 multiplayer, 0 duplicates, 0 modded" in text
    assert "1/2 (50.0%)" in text


def test_parse_dir_excludes_multiplayer_by_default(tmp_path):
    (tmp_path / "solo.run").write_text(json.dumps(make_run()))
    (tmp_path / "coop.run").write_text(json.dumps(make_run(players=MP_PLAYERS)))

    runs, choices, stats = parse_dir(tmp_path)
    assert list(runs["run_id"]) == ["solo"] and stats.n_multiplayer == 1
    assert set(choices["run_id"]) == {"solo"}

    runs, _, stats = parse_dir(tmp_path, include_multiplayer=True)
    assert set(runs["run_id"]) == {"solo", "coop"} and stats.n_multiplayer == 0


def test_parse_jsonl_gz_export(tmp_path):
    a = dict(make_run(win=True), run_hash="h1", player_token="tok")
    b = dict(make_run(), run_hash="h2", player_token=None)
    write_export(tmp_path / "runs-v0.111.0.jsonl.gz", [a, b])
    with gzip.open(tmp_path / "runs-v0.111.0.jsonl.gz", "at") as f:
        f.write("not json\n")

    runs, choices, stats = parse_dir(tmp_path)
    assert list(runs["run_id"]) == ["h1", "h2"]
    assert set(runs["source"]) == {"codex"} and set(choices["source"]) == {"codex"}
    assert runs["player_token"].iloc[0] == "tok" and pd.isna(runs["player_token"].iloc[1])
    assert stats.n_errors == 1 and ":3" in stats.errors[0]


def test_duplicates_keep_first_source(tmp_path):
    raw, ext = tmp_path / "raw", tmp_path / "ext"
    raw.mkdir()
    ext.mkdir()
    mine = make_run(seed="S", start_time=42)
    (raw / "42.run").write_text(json.dumps(mine))
    write_export(ext / "runs.jsonl.gz", [dict(mine, run_hash="dup"),
                                         dict(make_run(), run_hash="new")])

    runs, _, stats = parse_dir([raw, ext])
    assert runs.set_index("run_id")["source"].to_dict() == {"42": "mine", "new": "codex"}
    assert stats.n_duplicates == 1


VANILLA_CARDS = {"A", "B", "C", "D", "E", "F", "STRIKE_IRONCLAD", "BASH"}


def test_modded_runs_dropped_with_allowlist(tmp_path):
    (tmp_path / "ok.run").write_text(json.dumps(make_run()))
    modded_char = make_run(players=[{"id": 1, "character": "CHARACTER.HERMIT-HERMIT",
                                     "deck": [], "relics": []}])
    (tmp_path / "char.run").write_text(json.dumps(modded_char))
    modded_card = make_run(history=[[_point("monster", [
        {"player_id": 1, "card_choices": _offer("A", "MOD-ZAP", picked="A")}])]])
    (tmp_path / "card.run").write_text(json.dumps(modded_card))
    modded_deck = make_run()
    modded_deck["players"][0]["deck"].append({"id": "CARD.YUI_CARD_X"})
    (tmp_path / "deck.run").write_text(json.dumps(modded_deck))

    runs, _, stats = parse_dir(tmp_path, allowed_cards=VANILLA_CARDS)
    assert list(runs["run_id"]) == ["ok"] and stats.n_modded == 3

    runs, _, stats = parse_dir(tmp_path)
    assert len(runs) == 4 and stats.n_modded == 0


def test_load_card_allowlist(tmp_path):
    p = tmp_path / "cards.txt"
    p.write_text("# header\nBASH\n\nSTRIKE_IRONCLAD  # starter\n")
    assert load_card_allowlist(p) == {"BASH", "STRIKE_IRONCLAD"}


def test_repo_allowlist_covers_starters():
    cards = load_card_allowlist(Path(__file__).parents[1] / "configs/official_cards.txt")
    assert {"STRIKE_IRONCLAD", "BASH", "SIDESTEP", "FOLLOW_THROUGH"} <= cards
    assert not any("-" in c for c in cards)


def test_main_writes_parquet(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "1.run").write_text(json.dumps(make_run()))
    write_export(raw / "x.jsonl.gz", [dict(make_run(win=True), run_hash="h")])
    allowlist = tmp_path / "cards.txt"
    allowlist.write_text("\n".join(VANILLA_CARDS))
    out = tmp_path / "out"
    main([str(raw), "--out", str(out), "--card-allowlist", str(allowlist)])
    runs = pd.read_parquet(out / "runs.parquet")
    choices = pd.read_parquet(out / "card_choices.parquet")
    assert sorted(runs["source"]) == ["codex", "mine"]
    assert len(choices) == 12
    assert "Runs parsed:       2  (codex: 1, mine: 1)" in capsys.readouterr().out
