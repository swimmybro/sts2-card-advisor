import json
from collections import Counter

import pandas as pd
import pytest

from sts2_advisor.features import (
    SKIP, BadHpError, DeckReplayError, FeatureStats, deck_string, load_card_meta, main,
    offer_rows, reward_rows, split_by_group, starting_deck,
)
from sts2_advisor.parse import ParseStats

META = {
    "STRIKE": {"type": "Attack", "rarity": "Basic", "cost": 1},
    "DEFEND": {"type": "Skill", "rarity": "Basic", "cost": 1},
    "ANGER": {"type": "Attack", "rarity": "Common", "cost": 0},
    "INFLAME": {"type": "Power", "rarity": "Uncommon", "cost": 1},
    "HEADBUTT": {"type": "Attack", "rarity": "Common", "cost": 1},
    "CLUMSY": {"type": "Curse", "rarity": "Curse", "cost": -1},
}


def card(cid, **kw):
    return dict({"id": f"CARD.{cid}"}, **kw)


def offer(*cards, picked=None, upgraded=()):
    return [{"card": card(c, **({"current_upgrade_level": 1} if c in upgraded else {})),
             "was_picked": c == picked} for c in cards]


def point(kind, **stats):
    return {"map_point_type": kind, "rooms": [],
            "player_stats": [dict({"player_id": 1, "current_hp": 40, "max_hp": 80,
                                   "current_gold": 99}, **stats)]}


def make_run(history, final_deck, win=False, abandoned=False, relics=(), **extra):
    return dict({
        "ascension": 3, "build_id": "v0.111.0", "win": win, "was_abandoned": abandoned,
        "map_point_history": history,
        "players": [{"id": 1, "character": "CHARACTER.IRONCLAD",
                     "deck": [card(c) for c in final_deck],
                     "relics": [{"id": f"RELIC.R{i}", "floor_added_to_deck": f}
                                for i, f in enumerate(relics)]}],
    }, **extra)


# Starter deck: STRIKE x2, DEFEND x1.
#   floor 1 monster: offered ANGER/INFLAME/HEADBUTT, picks ANGER
#   floor 2 shop:    removes a STRIKE, offered INFLAME (shop, ignored)
#   floor 3 event:   transforms DEFEND -> CLUMSY
#   floor 4 elite:   offered INFLAME/HEADBUTT (INFLAME upgraded), skips
STANDARD_HISTORY = [
    [
        point("monster", card_choices=offer("ANGER", "INFLAME", "HEADBUTT", picked="ANGER"),
              cards_gained=[card("ANGER")]),
        point("shop", cards_removed=[card("STRIKE")],
              card_choices=offer("INFLAME")),
    ],
    [
        point("unknown", cards_transformed=[{"original_card": card("DEFEND"),
                                             "final_card": card("CLUMSY")}]),
        point("elite", current_hp=20,
              card_choices=offer("INFLAME", "HEADBUTT", upgraded=("INFLAME",))),
    ],
]
STANDARD_FINAL = ["STRIKE", "ANGER", "CLUMSY"]


def standard_rows(**kw):
    return offer_rows(make_run(STANDARD_HISTORY, STANDARD_FINAL, **kw), "r1", "mine", META)


def by_offer(rows):
    out = {}
    for r in rows:
        out.setdefault(r["floor"], {})[r["candidate"]] = r
    return out


def test_starting_deck_undoes_all_changes():
    run = make_run(STANDARD_HISTORY, STANDARD_FINAL)
    assert starting_deck(run) == {"STRIKE": 2, "DEFEND": 1}


def test_starting_deck_rejects_inconsistent_log():
    run = make_run([[point("monster", cards_gained=[card("ANGER")])]], ["STRIKE"])
    with pytest.raises(DeckReplayError):
        starting_deck(run)


def test_only_post_combat_rewards_become_offers():
    offers = by_offer(standard_rows())
    assert sorted(offers) == [1, 4]  # shop (floor 2) is excluded
    assert set(offers[1]) == {"ANGER", "INFLAME", "HEADBUTT", SKIP}
    assert offers[4][SKIP]["room_type"] == "elite" and offers[4][SKIP]["act"] == 2


def test_deck_state_is_before_the_pick():
    offers = by_offer(standard_rows())
    first = offers[1]["ANGER"]
    assert first["deck_size"] == 3 and first["cand_copies_in_deck"] == 0
    assert (first["deck_attacks"], first["deck_skills"]) == (2, 1)

    later = offers[4]["HEADBUTT"]
    # STRIKE removed, ANGER added, DEFEND -> CLUMSY
    assert later["deck_size"] == 3
    assert (later["deck_attacks"], later["deck_skills"], later["deck_curses"]) == (2, 0, 1)


def test_labels_and_skip_rows():
    offers = by_offer(standard_rows())
    assert [c for c, r in offers[1].items() if r["picked"]] == ["ANGER"]
    assert [c for c, r in offers[4].items() if r["picked"]] == [SKIP]
    skip = offers[4][SKIP]
    assert skip["is_skip"] and skip["cand_type"] is None and skip["n_offered"] == 2


def test_candidate_features():
    offers = by_offer(standard_rows())
    inflame = offers[4]["INFLAME"]
    assert inflame["cand_type"] == "Power" and inflame["cand_rarity"] == "Uncommon"
    assert inflame["cand_upgraded"] is True
    assert offers[4]["HEADBUTT"]["cand_upgraded"] is False


def test_run_state_features():
    rows = offer_rows(make_run(STANDARD_HISTORY, STANDARD_FINAL, relics=(1, 3, 4)),
                      "r1", "mine", META)
    offers = by_offer(rows)
    assert offers[4][SKIP]["hp_frac"] == 0.25 and offers[4][SKIP]["gold"] == 99
    # relics added on floors 1, 3, 4: before floor 4 only the first two count
    assert offers[1][SKIP]["n_relics"] == 0 and offers[4][SKIP]["n_relics"] == 2


def test_outcome_labels():
    assert {r["run_won"] for r in standard_rows(win=True)} == {True}
    assert {r["run_won"] for r in standard_rows(abandoned=True)} == {None}
    assert {r["floors_reached"] for r in standard_rows()} == {4}


def test_multiplayer_runs_produce_no_rows():
    run = make_run(STANDARD_HISTORY, STANDARD_FINAL)
    run["players"].append(dict(run["players"][0], id=2))
    assert offer_rows(run, "r1", "codex", META) == []


def test_impossible_hp_rejects_run():
    history = [[point("monster", current_hp=12341234, max_hp=73)]]
    with pytest.raises(BadHpError):
        offer_rows(make_run(history, ["STRIKE"]), "r1", "codex", META)


def test_multi_pick_rewards_left_out():
    history = [[
        point("monster", card_choices=offer("ANGER", "INFLAME", "HEADBUTT", "STRIKE",
                                             picked="ANGER") + offer("DEFEND", picked="DEFEND"),
              cards_gained=[card("ANGER"), card("DEFEND")]),
        point("monster", card_choices=offer("ANGER", "HEADBUTT", picked="HEADBUTT"),
              cards_gained=[card("HEADBUTT")]),
    ]]
    stats = FeatureStats(parse=ParseStats())
    rows = offer_rows(make_run(history, ["STRIKE", "ANGER", "DEFEND", "HEADBUTT"]),
                      "r1", "mine", META, stats)
    assert {r["floor"] for r in rows} == {2} and stats.n_multi_pick == 1
    assert rows[0]["deck_size"] == 3  # the left-out reward's picks still count


def test_groups():
    run = make_run(STANDARD_HISTORY, STANDARD_FINAL)
    assert {r["group"] for r in offer_rows(run, "a", "mine", META)} == {"run:a"}
    assert {r["group"] for r in offer_rows(dict(run, player_token="tok"), "b", "codex",
                                           META)} == {"tok"}
    assert {r["group"] for r in offer_rows(run, "c", "codex", META)} == {"run:c"}


def test_split_by_group_keeps_players_on_one_side():
    df = pd.DataFrame({"group": [f"g{i % 10}" for i in range(100)], "x": range(100)})
    train, test = split_by_group(df, test_frac=0.3, seed=1)
    assert len(train) + len(test) == 100
    assert not set(train["group"]) & set(test["group"])
    assert test["group"].nunique() == 3


def test_load_card_meta(tmp_path):
    p = tmp_path / "cards.csv"
    p.write_text("id,type,rarity,color,cost,is_x_cost,traits\n"
                 "BASH,Attack,Basic,ironclad,2,False,Vulnerable\n"
                 "ZAP,Skill,Basic,defect,1,False,\n")
    assert load_card_meta(p) == {
        "BASH": {"type": "Attack", "rarity": "Basic", "cost": 2,
                 "traits": frozenset({"Vulnerable"})},
        "ZAP": {"type": "Skill", "rarity": "Basic", "cost": 1, "traits": frozenset()},
    }


TRAIT_META = {
    "STRIKE": {"type": "Attack", "rarity": "Basic", "cost": 1, "traits": frozenset({"Strike"})},
    "DEFEND": {"type": "Skill", "rarity": "Basic", "cost": 1,
               "traits": frozenset({"Defend", "Block"})},
    "SHRUG": {"type": "Skill", "rarity": "Common", "cost": 1,
              "traits": frozenset({"Block", "Draw"})},
    "PERFECTED": {"type": "Attack", "rarity": "Common", "cost": 2,
                  "traits": frozenset({"Strike"})},
    "CURSE": {"type": "Curse", "rarity": "Curse", "cost": -1, "traits": frozenset()},
}


def test_deck_traits_and_synergy():
    deck = Counter({"STRIKE": 3, "DEFEND": 2, "SHRUG": 1, "CURSE": 1})
    rows = reward_rows({"hp": 10, "max_hp": 20}, [{"id": "PERFECTED"}, {"id": "SHRUG"},
                                                  {"id": "MYSTERY"}], deck, TRAIT_META)
    r = {row["candidate"]: row for row in rows}
    perfected = r["PERFECTED"]
    assert (perfected["deck_strike"], perfected["deck_block"], perfected["deck_draw"]) == (3, 3, 1)
    assert perfected["deck_basics"] == 5
    assert perfected["deck_avg_cost"] == 1.0  # the unplayable curse (cost -1) is ignored
    assert perfected["cand_synergy"] == 3 and perfected["cand_n_traits"] == 1
    assert r["SHRUG"]["cand_synergy"] == 3 + 1  # 3 Block cards + 1 Draw card
    assert r["MYSTERY"]["cand_synergy"] == 0 and r["MYSTERY"]["cand_type"] is None
    assert r[SKIP]["cand_synergy"] is None


def test_deck_cards_before_the_pick():
    offers = by_offer(standard_rows())
    assert offers[1]["ANGER"]["deck_cards"] == "DEFEND|STRIKE|STRIKE"  # starter only
    # floor 4: STRIKE removed, ANGER added, DEFEND -> CLUMSY
    assert offers[4][SKIP]["deck_cards"] == "ANGER|CLUMSY|STRIKE"
    assert deck_string(Counter()) == ""


def test_relics_held_at_each_reward():
    run = make_run(STANDARD_HISTORY, STANDARD_FINAL)
    run["players"][0]["relics"] = [{"id": "RELIC.BURNING_BLOOD", "floor_added_to_deck": 1},
                                   {"id": "RELIC.ANCHOR", "floor_added_to_deck": 3},
                                   {"id": "RELIC.VAJRA", "floor_added_to_deck": 4}]
    offers = by_offer(offer_rows(run, "r1", "mine", META))
    assert offers[1][SKIP]["relics_held"] == "" and offers[1][SKIP]["n_relics"] == 0
    # floor 4: relics from floors 1 and 3 (not the one gained at this reward)
    assert offers[4][SKIP]["relics_held"] == "ANCHOR|BURNING_BLOOD"
    assert offers[4][SKIP]["n_relics"] == 2


def test_main_end_to_end(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "1.run").write_text(json.dumps(
        make_run(STANDARD_HISTORY, STANDARD_FINAL, seed="a", start_time=1)))
    (raw / "2.run").write_text(json.dumps(  # log doesn't add up -> replay error
        make_run([[point("monster", cards_gained=[card("ANGER")])]], ["STRIKE"],
                 seed="b", start_time=2)))
    meta = tmp_path / "cards.csv"
    meta.write_text("id,type,rarity,color,cost,is_x_cost\n" + "".join(
        f"{k},{v['type']},{v['rarity']},x,{v['cost']},False\n" for k, v in META.items()))
    out = tmp_path / "offers.parquet"
    main([str(raw), "--out", str(out), "--card-meta", str(meta), "--allow-mods"])

    df = pd.read_parquet(out)
    assert len(df) == 7 and df["offer_id"].nunique() == 2
    assert df.groupby("offer_id")["picked"].sum().tolist() == [1, 1]
    text = capsys.readouterr().out
    assert "Runs used:         1\n" in text
    assert "features: 1 deck replay failed" in text
    assert "Card rewards:      2  (7 rows incl. SKIP; 0 multi-pick rewards left out)" in text
