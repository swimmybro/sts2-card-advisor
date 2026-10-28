from sts2_advisor.cards import TRAITS, card_traits, is_defensive, write_cards_csv
from sts2_advisor.features import load_card_meta


def test_card_traits_from_catalog_fields():
    blade_dance = {"id": "BLADE_DANCE", "keywords": ["Exhaust"], "spawns_cards": ["SHIV"],
                   "cards_draw": 0, "target": "Self"}
    assert card_traits(blade_dance) == ["Exhaust", "Shiv"]

    bash = {"id": "BASH", "powers_applied": [{"power": "Vulnerable", "amount": 2}],
            "tags": None, "damage": 8, "target": "AnyEnemy"}
    assert card_traits(bash) == ["Vulnerable"]

    shrug = {"id": "SHRUG_IT_OFF", "block": 8, "cards_draw": 1}
    assert card_traits(shrug) == ["Block", "Draw"]

    odd = {"id": "X", "keywords": ["Unplayable"], "powers_applied": [{"power": "Juggernaut"}],
           "target": "AllEnemies", "is_x_cost": True}
    assert card_traits(odd) == ["AoE", "XCost"]  # untracked keyword/power ignored


def test_traits_are_unique_and_ordered():
    assert len(TRAITS) == len(set(TRAITS))
    assert TRAITS.index("Exhaust") < TRAITS.index("Block")


def test_csv_round_trip(tmp_path):
    catalog = [{"id": "SHRUG_IT_OFF", "type": "Skill", "rarity": "Common", "color": "ironclad",
                "cost": 1, "block": 8, "cards_draw": 1},
               {"id": "ABRASIVE", "type": "Power", "rarity": "Rare", "color": "silent",
                "cost": 3}]
    out = tmp_path / "cards.csv"
    write_cards_csv(catalog, out)
    meta = load_card_meta(out)
    assert meta["SHRUG_IT_OFF"]["traits"] == frozenset({"Block", "Draw"})
    assert (meta["SHRUG_IT_OFF"]["draw"], meta["SHRUG_IT_OFF"]["defensive"]) == (1, True)
    assert meta["ABRASIVE"] == {"type": "Power", "rarity": "Rare", "cost": 3,
                                "traits": frozenset(), "draw": 0, "defensive": False}


def test_is_defensive():
    assert is_defensive({"block": 5})
    assert is_defensive({"powers_applied": [{"power": "Weak", "amount": 1}]})
    assert not is_defensive({"damage": 6, "powers_applied": [{"power": "Vulnerable"}]})
