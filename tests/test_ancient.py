import numpy as np
import pandas as pd
import pytest

from sts2_advisor import ancient
from sts2_advisor.ancient import AncientAdvisor, ancient_rows, normalize_option
from test_features import META, card, make_run, point


def neow(*options, chosen, model="EVENT.NEOW", **stats):
    p = point("ancient", ancient_choice=[{"TextKey": o, "was_chosen": o == chosen}
                                         for o in options], **stats)
    p["rooms"] = [{"model_id": model, "room_type": "event"}]
    return p


def test_rows_hold_every_option_and_the_state_before_the_choice():
    history = [
        [neow("A", "B", "C", chosen="B", current_hp=70), point("monster", current_hp=55,
                                                              cards_gained=[card("ANGER")])],
        [neow("D", "E", "F", chosen="F", model="EVENT.PAEL", current_hp=80, current_gold=150),
         neow("X", "Y", "Z", chosen="X", model="EVENT.MODDED_ANCIENT")],
    ]
    run = make_run(history, ["STRIKE", "STRIKE", "DEFEND", "ANGER"], win=True,
                   relics=(0, 1, 2))
    rows = ancient_rows(run, "r1", "mine", META)
    assert [r["option"] for r in rows] == ["A", "B", "C", "D", "E", "F"]  # modded one left out
    assert [r["picked"] for r in rows] == [False, True, False, False, False, True]
    neow_row, pael_row = rows[0], rows[3]
    assert (neow_row["ancient"], neow_row["act"], neow_row["floor"]) == ("NEOW", 1, 1)
    assert neow_row["hp"] is None and neow_row["deck_size"] == 3  # nothing before Neow
    # Pael: HP from the floor before, deck with the card gained there, relics from before
    assert (pael_row["ancient"], pael_row["act"], pael_row["floor"]) == ("PAEL", 2, 3)
    assert pael_row["hp"] == 55 and pael_row["deck_size"] == 4 and pael_row["n_relics"] == 3
    assert all(r["run_won"] == 1.0 and r["n_options"] == 3 for r in rows)


def test_normalize_option():
    assert normalize_option("Arcane Scroll") == "ARCANE_SCROLL"
    assert normalize_option("RELIC.PAELS_FLESH") == "PAELS_FLESH"
    assert normalize_option("Pael's Flesh") == "PAELS_FLESH"


def synthetic(n=3000, seed=0):
    """Neow offers three of GOOD/OK/BAD/MEH; GOOD is usually taken and wins most."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        options = list(rng.choice(["GOOD", "OK", "BAD", "MEH"], 3, replace=False))
        picked = "GOOD" if "GOOD" in options and rng.random() < 0.8 else options[rng.integers(3)]
        won = rng.random() < {"GOOD": 0.7, "OK": 0.4, "BAD": 0.1, "MEH": 0.3}[picked]
        for o in options:
            rows.append(dict({c: 1.0 for c in ancient.NUMERIC}, choice_id=f"c{i}",
                             group=f"g{i}", run_id=f"r{i}", source="community",
                             character="IRONCLAD", ancient="NEOW", option=o,
                             picked=o == picked, run_won=float(won)))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ancient")
    synthetic().to_parquet(tmp / "ancients.parquet")
    metrics = ancient.train(tmp / "ancients.parquet", tmp / "model")
    assert metrics["test"]["win_auc"] > 0.6 and metrics["test"]["pick_top1"] > 0.6
    return tmp / "model"


def test_advisor_suggests_the_winning_option(model_dir):
    adv = AncientAdvisor(model_dir)
    assert adv.ancient_for(["GOOD", "BAD"]) == "NEOW"
    state = {"character": "IRONCLAD", "ascension": 1, "act": 1, "floor": 1,
             "hp": 80, "max_hp": 80, "gold": 99}
    rows = adv.rank(state, [("STRIKE", False), ("DEFEND", True)], ["BURNING_BLOOD"],
                    ["BAD", "GOOD", "MEH"])
    assert rows.iloc[0]["option"] == "GOOD" and rows.iloc[0]["suggested"]
    assert rows["win"].iloc[0] > rows["win"].iloc[1:].max()
    # an option the models never saw is never suggested
    rows = adv.rank(state, [], [], ["BAD", "NEW_RELIC"])
    assert not rows.set_index("option").loc["NEW_RELIC", "known"]
    assert rows.iloc[0]["option"] == "BAD"
