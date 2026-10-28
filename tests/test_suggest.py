import json
import time

import pytest

from make_demo_models import main as make_demo_models
from sts2_advisor.advise import Advisor
from sts2_advisor.suggest import _cards_in, advise, main, read_save


@pytest.fixture(scope="module")
def advisor(tmp_path_factory):
    models = tmp_path_factory.mktemp("models")
    make_demo_models(models)
    return Advisor(models / "card_pick", None, models / "win_prob")


def save(path, players=None, pending=None, saved=None, **kw):
    """A current_run.save with the fields the advisor reads."""
    data = dict({
        "ascension": 5, "current_act_index": 1, "save_time": saved or int(time.time()),
        "map_point_history": [[{}] * 17, [{}] * 4],  # 21 floors finished
        "pre_finished_room": pending,
        "players": players or [{
            "character_id": "CHARACTER.IRONCLAD", "current_hp": 50, "max_hp": 80, "gold": 99,
            "deck": [{"id": "CARD.STRIKE", "current_upgrade_level": 1}, {"id": "CARD.DEFEND"}],
            "relics": [{"id": "RELIC.BURNING_BLOOD"}]}],
    }, **kw)
    path.write_text(json.dumps(data))
    return path


def test_read_save(tmp_path):
    s = read_save(save(tmp_path / "current_run.save"))
    assert (s["character"], s["ascension"], s["act"], s["floor"]) == ("IRONCLAD", 5, 2, 22)
    assert (s["hp"], s["max_hp"], s["gold"]) == (50, 80, 99)
    assert s["deck"] == [("STRIKE", True), ("DEFEND", False)]
    assert s["relics"] == ["BURNING_BLOOD"] and s["offered_in_save"] == []
    assert s["saved_seconds_ago"] < 60


def test_coop_save_needs_a_player(tmp_path):
    p = {"character_id": "CHARACTER.SILENT", "current_hp": 1, "max_hp": 1, "deck": [], "relics": []}
    path = save(tmp_path / "s.save", players=[p, dict(p, character_id="CHARACTER.DEFECT")])
    with pytest.raises(ValueError, match="co-op save.*--player"):
        read_save(path)
    assert read_save(path, player=1)["character"] == "DEFECT"


def test_offered_cards_found_in_an_open_reward_screen():
    pending = {"room_type": "elite", "rewards": [{"card_choices": [
        {"card": {"id": "CARD.ANGER"}, "was_picked": False},
        {"card": {"id": "CARD.INFLAME", "current_upgrade_level": 1}, "was_picked": False}]}]}
    assert _cards_in(pending) == [("ANGER", False), ("INFLAME", True)]


def test_advise_text(tmp_path, advisor):
    state = read_save(save(tmp_path / "s.save"))
    text = advise(advisor, state, [("GOOD", False), ("A", True), ("TYPO_CARD", False)])
    assert text.startswith("Ironclad A5, act 2 floor 22 (monster), HP 50/80, 99 gold, 2-card deck")
    assert "Take: GOOD" in text  # GOOD is what wins in the demo data
    assert "Not considered: TYPO_CARD" in text and "A+" in text


def test_main(tmp_path, advisor, monkeypatch, capsys):
    monkeypatch.setenv("MODEL_DIR", str(advisor.model_dir))
    monkeypatch.setenv("WIN_MODEL_DIR", str(advisor.win_model_dir))
    path = save(tmp_path / "s.save", saved=int(time.time()) - 5 * 86400)
    main(["Good", "A", "--elite", "--save", str(path)])
    out = capsys.readouterr().out
    assert "the save is from 5 days ago" in out and "(elite)" in out and "Take:" in out

    with pytest.raises(SystemExit, match="Name the offered cards"):
        main(["--save", str(path)])
    with pytest.raises(SystemExit, match="No run in progress"):
        main(["Good", "--save", str(tmp_path / "missing.save")])
