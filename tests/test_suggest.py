import json
import time

import pytest

from make_demo_models import main as make_demo_models
from sts2_advisor import screen, suggest
from sts2_advisor.advise import Advisor
from sts2_advisor.suggest import advise, main, read_save


@pytest.fixture(scope="module")
def advisor(tmp_path_factory):
    models = tmp_path_factory.mktemp("models")
    make_demo_models(models)
    return Advisor(models / "card_pick", None, models / "win_prob")


def save(path, players=None, pending=None, saved=None, **kw):
    """A current_run.save with the fields the advisor reads."""
    data = dict({
        "ascension": 5, "current_act_index": 1, "save_time": saved or int(time.time()),
        "map_point_history": [[{}] * 17, [{}] * 4],  # 21 points: floor 21 is the fight just won
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
    assert (s["character"], s["ascension"], s["act"], s["floor"]) == ("IRONCLAD", 5, 2, 21)
    assert (s["hp"], s["max_hp"], s["gold"]) == (50, 80, 99)
    assert s["deck"] == [("STRIKE", True), ("DEFEND", False)]
    assert s["relics"] == ["BURNING_BLOOD"] and s["reward_open"] is False
    assert s["saved_seconds_ago"] < 60


def test_coop_save_needs_a_player(tmp_path):
    p = {"character_id": "CHARACTER.SILENT", "current_hp": 1, "max_hp": 1, "deck": [], "relics": []}
    path = save(tmp_path / "s.save", players=[p, dict(p, character_id="CHARACTER.DEFECT")])
    with pytest.raises(ValueError, match="co-op save.*--player"):
        read_save(path)
    assert read_save(path, player=1)["character"] == "DEFECT"


def test_reward_open_from_the_save(tmp_path):
    assert read_save(save(tmp_path / "a.save"))["reward_open"] is False
    open_reward = save(tmp_path / "b.save", pending={"room_type": "elite", "is_pre_finished": True})
    assert read_save(open_reward)["reward_open"] is True
    shop = save(tmp_path / "c.save", pending={"room_type": "shop"})
    assert read_save(shop)["reward_open"] is False


def test_auto_advises_once_per_reward_when_the_screen_settles(tmp_path, advisor, monkeypatch,
                                                              capsys):
    path = save(tmp_path / "s.save", pending={"room_type": "monster", "is_pre_finished": True},
                start_time=1)
    # the screen shows a half-drawn reward, then settles on the full one
    frames = iter([[("GOOD", False)], [("GOOD", False), ("A", False)],
                   [("GOOD", False), ("A", False)], [("GOOD", False), ("A", False)]])
    monkeypatch.setattr(suggest, "cards_on_screen", lambda *a, **k: next(frames))
    notes = []
    monkeypatch.setattr(screen, "notify", lambda title, msg: notes.append(msg))
    suggest.auto(advisor, path, interval=0, max_checks=4)
    out = capsys.readouterr().out
    assert out.count("Take: ") == 1 and "Take: GOOD" in out  # once, not on every check
    assert notes == ["Take GOOD"]


def test_auto_advises_on_each_card_choice_of_the_same_fight(tmp_path, advisor, monkeypatch,
                                                            capsys):
    """One fight can give two card rewards (e.g. from a relic): each gets advice."""
    path = save(tmp_path / "s.save", pending={"room_type": "elite", "is_pre_finished": True},
                start_time=1)
    first, second = [("A", False), ("B", False)], [("GOOD", False), ("A", True)]
    frames = iter([first, first, first, second, second, second])
    monkeypatch.setattr(suggest, "cards_on_screen", lambda *a, **k: next(frames))
    monkeypatch.setattr(screen, "notify", lambda title, msg: None)
    suggest.auto(advisor, path, interval=0, max_checks=6)
    out = capsys.readouterr().out
    assert out.count("Take: ") == 2  # one for each card choice, none repeated
    assert "GOOD" in out.split("Take: ")[2]


def test_advise_text(tmp_path, advisor):
    state = read_save(save(tmp_path / "s.save"))
    text = advise(advisor, state, [("GOOD", False), ("A", True), ("TYPO_CARD", False)])
    assert text.startswith("Ironclad A5, act 2 floor 21 (monster), HP 50/80, 99 gold, 2-card deck")
    assert "Take: GOOD" in text  # GOOD is what wins in the demo data
    assert "Not considered: TYPO_CARD" in text and "A+" in text


def test_fight_type_from_the_open_reward_screen(tmp_path, advisor):
    # shaped like a real save at a reward screen: the room, but not the offered cards
    pending = {"encounter_id": "ENCOUNTER.GREMLIN_NOB", "is_pre_finished": True,
               "room_type": "elite"}
    state = read_save(save(tmp_path / "s.save", pending=pending))
    assert state["room_type"] == "elite" and state["reward_open"] is True
    assert "(elite)" in advise(advisor, state, [("GOOD", False)])
    # a typed flag wins over the save
    assert "(boss)" in advise(advisor, state, [("GOOD", False)], room_type="boss")
    # no open reward and no flag: a normal fight
    plain = read_save(save(tmp_path / "p.save"))
    assert "(monster)" in advise(advisor, plain, [("GOOD", False)])


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


def test_auto_advises_at_the_start_of_an_act(tmp_path, advisor, monkeypatch, capsys):
    from test_ancient import synthetic
    from sts2_advisor import ancient
    synthetic().to_parquet(tmp_path / "a.parquet")
    ancient.train(tmp_path / "a.parquet", tmp_path / "anc")
    ancients = ancient.AncientAdvisor(tmp_path / "anc")
    path = save(tmp_path / "s.save", start_time=1, map_point_history=[[{}]])  # Neow's room
    assert read_save(path)["at_ancient"] is True
    frames = iter([["BAD", "GOOD", "MEH"]] * 3)
    monkeypatch.setattr(suggest, "options_on_screen", lambda *a, **k: next(frames))
    monkeypatch.setattr(suggest, "cards_on_screen", lambda *a, **k: pytest.fail("not a reward"))
    notes = []
    monkeypatch.setattr(screen, "notify", lambda title, msg: notes.append(msg))
    suggest.auto(advisor, path, interval=0, max_checks=3, ancients=ancients)
    out = capsys.readouterr().out
    assert out.count("Take: ") == 1 and "Take: GOOD" in out and "Neow's choice" in out
    assert notes == ["Take GOOD"]
    # later in the act: no ancient
    later = save(tmp_path / "t.save", map_point_history=[[{}, {}]])
    assert read_save(later)["at_ancient"] is False
