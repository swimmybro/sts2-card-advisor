import pytest
from fastapi.testclient import TestClient

from sts2_advisor import serve
from sts2_advisor.serve import create_app, normalize_card, normalize_id
from sts2_advisor.train import run
from sts2_advisor.win import run as run_win
from test_train import config, synthetic_offers
from test_win import win_config


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A small pick model trained on synthetic rewards where GOOD is always
    taken, and with no GOOD the player skips at low HP; plus a win model."""
    tmp = tmp_path_factory.mktemp("api")
    synthetic_offers().to_parquet(tmp / "offers.parquet")
    run(config(tmp))
    run_win(win_config(tmp), pick_model=tmp / "model")  # also writes reliability.json
    meta = tmp / "cards.csv"  # costs must match synthetic_offers' cand_cost
    meta.write_text("id,type,rarity,color,cost,is_x_cost\n" + "".join(
        f"{c},Attack,Common,ironclad,{cost},False\n"
        for c, cost in {"GOOD": 1, "A": 0, "B": 1, "C": 2, "D": 2, "E": 3}.items()))
    return tmp, meta


@pytest.fixture(scope="module")
def client(trained):
    tmp, meta = trained
    return TestClient(create_app(model_dir=tmp / "model", card_meta=meta,
                                 win_model_dir=tmp / "no_win_model"))


@pytest.fixture(scope="module")
def win_client(trained):
    tmp, meta = trained
    return TestClient(create_app(model_dir=tmp / "model", card_meta=meta,
                                 win_model_dir=tmp / "win"))


def state(**kw):
    return dict({"character": "ironclad", "ascension": 5, "act": 1, "floor": 3,
                 "hp": 60, "max_hp": 80, "gold": 99, "deck": ["Strike", "Defend+"],
                 "relics": ["Burning Blood"], "offered": ["A", "Good", "B"]}, **kw)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["best_iteration"] is not None
    assert body["win_model_dir"] is None


def test_no_win_model_means_no_win_fields(client):
    body = client.post("/rank", json=state()).json()
    # without a win model, the pick model's top option is the suggestion
    assert body["suggestion_source"] == "pick_model"
    assert body["suggestion"] == body["popular_choice"]
    assert all(r["win_chance"] is None for r in body["ranking"])


def test_win_model_drives_the_suggestion(win_client):
    assert win_client.get("/health").json()["win_model_dir"].endswith("win")
    body = win_client.post("/rank", json=state()).json()
    chances = [r["win_chance"] for r in body["ranking"]]
    assert {r["card"] for r in body["ranking"]} == {"GOOD", "A", "B", "SKIP"}
    assert all(0.0 <= c <= 1.0 for c in chances)
    assert body["suggestion"] == body["ranking"][0]["card"]  # suggestion listed first
    assert chances[1:] == sorted(chances[1:], reverse=True)  # the rest by win chance
    assert body["popular_choice"] == "GOOD"  # what players take, from the pick model
    if body["suggestion_source"] == "win_model":
        assert chances[0] == max(chances)
    else:  # several tied on win chance: the one players take most
        assert body["suggestion_source"] == "win_tie_popular"
        assert body["suggestion"] in body["win_tied"]


def test_reliability_fields(win_client):
    body = win_client.post("/rank", json=state()).json()
    ranking = {r["card"]: r for r in body["ranking"]}
    assert sum(r["pick_chance"] for r in ranking.values()) == pytest.approx(1.0, abs=1e-3)
    for r in ranking.values():
        assert 0.0 <= r["raw_win_chance"] <= 1.0
        assert r["low_confidence"] == (r["pick_chance"] < 0.05)
    # GOOD is always taken in training, so it is never low-confidence
    assert ranking["GOOD"]["low_confidence"] is False
    assert isinstance(body["win_tied"], list)
    if body["win_tied"]:
        assert len(body["win_tied"]) > 1 and body["suggestion"] in body["win_tied"]
        assert body["suggestion_source"] == "win_tie_popular"


def test_rank_suggests_learned_pick(client):
    body = client.post("/rank", json=state()).json()
    assert body["suggestion"] == "GOOD"
    assert [r["card"] for r in body["ranking"]][0] == "GOOD"
    assert {r["card"] for r in body["ranking"]} == {"GOOD", "A", "B", "SKIP"}
    scores = [r["score"] for r in body["ranking"]]
    assert scores == sorted(scores, reverse=True)


def test_rank_suggests_skip_at_low_hp(client):
    body = client.post("/rank", json=state(hp=5, offered=["C", "D", "E"])).json()
    assert body["suggestion"] == "SKIP"
    assert body["ranking"][0]["is_skip"] is True


def test_unknown_cards_flagged(client):
    body = client.post("/rank", json=state(offered=["Good", "Totally New Card"])).json()
    assert body["unknown_cards"] == ["TOTALLY_NEW_CARD"]
    known = {r["card"]: r["known"] for r in body["ranking"]}
    assert known == {"GOOD": True, "TOTALLY_NEW_CARD": False, "SKIP": True}


def test_upgraded_deck_cards_count_toward_upgrade_density(client, monkeypatch):
    seen = {}
    real = serve.reward_rows

    def spy(base, *a, **kw):
        seen.update(base)
        return real(base, *a, **kw)

    monkeypatch.setattr(serve, "reward_rows", spy)
    client.post("/rank", json=state(deck=["Strike+", "Defend+", "Bash"]))
    assert seen["deck_upgraded"] == 2


def test_upgrades_from_object_or_plus(client):
    body = client.post("/rank", json=state(
        offered=[{"id": "A", "upgraded": True}, "B+", "Good"])).json()
    upgraded = {r["card"]: r["upgraded"] for r in body["ranking"]}
    assert upgraded == {"A": True, "B": True, "GOOD": False, "SKIP": False}


@pytest.mark.parametrize("bad", [
    {"character": "Hermit"},
    {"hp": 90, "max_hp": 80},
    {"offered": []},
    {"floor": 0},
])
def test_invalid_requests_rejected(client, bad):
    assert client.post("/rank", json=state(**bad)).status_code == 422


def test_missing_pick_model_gives_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="No pick model"):
        create_app(model_dir=tmp_path / "nothing_here")


def test_normalize_card():
    assert normalize_card("Shrug It Off") == ("SHRUG_IT_OFF", False)
    assert normalize_card("CARD.BASH+") == ("BASH", True)
    assert normalize_card(" card.pommel_strike ") == ("POMMEL_STRIKE", False)


def test_normalize_relic():
    assert normalize_id("Burning Blood", "relic") == "BURNING_BLOOD"
    assert normalize_id("RELIC.NEOWS_TALISMAN", "relic") == "NEOWS_TALISMAN"
