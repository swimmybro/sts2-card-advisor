import pytest
from fastapi.testclient import TestClient

from sts2_advisor.serve import create_app, normalize_card, normalize_id
from sts2_advisor.train import run
from test_train import config, synthetic_offers


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """API over a small model trained on synthetic rewards where GOOD is always
    taken, and with no GOOD the player skips at low HP."""
    tmp = tmp_path_factory.mktemp("api")
    synthetic_offers().to_parquet(tmp / "offers.parquet")
    run(config(tmp))
    meta = tmp / "cards.csv"  # costs must match synthetic_offers' cand_cost
    meta.write_text("id,type,rarity,color,cost,is_x_cost\n" + "".join(
        f"{c},Attack,Common,ironclad,{cost},False\n"
        for c, cost in {"GOOD": 1, "A": 0, "B": 1, "C": 2, "D": 2, "E": 3}.items()))
    return TestClient(create_app(model_dir=tmp / "model", card_meta=meta))


def state(**kw):
    return dict({"character": "ironclad", "ascension": 5, "act": 1, "floor": 3,
                 "hp": 60, "max_hp": 80, "gold": 99, "deck": ["Strike", "Defend+"],
                 "relics": ["Burning Blood"], "offered": ["A", "Good", "B"]}, **kw)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["best_iteration"] is not None


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


def test_normalize_card():
    assert normalize_card("Shrug It Off") == ("SHRUG_IT_OFF", False)
    assert normalize_card("CARD.BASH+") == ("BASH", True)
    assert normalize_card(" card.pommel_strike ") == ("POMMEL_STRIKE", False)


def test_normalize_relic():
    assert normalize_id("Burning Blood", "relic") == "BURNING_BLOOD"
    assert normalize_id("RELIC.NEOWS_TALISMAN", "relic") == "NEOWS_TALISMAN"
