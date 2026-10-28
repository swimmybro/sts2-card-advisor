"""HTTP API that ranks the cards in a card reward.

Send the current game state and the offered cards; get them back ranked
(plus ``SKIP``) by the trained card-pick model. Features are computed with
the same code as the training data (:func:`features.reward_rows`).

Run::

    MODEL_DIR=models/card_pick .venv/bin/uvicorn --factory sts2_advisor.serve:create_app

Then open http://127.0.0.1:8000/docs for interactive docs.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator, model_validator

from sts2_advisor.features import DEFAULT_CARD_META, SKIP, load_card_meta, reward_rows
from sts2_advisor.parse import OFFICIAL_CHARACTERS
from sts2_advisor.train import load_model, predict

DEFAULT_MODEL_DIR = Path("models/card_pick")


def normalize_card(name: str) -> Tuple[str, bool]:
    """'Shrug It Off+' / 'CARD.SHRUG_IT_OFF' -> ('SHRUG_IT_OFF', upgraded?)."""
    name = name.strip()
    upgraded = name.endswith("+")
    name = re.sub(r"^card\.", "", name.rstrip("+").strip(), flags=re.IGNORECASE)
    return re.sub(r"[\s']+", "_", name).upper(), upgraded


class OfferedCard(BaseModel):
    id: str = Field(description="Card name or ID, e.g. 'Inflame' or 'INFLAME'")
    upgraded: bool = False


class RankRequest(BaseModel):
    character: str = Field(examples=["IRONCLAD"])
    ascension: int = Field(0, ge=0, le=20)
    act: int = Field(1, ge=1, le=5)
    floor: int = Field(ge=1, examples=[5])
    room_type: Literal["monster", "elite", "boss"] = "monster"
    hp: int = Field(ge=0, examples=[52])
    max_hp: int = Field(gt=0, examples=[80])
    gold: int = Field(0, ge=0)
    deck: List[str] = Field(
        description="Current deck before the pick; append '+' for upgraded cards",
        examples=[["Strike Ironclad"] * 5 + ["Defend Ironclad"] * 4 + ["Bash"]])
    relics: List[str] = Field(default_factory=list, examples=[["Burning Blood"]])
    offered: List[Union[OfferedCard, str]] = Field(
        min_length=1, max_length=10, examples=[["Inflame", "Anger", "Headbutt+"]])

    @field_validator("character")
    @classmethod
    def _official_character(cls, v: str) -> str:
        v = re.sub(r"^character\.", "", v.strip(), flags=re.IGNORECASE).upper()
        if v not in OFFICIAL_CHARACTERS:
            raise ValueError(f"must be one of {sorted(OFFICIAL_CHARACTERS)}")
        return v

    @model_validator(mode="after")
    def _hp_in_range(self) -> "RankRequest":
        if self.hp > self.max_hp:
            raise ValueError("hp cannot exceed max_hp")
        return self


class RankedCard(BaseModel):
    card: str
    upgraded: bool
    is_skip: bool
    score: float
    known: bool = Field(description="False if the model never saw this card in training")


class RankResponse(BaseModel):
    suggestion: str
    ranking: List[RankedCard]
    unknown_cards: List[str]


def create_app(model_dir: Optional[Path] = None, card_meta: Optional[Path] = None) -> FastAPI:
    """Build the app. Paths default to ``$MODEL_DIR`` / ``$CARD_META``."""
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    meta = load_card_meta(Path(card_meta or os.environ.get("CARD_META", DEFAULT_CARD_META)))
    booster, categories = load_model(model_dir)
    known_cards = set(categories["candidate"])
    metrics_path = model_dir / "metrics.json"
    model_info = {"model_dir": str(model_dir)}
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        model_info.update(best_iteration=m.get("best_iteration"),
                          train_filter=m.get("train_filter"),
                          test_top1=m.get("test", {}).get("model", {}).get("top1"))

    app = FastAPI(title="STS2 Card Advisor",
                  description="Ranks the cards in a Slay the Spire 2 card reward.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **model_info}

    @app.post("/rank", response_model=RankResponse)
    def rank(req: RankRequest) -> RankResponse:
        deck = Counter(normalize_card(c)[0] for c in req.deck)
        candidates = []
        for c in req.offered:
            cid, up = normalize_card(c) if isinstance(c, str) else normalize_card(c.id)
            candidates.append({"id": cid, "upgraded": up or getattr(c, "upgraded", False)})
        base = {"character": req.character, "ascension": req.ascension, "act": req.act,
                "floor": req.floor, "room_type": req.room_type, "hp": req.hp,
                "max_hp": req.max_hp, "gold": req.gold, "n_relics": len(req.relics)}
        rows = pd.DataFrame(reward_rows(base, candidates, deck, meta))
        rows["score"] = predict(booster, categories, rows)
        rows = rows.sort_values("score", ascending=False)
        ranking = [
            RankedCard(card=r.candidate, upgraded=bool(r.cand_upgraded) if not r.is_skip else False,
                       is_skip=bool(r.is_skip), score=round(float(r.score), 4),
                       known=bool(r.is_skip) or r.candidate in known_cards)
            for r in rows.itertuples()
        ]
        return RankResponse(
            suggestion=ranking[0].card,
            ranking=ranking,
            unknown_cards=sorted({c["id"] for c in candidates} - known_cards),
        )

    return app
