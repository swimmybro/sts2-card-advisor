"""HTTP API that ranks the cards in a card reward.

Send the current game state and the offered cards; get them back ranked
(plus ``SKIP``) by the trained card-pick model. Features are computed with
the same code as the training data (:func:`features.reward_rows`).

Run::

    MODEL_DIR=models/card_pick .venv/bin/uvicorn --factory sts2_advisor.serve:create_app

Then open http://127.0.0.1:8000/docs for interactive docs.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator, model_validator

from sts2_advisor.advise import Advisor
from sts2_advisor.features import relic_fields, reward_rows
from sts2_advisor.parse import OFFICIAL_CHARACTERS


def normalize_id(name: str, prefix: str) -> str:
    """'Burning Blood' / 'RELIC.BURNING_BLOOD' -> 'BURNING_BLOOD' (prefix='relic')."""
    name = re.sub(rf"^{prefix}\.", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(r"[\s']+", "_", name).upper()


def normalize_card(name: str) -> Tuple[str, bool]:
    """'Shrug It Off+' / 'CARD.SHRUG_IT_OFF' -> ('SHRUG_IT_OFF', upgraded?)."""
    name = name.strip()
    return normalize_id(name.rstrip("+").strip(), "card"), name.endswith("+")


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
    score: float = Field(description="Pick-model score: higher = players more likely to take it")
    pick_chance: Optional[float] = Field(
        None, description="Estimated share of players who would take this option")
    win_chance: Optional[float] = Field(
        None, description="Predicted chance of winning the run if you take it, adjusted "
                          "toward the reward's typical outcome when few players take it")
    raw_win_chance: Optional[float] = Field(
        None, description="The win model's estimate before that adjustment")
    low_confidence: Optional[bool] = Field(
        None, description="True when few players take this option, so its win chance "
                          "rests on little evidence")
    known: bool = Field(description="False if the model never saw this card in training")


class RankResponse(BaseModel):
    suggestion: str = Field(
        description="The recommended option: highest win chance (or, without a win "
                    "model, what players most often take)")
    suggestion_source: Literal["win_model", "pick_model"]
    popular_choice: str = Field(description="What players most often take here (pick model)")
    win_tied: List[str] = Field(
        default_factory=list,
        description="Options whose win chance is within the tie margin of the best; "
                    "empty unless two or more are effectively tied")
    ranking: List[RankedCard] = Field(
        description="All options, best first (by win chance when a win model is loaded)")
    unknown_cards: List[str]


def create_app(model_dir: Optional[Path] = None, card_meta: Optional[Path] = None,
               win_model_dir: Optional[Path] = None) -> FastAPI:
    """Build the app. Paths default to ``$MODEL_DIR``, ``$CARD_META`` and
    ``$WIN_MODEL_DIR``; the win model is optional and skipped if missing."""
    advisor = Advisor(model_dir, card_meta, win_model_dir)
    model_info = advisor.info()

    app = FastAPI(title="STS2 Card Advisor",
                  description="Ranks the cards in a Slay the Spire 2 card reward.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **model_info}

    @app.post("/rank", response_model=RankResponse)
    def rank(req: RankRequest) -> RankResponse:
        deck_cards = [normalize_card(c) for c in req.deck]
        deck = Counter(cid for cid, _ in deck_cards)
        candidates = []
        for c in req.offered:
            cid, up = normalize_card(c) if isinstance(c, str) else normalize_card(c.id)
            candidates.append({"id": cid, "upgraded": up or getattr(c, "upgraded", False)})
        base = {"character": req.character, "ascension": req.ascension, "act": req.act,
                "floor": req.floor, "room_type": req.room_type, "hp": req.hp,
                "max_hp": req.max_hp, "gold": req.gold,
                "deck_upgraded": sum(up for _, up in deck_cards),
                **relic_fields(normalize_id(r, "relic") for r in req.relics)}
        rows = pd.DataFrame(reward_rows(base, candidates, deck, advisor.meta)).assign(offer_id=0)
        rows = advisor.score(rows)
        tied = rows[rows["tied"]].sort_values("win", ascending=False)["candidate"].tolist()
        popular = rows.loc[rows["score"].idxmax(), "candidate"]
        rows = rows.sort_values("win" if advisor.win else "score", ascending=False)

        def num(v):
            return None if np.isnan(v) else round(float(v), 4)

        ranking = [
            RankedCard(card=r.candidate, upgraded=bool(r.cand_upgraded) if not r.is_skip else False,
                       is_skip=bool(r.is_skip), score=round(float(r.score), 4),
                       pick_chance=num(r.pick_chance), win_chance=num(r.win),
                       raw_win_chance=num(r.raw_win) if advisor.reliability else None,
                       low_confidence=(bool(r.low_confidence) if advisor.reliability else None),
                       known=bool(r.is_skip) or r.candidate in advisor.known_cards)
            for r in rows.itertuples()
        ]
        return RankResponse(
            suggestion=ranking[0].card,
            suggestion_source="win_model" if advisor.win else "pick_model",
            popular_choice=popular,
            win_tied=tied,
            ranking=ranking,
            unknown_cards=sorted({c["id"] for c in candidates} - advisor.known_cards),
        )

    return app
