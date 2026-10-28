"""Score the options at card rewards with the pick and win models.

Shared by the API and the ``suggest`` command (one live reward) and the run
reviewer (every reward in a finished run), so all produce identical numbers.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sts2_advisor.features import DEFAULT_CARD_META, load_card_meta, relic_fields, reward_rows
from sts2_advisor.train import add_pick_features, load_model, predict
from sts2_advisor.win import adjust_win_chances, near_best, pick_probabilities

DEFAULT_MODEL_DIR = Path("models/card_pick")
DEFAULT_WIN_MODEL_DIR = Path("models/win_prob")


class Advisor:
    """The pick model, the optional win model and its reliability settings."""

    def __init__(self, model_dir: Optional[Path] = None, card_meta: Optional[Path] = None,
                 win_model_dir: Optional[Path] = None):
        """Paths default to ``$MODEL_DIR``, ``$CARD_META`` and ``$WIN_MODEL_DIR``;
        the win model is optional and skipped if missing."""
        self.model_dir = Path(model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
        if not (self.model_dir / "model.txt").exists():
            raise RuntimeError(
                f"No pick model at {self.model_dir}. Train one with "
                "`python -m sts2_advisor.train`, or set MODEL_DIR (in Docker, mount your "
                "models: -v \"$PWD/models:/app/models:ro\").")
        self.meta = load_card_meta(Path(card_meta or os.environ.get("CARD_META",
                                                                    DEFAULT_CARD_META)))
        self.pick = load_model(self.model_dir)
        self.known_cards = set(self.pick[1]["candidate"])
        win_dir = Path(win_model_dir or os.environ.get("WIN_MODEL_DIR", DEFAULT_WIN_MODEL_DIR))
        self.win = load_model(win_dir) if (win_dir / "model.txt").exists() else None
        self.win_model_dir = win_dir if self.win else None
        rel = win_dir / "reliability.json"
        self.reliability: Optional[Dict[str, Any]] = (
            json.loads(rel.read_text()) if self.win and rel.exists() else None)

    def info(self) -> Dict[str, Any]:
        out = {"model_dir": str(self.model_dir),
               "win_model_dir": str(self.win_model_dir) if self.win else None,
               "reliability": self.reliability}
        metrics = self.model_dir / "metrics.json"
        if metrics.exists():
            m = json.loads(metrics.read_text())
            out.update(best_iteration=m.get("best_iteration"),
                       train_filter=m.get("train_filter"),
                       test_top1=m.get("test", {}).get("model", {}).get("top1"))
        return out

    def score(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Add scores to option rows (one or more rewards, keyed by ``offer_id``).

        Adds ``score`` (pick model) and, with a win model, ``raw_win`` and
        ``win``. With reliability settings, ``win`` is the adjusted chance and
        ``pick_chance``, ``low_confidence`` and ``tied`` (within the tie margin
        of the reward's best, when two or more are) are added too.

        ``suggested`` marks each reward's recommended option and
        ``suggestion_source`` says how it was chosen: the highest win chance
        (``win_model``); among options tied on win chance, the one players take
        most (``win_tie_popular``), since win chances can't separate them but
        players' picks often reflect synergies; or, without a win model, what
        players take most (``pick_model``).
        """
        rows = rows.copy()
        rows["score"] = predict(*self.pick, rows)
        if self.win and self.win[1].get("pick_features"):  # win model uses the pick model's view
            rows = add_pick_features(rows.assign(pick_score=rows["score"]))
        rows["raw_win"] = predict(*self.win, rows) if self.win else np.nan
        rows["win"] = rows["raw_win"]
        rows["pick_chance"] = np.nan
        rows["low_confidence"] = None
        rows["tied"] = False
        rel = self.reliability
        if rel:
            ids = rows["offer_id"].to_numpy()
            rows["pick_chance"] = pick_probabilities(ids, rows["score"], rel["temperature"])
            rows["win"], _ = adjust_win_chances(ids, rows["raw_win"], rows["pick_chance"],
                                                rel["shrink_k"])
            rows["low_confidence"] = rows["pick_chance"] < rel["low_confidence_below"]
            close = pd.Series(near_best(ids, rows["win"], rel["tie_margin"]), index=rows.index)
            rows["tied"] = close & (close.groupby(ids).transform("sum") > 1)
        rows["known"] = rows["is_skip"].astype(bool) | rows["candidate"].isin(self.known_cards)
        return _suggest(rows, has_win=self.win is not None)


    def rank_reward(self, state: Dict[str, Any], deck: List[Tuple[str, bool]],
                    relics: List[str], offered: List[Tuple[str, bool]]) -> pd.DataFrame:
        """Score one live card reward.

        ``state``: character, ascension, act, floor, room_type, hp, max_hp,
        gold. ``deck`` and ``offered`` are ``(card id, upgraded)`` pairs,
        ``relics`` relic ids (all normalized, see :func:`normalize_card`).
        Returns the scored options, the suggestion first and then by win chance.
        """
        base = dict(state, deck_upgraded=sum(up for _, up in deck), **relic_fields(relics))
        candidates = [{"id": cid, "upgraded": up} for cid, up in offered]
        rows = pd.DataFrame(reward_rows(base, candidates, Counter(cid for cid, _ in deck),
                                        self.meta)).assign(offer_id=0)
        rows = self.score(rows)
        return rows.sort_values(["suggested", "win" if self.win else "score"], ascending=False)


def normalize_id(name: str, prefix: str) -> str:
    """'Burning Blood' / 'RELIC.BURNING_BLOOD' -> 'BURNING_BLOOD' (prefix='relic')."""
    name = re.sub(rf"^{prefix}\.", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(r"[\s']+", "_", name).upper()


def normalize_card(name: str) -> Tuple[str, bool]:
    """'Shrug It Off+' / 'CARD.SHRUG_IT_OFF' -> ('SHRUG_IT_OFF', upgraded?)."""
    name = name.strip()
    return normalize_id(name.rstrip("+").strip(), "card"), name.endswith("+")


def _suggest(rows: pd.DataFrame, has_win: bool) -> pd.DataFrame:
    """Mark each reward's recommended option (see :meth:`Advisor.score`)."""
    ids = rows["offer_id"].to_numpy()
    if has_win:
        tie = rows["tied"].astype(bool).groupby(ids).transform("any").to_numpy()
        # in a tie, rank the tied options by the pick model; otherwise by win chance
        key = np.where(tie, np.where(rows["tied"].astype(bool), rows["score"], -np.inf),
                       rows["win"])
        source = np.where(tie, "win_tie_popular", "win_model")
    else:
        key, source = rows["score"].to_numpy(), np.full(len(rows), "pick_model")
    if "known" in rows:  # never suggest a card the models haven't seen (often a typo)
        key = np.where(rows["known"].to_numpy(dtype=bool), key, -np.inf)
    best = pd.Series(key, index=rows.index).groupby(ids).idxmax()
    return rows.assign(suggested=rows.index.isin(best.to_numpy()), suggestion_source=source)
