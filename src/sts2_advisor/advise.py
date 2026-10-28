"""Score the options at card rewards with the pick and win models.

Shared by the API (one live reward) and the run reviewer (every reward in a
finished run), so both produce identical numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sts2_advisor.features import DEFAULT_CARD_META, load_card_meta
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
        return _suggest(rows, has_win=self.win is not None)


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
    best = pd.Series(key, index=rows.index).groupby(ids).idxmax()
    return rows.assign(suggested=rows.index.isin(best.to_numpy()), suggestion_source=source)
