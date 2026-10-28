"""Parse Slay the Spire 2 ``.run`` files into flat tables.

Produces two tables:

* ``runs``         – one row per (run, player) with outcome and final-deck info.
* ``card_choices`` – one row per card offered in a card reward / shop / event,
                     with whether it was picked.

Usage::

    python -m sts2_advisor.parse data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)

RUN_COLUMNS = [
    "run_id", "player_id", "character", "n_players", "ascension", "win",
    "was_abandoned", "game_mode", "build_id", "schema_version", "seed",
    "start_time", "run_time", "acts_reached", "floors_reached",
    "killed_by_encounter", "deck_size", "n_relics", "deck",
]
CHOICE_COLUMNS = [
    "run_id", "player_id", "character", "ascension", "win", "act", "floor",
    "map_point_type", "card_id", "upgrade_level", "was_picked", "n_offered",
    "offer_skipped",
]


class RunParseError(ValueError):
    """Raised when a run file is not a usable run record."""


def load_run(path: Path) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RunParseError(f"{path.name}: {e}") from e
    if not isinstance(data, dict):
        raise RunParseError(f"{path.name}: top level is not an object")
    for key in ("players", "map_point_history", "win"):
        if key not in data:
            raise RunParseError(f"{path.name}: missing '{key}'")
    return data


def _strip(identifier: Optional[str]) -> Optional[str]:
    """'CARD.MOLTEN_FIST' -> 'MOLTEN_FIST'; 'NONE.NONE' -> None."""
    if not identifier or identifier == "NONE.NONE":
        return None
    return identifier.split(".", 1)[-1]


def parse_run(data: Dict[str, Any], run_id: str) -> Tuple[List[dict], List[dict]]:
    """Flatten one run into (run rows, card-choice rows)."""
    players = data["players"]
    history = data["map_point_history"]
    character = {p["id"]: _strip(p.get("character")) for p in players}
    ascension = data.get("ascension")
    win = bool(data["win"])

    choices: List[dict] = []
    floor = 0
    for act_idx, act in enumerate(history, start=1):
        for point in act:
            floor += 1
            for stats in point.get("player_stats", []):
                offer = stats.get("card_choices") or []
                if not offer:
                    continue
                pid = stats.get("player_id")
                skipped = not any(c.get("was_picked") for c in offer)
                for c in offer:
                    card = c.get("card", {})
                    choices.append({
                        "run_id": run_id,
                        "player_id": pid,
                        "character": character.get(pid),
                        "ascension": ascension,
                        "win": win,
                        "act": act_idx,
                        "floor": floor,
                        "map_point_type": point.get("map_point_type"),
                        "card_id": _strip(card.get("id")),
                        "upgrade_level": card.get("current_upgrade_level", 0),
                        "was_picked": bool(c.get("was_picked")),
                        "n_offered": len(offer),
                        "offer_skipped": skipped,
                    })

    runs = []
    for p in players:
        deck = [_strip(c.get("id")) for c in p.get("deck", [])]
        runs.append({
            "run_id": run_id,
            "player_id": p["id"],
            "character": character[p["id"]],
            "n_players": len(players),
            "ascension": ascension,
            "win": win,
            "was_abandoned": bool(data.get("was_abandoned", False)),
            "game_mode": data.get("game_mode"),
            "build_id": data.get("build_id"),
            "schema_version": data.get("schema_version"),
            "seed": data.get("seed"),
            "start_time": data.get("start_time"),
            "run_time": data.get("run_time"),
            "acts_reached": len(history),
            "floors_reached": floor,
            "killed_by_encounter": _strip(data.get("killed_by_encounter")),
            "deck_size": len(deck),
            "n_relics": len(p.get("relics", [])),
            "deck": "|".join(d for d in deck if d),
        })
    return runs, choices


def parse_dir(
    raw_dir: Path, include_multiplayer: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], int]:
    """Parse every ``*.run`` in ``raw_dir``. Bad files are skipped and reported.

    Multiplayer runs are excluded unless ``include_multiplayer`` is set; the
    number excluded is returned as the last element.
    """
    run_rows: List[dict] = []
    choice_rows: List[dict] = []
    errors: List[str] = []
    n_multiplayer = 0
    for path in sorted(Path(raw_dir).glob("*.run")):
        try:
            data = load_run(path)
            if len(data["players"]) > 1 and not include_multiplayer:
                n_multiplayer += 1
                continue
            runs, choices = parse_run(data, run_id=path.stem)
        except (RunParseError, KeyError, TypeError) as e:
            log.warning("skipping %s: %s", path.name, e)
            errors.append(f"{path.name}: {e}")
            continue
        run_rows.extend(runs)
        choice_rows.extend(choices)
    runs_df = pd.DataFrame(run_rows, columns=RUN_COLUMNS)
    choices_df = pd.DataFrame(choice_rows, columns=CHOICE_COLUMNS)
    return runs_df, choices_df, errors, n_multiplayer


def summarize(runs: pd.DataFrame, choices: pd.DataFrame, errors: Iterable[str],
              n_multiplayer: int = 0) -> str:
    errors = list(errors)
    n_runs = runs["run_id"].nunique()
    per_run = runs.drop_duplicates("run_id")
    lines = [
        f"Files parsed:      {n_runs}  (skipped: {len(errors)}, "
        f"multiplayer excluded: {n_multiplayer})",
        f"Player-run rows:   {len(runs)}  (multiplayer runs: {(per_run['n_players'] > 1).sum()})",
        f"Wins:              {int(per_run['win'].sum())}/{n_runs} "
        f"({per_run['win'].mean():.1%})" if n_runs else "Wins:              n/a",
        f"Abandoned:         {int(per_run['was_abandoned'].sum())}",
        f"Builds:            {dict(Counter(per_run['build_id']).most_common())}",
        "",
        "By character (player-runs, win rate, mean ascension):",
    ]
    by_char = (runs.groupby("character")
               .agg(runs=("run_id", "size"), win_rate=("win", "mean"),
                    asc=("ascension", "mean"))
               .sort_values("runs", ascending=False))
    for char, r in by_char.iterrows():
        lines.append(f"  {char:<12} {int(r['runs']):>4}  {r['win_rate']:6.1%}  A{r['asc']:.1f}")

    lines += ["", f"Card-choice rows:  {len(choices)}"]
    if len(choices):
        offers = choices.drop_duplicates(["run_id", "player_id", "floor"])
        picked = choices[choices["was_picked"]]
        lines += [
            f"Offers:            {len(offers)}  (skipped: {offers['offer_skipped'].mean():.1%})",
            f"Distinct cards:    {choices['card_id'].nunique()}",
            "",
            "Most-picked cards (picked / offered):",
        ]
        offered = choices["card_id"].value_counts()
        for card, n in picked["card_id"].value_counts().head(10).items():
            lines.append(f"  {card:<24} {n:>4} / {offered[card]:<4} ({n / offered[card]:.0%})")
    if errors:
        lines += ["", "Skipped files:"] + [f"  {e}" for e in errors[:10]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw_dir", type=Path, nargs="?", default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--include-multiplayer", action="store_true",
                    help="keep co-op runs (excluded by default)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    runs, choices, errors, n_multiplayer = parse_dir(args.raw_dir, args.include_multiplayer)
    args.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out / "runs.csv", index=False)
    choices.to_csv(args.out / "card_choices.csv", index=False)
    print(summarize(runs, choices, errors, n_multiplayer))
    print(f"\nWrote {args.out / 'runs.csv'} and {args.out / 'card_choices.csv'}")


if __name__ == "__main__":
    main()
