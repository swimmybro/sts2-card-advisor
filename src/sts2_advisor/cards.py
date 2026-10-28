"""Card metadata: fetch the Spire Codex catalog and derive card traits.

Writes ``configs/cards.csv`` with each card's type, rarity, cost and a
``traits`` column (``|``-separated) used for deck-synergy features.

Refresh after a game patch::

    python -m sts2_advisor.cards --out configs/cards.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

CATALOG_URL = "https://spire-codex.com/api/cards"

KEYWORD_TRAITS = ["Exhaust", "Ethereal", "Retain", "Innate", "Sly"]
TAG_TRAITS = ["Strike", "Defend", "OstyAttack", "Shiv", "Minion"]
POWER_TRAITS = ["Vulnerable", "Weak", "Strength", "Poison", "Dexterity", "Focus",
                "Doom", "Plating"]
SPAWN_TRAITS = {"SHIV": "Shiv", "SOUL": "Soul", "SOVEREIGN_BLADE": "SovereignBlade"}
ROLE_TRAITS = ["Block", "Draw", "Energy", "HpLoss", "AoE", "XCost"]

TRAITS: List[str] = (KEYWORD_TRAITS + TAG_TRAITS + POWER_TRAITS
                     + sorted(set(SPAWN_TRAITS.values()) - set(TAG_TRAITS)) + ROLE_TRAITS)


def card_traits(card: Dict[str, Any]) -> List[str]:
    """Traits of one catalog entry, restricted to :data:`TRAITS`, in that order."""
    found = set(card.get("keywords") or []) | set(card.get("tags") or [])
    found |= {p.get("power") for p in card.get("powers_applied") or []}
    found |= {SPAWN_TRAITS[s] for s in card.get("spawns_cards") or [] if s in SPAWN_TRAITS}
    roles = {
        "Block": (card.get("block") or 0) > 0,
        "Draw": (card.get("cards_draw") or 0) > 0,
        "Energy": (card.get("energy_gain") or 0) > 0,
        "HpLoss": (card.get("hp_loss") or 0) > 0,
        "AoE": card.get("target") == "AllEnemies",
        "XCost": bool(card.get("is_x_cost")),
    }
    found |= {r for r, on in roles.items() if on}
    return [t for t in TRAITS if t in found]


DEFENSIVE_POWERS = {"Weak", "Intangible", "Plating", "Buffer"}


def is_defensive(card: Dict[str, Any]) -> bool:
    """Reduces incoming damage: gives block, or applies Weak / Intangible / Plating / Buffer."""
    return (card.get("block") or 0) > 0 or any(
        p.get("power") in DEFENSIVE_POWERS for p in card.get("powers_applied") or [])


def write_cards_csv(catalog: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "rarity", "color", "cost", "is_x_cost", "traits",
                    "draw", "defensive"])
        for c in sorted(catalog, key=lambda c: c["id"]):
            w.writerow([c["id"], c["type"], c["rarity"], c["color"], c["cost"],
                        bool(c.get("is_x_cost")), "|".join(card_traits(c)),
                        c.get("cards_draw") or 0, is_defensive(c)])


def fetch_catalog(url: str = CATALOG_URL) -> List[Dict[str, Any]]:
    req = urllib.request.Request(url, headers={"User-Agent": "sts2-card-advisor"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("configs/cards.csv"))
    ap.add_argument("--catalog", type=Path, help="use a saved catalog JSON instead of fetching")
    args = ap.parse_args(argv)
    catalog = json.loads(args.catalog.read_text()) if args.catalog else fetch_catalog()
    write_cards_csv(catalog, args.out)
    print(f"Wrote {len(catalog)} cards to {args.out}")


if __name__ == "__main__":
    main()
