"""What should I pick? Rank a card reward using your run in progress.

Reads your deck, relics, HP, gold, character, ascension, act and floor from
the game's ``current_run.save``; you type the offered cards ('+' = upgraded):

    python -m sts2_advisor.suggest Snakebite Backflip Accuracy+
    python -m sts2_advisor.suggest "Shrug It Off" Anger Headbutt --elite

The save is found next to the run history (see ``watch.py``), or pass
``--save`` / set ``$STS2_SAVE``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sts2_advisor.advise import Advisor, normalize_card
from sts2_advisor.parse import OFFICIAL_CHARACTERS, _strip
from sts2_advisor.watch import find_history_dir

STALE_AFTER = 3 * 3600  # seconds; an older save is probably not the run you're in


def find_save(explicit: Optional[Path] = None) -> Path:
    if explicit:
        return Path(explicit)
    if os.environ.get("STS2_SAVE"):
        return Path(os.environ["STS2_SAVE"])
    return find_history_dir().parent / "current_run.save"


def _cards_in(obj: Any) -> List[Tuple[str, bool]]:
    """Card choices found anywhere under ``obj`` (best effort: the save's format
    for an open reward screen is undocumented)."""
    found: List[Tuple[str, bool]] = []
    if isinstance(obj, dict):
        card = obj.get("card") if isinstance(obj.get("card"), dict) else obj
        cid = card.get("id")
        if isinstance(cid, str) and cid.startswith("CARD.") and "was_picked" in obj:
            found.append((_strip(cid), (card.get("current_upgrade_level") or 0) > 0))
        for k, v in obj.items():
            if k != "deck":
                found += _cards_in(v)
    elif isinstance(obj, list):
        for v in obj:
            found += _cards_in(v)
    return found


def read_save(path: Path, player: Optional[int] = None) -> Dict[str, Any]:
    """The run state the advisor needs, from ``current_run.save``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    players = data["players"]
    if len(players) > 1 and player is None:
        names = ", ".join(f"{i}: {_strip(p.get('character_id'))}" for i, p in enumerate(players))
        raise ValueError(f"co-op save with {len(players)} players ({names}); pass --player N")
    p = players[player or 0]
    pending = data.get("pre_finished_room")
    return {
        "character": _strip(p.get("character_id")),
        "ascension": data.get("ascension", 0),
        "act": data.get("current_act_index", 0) + 1,
        # map points finished so far, plus the room you're in now
        "floor": sum(len(act) for act in data.get("map_point_history", [])) + 1,
        "room_type": (pending or {}).get("room_type") or (pending or {}).get("map_point_type"),
        "hp": p.get("current_hp"), "max_hp": p.get("max_hp"), "gold": p.get("gold", 0),
        "deck": [(_strip(c["id"]), (c.get("current_upgrade_level") or 0) > 0)
                 for c in p.get("deck", [])],
        "relics": [_strip(r["id"]) for r in p.get("relics", [])],
        "offered_in_save": _cards_in(pending) if pending else [],
        "saved_seconds_ago": time.time() - data.get("save_time", os.path.getmtime(path)),
    }


def _ago(seconds: float) -> str:
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            n = int(seconds // size)
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    return "just now"


def advise(advisor: Advisor, state: Dict[str, Any], offered: List[Tuple[str, bool]],
           room_type: str = "monster", floor: Optional[int] = None) -> str:
    """The recommendation and every option's numbers, as text."""
    s = {"character": state["character"], "ascension": state["ascension"],
         "act": state["act"], "floor": floor or state["floor"],
         "room_type": state.get("room_type") or room_type,
         "hp": state["hp"], "max_hp": state["max_hp"], "gold": state["gold"]}
    rows = advisor.rank_reward(s, state["deck"], state["relics"], offered)
    label = {r.Index: r.candidate + ("+" if r.cand_upgraded is True else "") for r in rows.itertuples()}
    top = rows.iloc[0]
    lines = [f"{s['character'].title()} A{s['ascension']}, act {s['act']} floor {s['floor']} "
             f"({s['room_type']}), HP {s['hp']}/{s['max_hp']}, {s['gold']} gold, "
             f"{len(state['deck'])}-card deck, {len(state['relics'])} relics", ""]
    if advisor.win is None:
        lines.append(f"  Take: {label[top.name]}  (what players take most; no win model loaded)")
    elif top["suggestion_source"] == "win_tie_popular":
        tied = [label[i] for i in rows.index[rows["tied"]]]
        lines.append(f"  Take: {label[top.name]}  (close call: {', '.join(tied)} are within "
                     f"{advisor.reliability['tie_margin']:.0%} on win chance; players take "
                     f"{label[top.name]} most)")
    else:
        others = rows.iloc[1:][rows["known"].iloc[1:]]  # unknown cards' numbers are meaningless
        margin = top["win"] - others["win"].max() if len(others) else 0.0
        strength = "clear call" if margin >= 0.02 else "slight edge"
        lines.append(f"  Take: {label[top.name]}  ({strength}: +{margin * 100:.1f} pts of win "
                     f"chance over the next option)")
    unknown = [label[i] for i in rows.index[~rows["known"]]]
    if unknown:
        lines.append(f"  Not considered: {', '.join(unknown)} (not a card the models know; check "
                     "the spelling)")
    lines += ["", f"  {'option':<20} {'win chance':>10} {'players take':>13}"]
    for r in rows.itertuples():
        win = "-" if not r.known or np.isnan(r.win) else f"{r.win:.1%}"
        players = "-" if not r.known or np.isnan(r.pick_chance) else f"{r.pick_chance:.0%}"
        notes = []
        if r.low_confidence is True and r.known:
            notes.append("rarely taken here: less reliable")
        if not r.known:
            notes.append("unknown card (typo?)")
        lines.append(f"  {label[r.Index]:<20} {win:>10} {players:>13}  {'; '.join(notes)}".rstrip())
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cards", nargs="*", help="the offered cards; '+' for upgraded")
    room = ap.add_mutually_exclusive_group()
    room.add_argument("--elite", action="store_const", const="elite", dest="room")
    room.add_argument("--boss", action="store_const", const="boss", dest="room")
    ap.add_argument("--floor", type=int, help="override the floor read from the save")
    ap.add_argument("--save", type=Path, help="path to current_run.save")
    ap.add_argument("--player", type=int, help="which player in a co-op save (0, 1, ...)")
    args = ap.parse_args(argv)

    save = find_save(args.save)
    if not save.exists():
        raise SystemExit(f"No run in progress: {save} doesn't exist.")
    try:
        state = read_save(save, args.player)
    except ValueError as e:
        raise SystemExit(str(e))
    if state["character"] not in OFFICIAL_CHARACTERS:
        raise SystemExit(f"The advisor only knows the official characters, not {state['character']}.")
    offered = [normalize_card(c) for c in args.cards] or state["offered_in_save"]
    if not offered:
        raise SystemExit("Name the offered cards, e.g.: python -m sts2_advisor.suggest "
                         "Snakebite Backflip Accuracy+  (the save doesn't list them)")
    if state["saved_seconds_ago"] > STALE_AFTER:
        print(f"Note: the save is from {_ago(state['saved_seconds_ago'])}; "
              "is this the run you're playing?\n")
    print(advise(Advisor(), state, offered, room_type=args.room or "monster", floor=args.floor))


if __name__ == "__main__":
    main()
