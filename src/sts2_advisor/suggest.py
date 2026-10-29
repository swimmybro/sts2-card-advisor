"""What should I pick? Rank a card reward using your run in progress.

Reads your deck, relics, HP, gold, character, ascension, act, floor and (on
an open reward screen) the kind of fight from the game's ``current_run.save``;
you type the offered cards ('+' = upgraded), which the game doesn't save:

    python -m sts2_advisor.suggest Snakebite Backflip Accuracy+
    python -m sts2_advisor.suggest "Shrug It Off" Anger Headbutt

``--elite`` / ``--boss`` are only needed if the save doesn't say (e.g. when
asking about a reward you've already left).

At the start of an act, ``--ancient`` asks about the ancient's relics instead:

    python -m sts2_advisor.suggest --ancient Kaleidoscope "Arcane Scroll" "Dowsing Rod"

The save is found next to the run history (see ``watch.py``), or pass
``--save`` / set ``$STS2_SAVE``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
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


def read_save(path: Path, player: Optional[int] = None) -> Dict[str, Any]:
    """The run state the advisor needs, from ``current_run.save``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    players = data["players"]
    if len(players) > 1 and player is None:
        names = ", ".join(f"{i}: {_strip(p.get('character_id'))}" for i, p in enumerate(players))
        raise ValueError(f"co-op save with {len(players)} players ({names}); pass --player N")
    p = players[player or 0]
    pending = data.get("pre_finished_room")
    history = data.get("map_point_history", [])
    return {
        "character": _strip(p.get("character_id")),
        "ascension": data.get("ascension", 0),
        "act": data.get("current_act_index", 0) + 1,
        # At a card reward the fight you just won is already in the history
        # (checked against a finished run: the save at the floor-2 reward had 2 points).
        "floor": sum(len(act) for act in history),
        "room_type": (pending or {}).get("room_type") or (pending or {}).get("map_point_type"),
        "hp": p.get("current_hp"), "max_hp": p.get("max_hp"), "gold": p.get("gold", 0),
        "deck": [(_strip(c["id"]), (c.get("current_upgrade_level") or 0) > 0)
                 for c in p.get("deck", [])],
        "relics": [_strip(r["id"]) for r in p.get("relics", [])],
        # A fight's reward screen is open (the save doesn't list its cards, though).
        "reward_open": bool(pending) and (pending.get("room_type") in ("monster", "elite", "boss")),
        # An act opens with an ancient: still in the act's first room.
        "at_ancient": not history or len(history[-1]) <= 1,
        "run": data.get("start_time"),
        "saved_seconds_ago": time.time() - data.get("save_time", os.path.getmtime(path)),
    }


def _ago(seconds: float) -> str:
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            n = int(seconds // size)
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    return "just now"


def advise(advisor: Advisor, state: Dict[str, Any], offered: List[Tuple[str, bool]],
           room_type: Optional[str] = None, floor: Optional[int] = None) -> str:
    """The recommendation and every option's numbers, as text."""
    return advise_with_choice(advisor, state, offered, room_type, floor)[0]


def advise_with_choice(advisor: Advisor, state: Dict[str, Any], offered: List[Tuple[str, bool]],
                       room_type: Optional[str] = None, floor: Optional[int] = None
                       ) -> Tuple[str, str]:
    """Like :func:`advise`, plus the suggested option's label (e.g. 'INFLAME+')."""
    s = {"character": state["character"], "ascension": state["ascension"],
         "act": state["act"], "floor": floor or state["floor"],
         # a --elite/--boss flag wins, then the save (open reward screen), then a normal fight
         "room_type": room_type or state.get("room_type") or "monster",
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
    return "\n".join(lines), label[top.name]


def advise_ancient(advisor, state: Dict[str, Any], options: List[str]) -> Tuple[str, str]:
    """Advice on an ancient's relics at the start of an act, as text, plus
    the suggested option."""
    rows = advisor.rank(state, state["deck"], state["relics"], options)
    top = rows.iloc[0]
    ancient = (top["ancient"] or "an ancient").title()
    lines = [f"{state['character'].title()} A{state['ascension']}, act {state['act']}: "
             f"{ancient}'s choice, HP {state['hp']}/{state['max_hp']}, {state['gold']} gold, "
             f"{len(state['deck'])}-card deck, {len(state['relics'])} relics", ""]
    if top["suggestion_source"] == "win_tie_popular":
        tied = list(rows.loc[rows["tied"], "option"])
        lines.append(f"  Take: {top['option']}  (close call: {', '.join(tied)} are within "
                     f"{advisor.reliability['tie_margin']:.0%} on win chance; players take "
                     f"{top['option']} most)")
    else:
        others = rows.iloc[1:][rows["known"].iloc[1:]]
        margin = top["win"] - others["win"].max() if len(others) else 0.0
        strength = "clear call" if margin >= 0.02 else "slight edge"
        lines.append(f"  Take: {top['option']}  ({strength}: +{margin * 100:.1f} pts of win "
                     "chance over the next option)")
    lines += ["", f"  {'option':<20} {'win chance':>10} {'players take':>13}"]
    for r in rows.itertuples():
        win = f"{r.win:.1%}" if r.known else "-"
        players = f"{r.pick_chance:.0%}" if r.known else "-"
        note = ("unknown relic (typo?)" if not r.known
                else "rarely taken here: less reliable" if r.low_confidence else "")
        lines.append(f"  {r.option:<20} {win:>10} {players:>13}  {note}".rstrip())
    return "\n".join(lines), top["option"]


def load_ancient_advisor():
    """The start-of-act advisor, or None if its models aren't trained."""
    from sts2_advisor.ancient import AncientAdvisor
    try:
        return AncientAdvisor()
    except RuntimeError:
        return None


SCREEN_HELP = ("No card names found on screen. Is the card choice open (not the list of "
               "rewards)? If this keeps happening, allow Screen Recording for your terminal "
               "app in System Settings > Privacy & Security, then restart the terminal.")


def _load_state(save: Path, player: Optional[int]) -> Dict[str, Any]:
    if not save.exists():
        raise SystemExit(f"No run in progress: {save} doesn't exist.")
    try:
        state = read_save(save, player)
    except ValueError as e:
        raise SystemExit(str(e))
    if state["character"] not in OFFICIAL_CHARACTERS:
        raise SystemExit(f"The advisor only knows the official characters, not {state['character']}.")
    return state


def _on_screen(read, image: Optional[Path], display: Optional[int]):
    """``read(boxes, PIL image)`` on ``image``, or on a fresh screenshot."""
    from PIL import Image

    from sts2_advisor import screen

    def go(path: Path):
        with Image.open(path) as img:
            return read(screen.read_text(path), img)
    if image is not None:
        return go(image)
    with tempfile.TemporaryDirectory() as tmp:
        return go(screen.capture(Path(tmp) / "screen.png", display))


def cards_on_screen(character: str, image: Optional[Path] = None,
                    display: Optional[int] = None) -> List[Tuple[str, bool]]:
    """The offered cards read from ``image``, or from a fresh screenshot."""
    from sts2_advisor import screen
    pool = screen.card_pool(character)
    return _on_screen(lambda boxes, img: screen.find_offered(boxes, pool, image=img),
                      image, display)


def options_on_screen(known: List[str], image: Optional[Path] = None,
                      display: Optional[int] = None) -> List[str]:
    """An ancient's relics read from ``image``, or from a fresh screenshot."""
    from sts2_advisor import screen
    return _on_screen(lambda boxes, img: screen.find_options(boxes, known), image, display)


def auto(advisor: Advisor, save: Path, player: Optional[int] = None,
         interval: float = 1.5, display: Optional[int] = None,
         max_checks: Optional[int] = None, ancients=None) -> None:
    """Watch for card rewards (and, with ``ancients``, the start-of-act
    choices) and advise on each one, reading the options off the screen.
    ``max_checks`` stops after that many checks (for tests)."""
    from sts2_advisor import screen
    what = "card rewards" + (" and start-of-act choices" if ancients else "")
    print(f"Watching for {what} (Ctrl-C to stop). Open the card choice to get advice.")
    done, last, tries, hinted, checks = set(), None, 0, False, 0
    while max_checks is None or checks < max_checks:
        checks += 1
        try:
            state = read_save(save, player)
        except (OSError, ValueError, KeyError):  # mid-write, or no run in progress
            time.sleep(interval)
            continue
        if state["reward_open"]:
            offered = cards_on_screen(state["character"], display=display)
        elif ancients is not None and state["at_ancient"]:
            offered = options_on_screen(ancients.options, display=display)
        else:
            last, tries = None, 0
            time.sleep(interval)
            continue
        tries += 1
        # Advise once per choice, not per floor: one fight can give two card
        # rewards (e.g. from a relic), each with its own cards.
        key = (state["run"], state["floor"], tuple(offered))
        # the same options on two screenshots in a row: the screen has settled
        if len(offered) >= 2 and offered == last and key not in done:
            if state["reward_open"]:
                text, choice = advise_with_choice(advisor, state, offered)
            else:
                text, choice = advise_ancient(ancients, state, offered)
            print("\n" + text + "\n", flush=True)
            screen.notify("Card advisor", f"Take {choice}")
            done.add(key)
        # (an ancient's screen may be closed already, so no hint there)
        elif state["reward_open"] and tries == 20 and not hinted:
            print(SCREEN_HELP, flush=True)
            hinted = True
        last = offered
        time.sleep(interval)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cards", nargs="*", help="the offered cards; '+' for upgraded")
    room = ap.add_mutually_exclusive_group()
    room.add_argument("--elite", action="store_const", const="elite", dest="room")
    room.add_argument("--boss", action="store_const", const="boss", dest="room")
    room.add_argument("--ancient", action="store_const", const="ancient", dest="room",
                      help="the names are an ancient's relics at the start of an act")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--screen", action="store_true",
                      help="read the offered cards off the screen (macOS)")
    mode.add_argument("--image", type=Path, help="read the offered cards from a screenshot file")
    mode.add_argument("--auto", action="store_true",
                      help="keep watching and advise at every card reward (macOS)")
    ap.add_argument("--display", type=int, help="which display to read with several (1 = main)")
    ap.add_argument("--floor", type=int, help="override the floor read from the save")
    ap.add_argument("--save", type=Path, help="path to current_run.save")
    ap.add_argument("--player", type=int, help="which player in a co-op save (0, 1, ...)")
    args = ap.parse_args(argv)

    save = find_save(args.save)
    if args.auto:
        ancients = load_ancient_advisor()
        if ancients is None:
            print("(No start-of-act model trained: advising on card rewards only.)")
        try:
            auto(Advisor(), save, args.player, display=args.display, ancients=ancients)
        except KeyboardInterrupt:
            print("\nStopped.")
        return
    state = _load_state(save, args.player)
    if args.room == "ancient":
        return _main_ancient(args, state)
    offered = [normalize_card(c) for c in args.cards]
    if not offered and (args.screen or args.image):
        offered = cards_on_screen(state["character"], args.image, args.display)
        if not offered:
            raise SystemExit(SCREEN_HELP)
        print("Read from the screen: " + ", ".join(c + ("+" if up else "") for c, up in offered) + "\n")
    if not offered:
        raise SystemExit("Name the offered cards (the game doesn't save them), e.g.: "
                         "python -m sts2_advisor.suggest Snakebite Backflip Accuracy+  "
                         "- or use --screen to read them off the screen")
    if state["saved_seconds_ago"] > STALE_AFTER:
        print(f"Note: the save is from {_ago(state['saved_seconds_ago'])}; "
              "is this the run you're playing?\n")
    print(advise(Advisor(), state, offered, room_type=args.room, floor=args.floor))


def _main_ancient(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    from sts2_advisor.ancient import normalize_option
    ancients = load_ancient_advisor()
    if ancients is None:
        raise SystemExit("No start-of-act model trained: run `python -m sts2_advisor.ancient "
                         "build` then `python -m sts2_advisor.ancient train`.")
    options = [normalize_option(o) for o in args.cards]
    if not options and (args.screen or args.image):
        options = options_on_screen(ancients.options, args.image, args.display)
        if not options:
            raise SystemExit("No relic names found on screen. Is the ancient's choice open?")
        print("Read from the screen: " + ", ".join(options) + "\n")
    if not options:
        raise SystemExit("Name the ancient's relics, e.g.: --ancient Kaleidoscope "
                         "\"Arcane Scroll\" \"Dowsing Rod\"  - or add --screen")
    print(advise_ancient(ancients, state, options)[0])


if __name__ == "__main__":
    main()
