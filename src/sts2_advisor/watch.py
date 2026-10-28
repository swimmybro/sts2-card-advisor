"""Watch the game's run history: copy new runs and review your card picks.

Each finished run the game writes a ``<id>.run`` file to its history folder.
This copies new ones into ``data/raw`` and, for solo runs, compares every card
reward's pick with the pick model's and the win model's suggestions.

    python -m sts2_advisor.watch                  # keep watching (every 30 s)
    python -m sts2_advisor.watch --once           # copy new runs, review them, exit
    python -m sts2_advisor.watch --review latest  # review a run already in data/raw

The history folder (``SlayTheSpire2/steam/<steam-id>/profile*/saves/history``
in the game's data folder) is found automatically on macOS, and in the
expected places on Windows and Linux; otherwise set ``$STS2_HISTORY_DIR`` /
``--history``.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from sts2_advisor.advise import Advisor
from sts2_advisor.features import RunRejected, offer_rows
from sts2_advisor.parse import RunParseError, _strip, load_run

log = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")
HISTORY_GLOB = "steam/*/profile*/saves/history"


# ------------------------------------------------------------------------ sync

def game_data_dirs() -> List[Path]:
    """Where the game keeps its data on macOS, Windows and Linux.

    The macOS location is confirmed; the others follow the same layout
    (``SlayTheSpire2`` in the platform's per-user app-data folder).
    """
    home = Path.home()
    dirs = [home / "Library" / "Application Support" / "SlayTheSpire2"]  # macOS
    if os.environ.get("APPDATA"):  # Windows: C:\Users\<you>\AppData\Roaming
        dirs.append(Path(os.environ["APPDATA"]) / "SlayTheSpire2")
    dirs.append(home / "AppData" / "Roaming" / "SlayTheSpire2")
    xdg = os.environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")  # Linux
    dirs.append(Path(xdg) / "SlayTheSpire2")
    return list(dict.fromkeys(dirs))  # drop duplicates, keep order


def find_history_dir(explicit: Optional[Path] = None) -> Path:
    """``--history``, else ``$STS2_HISTORY_DIR``, else the most recently used
    profile in the game's data folder (see :func:`game_data_dirs`)."""
    if explicit:
        return Path(explicit)
    if os.environ.get("STS2_HISTORY_DIR"):
        return Path(os.environ["STS2_HISTORY_DIR"])
    found = sorted((p for d in game_data_dirs() for p in d.glob(HISTORY_GLOB)),
                   key=lambda p: p.stat().st_mtime)
    if not found:
        searched = "\n  ".join(str(d / HISTORY_GLOB) for d in game_data_dirs())
        raise FileNotFoundError(
            f"No run history found. Looked in:\n  {searched}\n"
            "Pass --history or set STS2_HISTORY_DIR to your 'saves/history' folder.")
    if len(found) > 1:
        log.info("several profiles found; using the most recent: %s", found[-1])
    return found[-1]


def sync(history_dir: Path, raw_dir: Path) -> List[Path]:
    """Copy ``.run`` files not yet in ``raw_dir``; return the new copies, oldest first.

    Files that aren't valid run JSON yet (still being written) are skipped and
    picked up on a later call. Copies are written to a temporary name and
    renamed, so readers never see a half-written file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    new = []
    for src in sorted(history_dir.glob("*.run")):
        dst = raw_dir / src.name
        if dst.exists():
            continue
        try:
            load_run(src)
        except RunParseError as e:
            log.info("not ready yet, will retry: %s", e)
            continue
        tmp = dst.with_name(dst.name + ".partial")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        new.append(dst)
    return new


# ---------------------------------------------------------------------- review

@dataclass
class RunReview:
    run_id: str
    character: Optional[str] = None
    ascension: Optional[int] = None
    outcome: str = ""
    floors: int = 0
    rewards: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""  # why there's no per-reward review, if there isn't one


def _outcome(data: dict) -> str:
    if data.get("was_abandoned"):
        return "abandoned"
    if data["win"]:
        return "won"
    killer = _strip(data.get("killed_by_encounter"))
    return f"lost to {killer}" if killer else "lost"


def review_run(path: Path, advisor: Advisor) -> RunReview:
    """One row per card reward: your pick, each model's top option, win chances."""
    data = load_run(path)
    player = data["players"][0]
    review = RunReview(
        run_id=path.stem, character=_strip(player.get("character")),
        ascension=data.get("ascension"), outcome=_outcome(data),
        floors=sum(len(act) for act in data["map_point_history"]))
    if len(data["players"]) > 1:
        review.note = "co-op run: copied, but reviews cover solo runs only"
        return review
    try:
        rows = pd.DataFrame(offer_rows(data, path.stem, "mine", advisor.meta))
    except RunRejected as e:
        review.note = f"can't review this run ({e.reason})"
        return review
    if rows.empty:
        review.note = "no card rewards"
        return review

    scored = advisor.score(rows)
    has_win = advisor.win is not None
    out = []
    for offer_id, g in scored.groupby("offer_id", sort=False):
        yours = g[g["picked"]].iloc[0]
        pick_top = g.loc[g["score"].idxmax()]
        rec = {"floor": int(yours["floor"]), "act": int(yours["act"]),
               "room": yours["room_type"],
               "offered": [_label(r) for _, r in g[~g["is_skip"]].iterrows()],
               "yours": _label(yours), "pick_top": _label(pick_top),
               "agrees_pick": bool(yours["candidate"] == pick_top["candidate"])}
        if has_win:
            best = g[g["suggested"]].iloc[0]  # highest win chance, ties broken by popularity
            rec.update(
                win_top=_label(best), your_win=float(yours["win"]),
                best_win=float(best["win"]),
                best_low_confidence=bool(best["low_confidence"]),
                tie_broken=best["suggestion_source"] == "win_tie_popular",
                # a pick tied with the best counts as agreeing with the win model
                agrees_win=bool(yours["candidate"] == best["candidate"] or yours["tied"]),
                tied=[_label(r) for _, r in g[g["tied"]].iterrows()])
            rec["gap"] = 0.0 if rec["agrees_win"] else rec["best_win"] - rec["your_win"]
        out.append(rec)
    review.rewards = pd.DataFrame(out)
    return review


def _label(row: pd.Series) -> str:
    if row["is_skip"]:
        return "SKIP"
    return row["candidate"] + ("+" if row.get("cand_upgraded") else "")


def format_review(review: RunReview, n_worst: int = 3) -> str:
    head = (f"Run {review.run_id}: {review.character} A{review.ascension}, "
            f"{review.outcome} after {review.floors} floors")
    if review.note:
        return f"{head}\n  {review.note}"
    r = review.rewards
    has_win = "win_top" in r
    lines = [head, ""]
    for x in r.itertuples():
        line = f"  F{x.floor:<3} {x.room:<8} {', '.join(x.offered):<44} you: {x.yours:<16}"
        if has_win:
            best = x.win_top + (" (tie)" if x.tie_broken else "")
            flag = " (low confidence)" if x.best_low_confidence and not x.agrees_win else ""
            line += (f" best: {best:<16} win {x.your_win:5.1%} vs {x.best_win:5.1%}{flag}"
                     f"   popular: {x.pick_top}")
        else:
            line += f" popular: {x.pick_top}"
        lines.append(line.rstrip())
    n = len(r)
    lines.append("")
    if has_win:
        lines.append(f"  Took the best option (ties count) on {int(r['agrees_win'].sum())}/{n} rewards")
    lines.append(f"  Took the popular option on {int(r['agrees_pick'].sum())}/{n}")
    if has_win:
        worst = r[r["gap"] > 0].sort_values("gap", ascending=False).head(n_worst)
        if len(worst):
            lines.append("  Biggest win-chance differences from the win model's choice:")
            for x in worst.itertuples():
                flag = ", low confidence" if x.best_low_confidence else ""
                lines.append(f"    F{x.floor}: took {x.yours}, model preferred {x.win_top} "
                             f"({x.your_win:.1%} -> {x.best_win:.1%}{flag})")
        else:
            lines.append("  No reward where the win model clearly preferred another option.")
    return "\n".join(lines)


# ------------------------------------------------------------------------- CLI

def _resolve_run(arg: str, raw_dir: Path) -> Path:
    if arg == "latest":
        runs = sorted(raw_dir.glob("*.run"), key=lambda p: p.stem)
        if not runs:
            raise FileNotFoundError(f"no runs in {raw_dir}")
        return runs[-1]
    p = Path(arg)
    return p if p.exists() else raw_dir / f"{arg}.run"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history", type=Path, help="the game's run history folder")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--once", action="store_true", help="sync and review once, then exit")
    ap.add_argument("--interval", type=float, default=30, help="seconds between checks")
    ap.add_argument("--review", metavar="RUN",
                    help="review one run: 'latest', a run id, or a path")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    advisor = Advisor()

    if args.review:
        print(format_review(review_run(_resolve_run(args.review, args.raw_dir), advisor)))
        return

    history = find_history_dir(args.history)
    print(f"Watching {history} -> {args.raw_dir}")
    while True:
        for path in sync(history, args.raw_dir):
            print(f"\nNew run copied: {path.name}")
            print(format_review(review_run(path, advisor)))
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
