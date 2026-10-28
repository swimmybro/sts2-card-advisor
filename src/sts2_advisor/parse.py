"""Parse Slay the Spire 2 runs into flat tables.

Inputs (files, or directories containing them):

* ``*.run``      – raw run files copied from the game (``source="mine"``).
* ``*.jsonl.gz`` – Spire Codex bulk exports, one raw run JSON per line
                   (``source="codex"``).

Produces two Parquet tables:

* ``runs``         – one row per (run, player) with outcome and final-deck info.
* ``card_choices`` – one row per card offered in a card reward / shop / event,
                     with whether it was picked.

Runs are de-duplicated on ``(seed, start_time)``; the first copy seen wins, so
list your own runs before external exports. Runs with modded characters or
cards (anything outside ``configs/official_cards.txt``) are dropped by default.

Usage::

    python -m sts2_advisor.parse data/raw data/external --out data/processed
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

RUN_SCHEMA = pa.schema([
    ("run_id", pa.string()), ("source", pa.string()),
    ("player_token", pa.string()), ("player_id", pa.int64()),
    ("character", pa.string()), ("n_players", pa.int64()),
    ("ascension", pa.int64()), ("win", pa.bool_()),
    ("was_abandoned", pa.bool_()), ("game_mode", pa.string()),
    ("build_id", pa.string()), ("schema_version", pa.int64()),
    ("seed", pa.string()), ("start_time", pa.int64()),
    ("run_time", pa.int64()), ("acts_reached", pa.int64()),
    ("floors_reached", pa.int64()), ("killed_by_encounter", pa.string()),
    ("deck_size", pa.int64()), ("n_relics", pa.int64()), ("deck", pa.string()),
])
CHOICE_SCHEMA = pa.schema([
    ("run_id", pa.string()), ("source", pa.string()),
    ("player_id", pa.int64()), ("character", pa.string()),
    ("ascension", pa.int64()), ("win", pa.bool_()), ("act", pa.int64()),
    ("floor", pa.int64()), ("map_point_type", pa.string()),
    ("card_id", pa.string()), ("upgrade_level", pa.int64()),
    ("was_picked", pa.bool_()), ("n_offered", pa.int64()),
    ("offer_skipped", pa.bool_()),
])
RUN_COLUMNS = RUN_SCHEMA.names
CHOICE_COLUMNS = CHOICE_SCHEMA.names
MAX_ERRORS_KEPT = 20
OFFICIAL_CHARACTERS = {"IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER"}
DEFAULT_CARD_ALLOWLIST = Path("configs/official_cards.txt")


class RunParseError(ValueError):
    """Raised when a run file is not a usable run record."""


@dataclass
class ParseStats:
    n_errors: int = 0
    n_multiplayer: int = 0
    n_duplicates: int = 0
    n_modded: int = 0
    errors: List[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        log.warning("skipping %s", msg)
        self.n_errors += 1
        if len(self.errors) < MAX_ERRORS_KEPT:
            self.errors.append(msg)


def _validate(data: Any, name: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise RunParseError(f"{name}: top level is not an object")
    for key in ("players", "map_point_history", "win"):
        if key not in data:
            raise RunParseError(f"{name}: missing '{key}'")
    return data


def load_run(path: Path) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RunParseError(f"{path.name}: {e}") from e
    return _validate(data, path.name)


def load_card_allowlist(path: Path) -> Set[str]:
    """Read card IDs (one per line, ``#`` comments allowed)."""
    with open(path, encoding="utf-8") as f:
        return {s for s in (line.split("#", 1)[0].strip() for line in f) if s}


def _is_modded(runs: List[dict], choices: List[dict], allowed_cards: Set[str]) -> bool:
    if any(r["character"] not in OFFICIAL_CHARACTERS for r in runs):
        return True
    cards = {c["card_id"] for c in choices}
    for r in runs:
        cards.update(r["deck"].split("|") if r["deck"] else [])
    cards.discard(None)
    return not cards <= allowed_cards


def _input_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.glob("*.run")) + sorted(p.glob("*.jsonl.gz"))
        else:
            files.append(p)
    return files


def iter_raw_runs(paths: Iterable[Path], stats: ParseStats
                  ) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """Yield ``(run_id, source, raw_json)`` for every readable run in ``paths``."""
    for path in _input_files(paths):
        if path.name.endswith(".jsonl.gz"):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, start=1):
                        name = f"{path.name}:{lineno}"
                        try:
                            data = _validate(json.loads(line), name)
                        except (json.JSONDecodeError, RunParseError) as e:
                            stats.error(f"{name}: {e}")
                            continue
                        yield str(data.get("run_hash") or name), "codex", data
            except (OSError, EOFError, UnicodeDecodeError) as e:
                stats.error(f"{path.name}: {e}")
        else:
            try:
                yield path.stem, "mine", load_run(path)
            except RunParseError as e:
                stats.error(str(e))


def _strip(identifier: Optional[str]) -> Optional[str]:
    """'CARD.MOLTEN_FIST' -> 'MOLTEN_FIST'; 'NONE.NONE' -> None."""
    if not identifier or identifier == "NONE.NONE":
        return None
    return identifier.split(".", 1)[-1]


def parse_run(data: Dict[str, Any], run_id: str, source: str = "mine"
              ) -> Tuple[List[dict], List[dict]]:
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
                        "source": source,
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
            "source": source,
            "player_token": data.get("player_token"),
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


def iter_parsed(paths: Iterable[Path], stats: ParseStats,
                include_multiplayer: bool = False,
                allowed_cards: Optional[Set[str]] = None,
                ) -> Iterator[Tuple[List[dict], List[dict]]]:
    """Parse runs one at a time, dropping multiplayer runs and duplicates.

    If ``allowed_cards`` is given, runs with an unofficial character or any
    card outside the set are dropped as modded.
    """
    for _, _, _, runs, choices in iter_kept_runs(
            paths, stats, include_multiplayer, allowed_cards):
        yield runs, choices


def iter_kept_runs(paths: Iterable[Path], stats: ParseStats,
                   include_multiplayer: bool = False,
                   allowed_cards: Optional[Set[str]] = None,
                   ) -> Iterator[Tuple[str, str, Dict[str, Any], List[dict], List[dict]]]:
    """Like :func:`iter_parsed`, but also yields ``run_id``, ``source`` and the raw JSON."""
    seen = set()
    for run_id, source, data in iter_raw_runs(paths, stats):
        if len(data["players"]) > 1 and not include_multiplayer:
            stats.n_multiplayer += 1
            continue
        key = (data.get("seed"), data.get("start_time"))
        if key in seen:
            stats.n_duplicates += 1
            continue
        seen.add(key)
        try:
            runs, choices = parse_run(data, run_id, source)
        except (KeyError, TypeError, AttributeError) as e:
            stats.error(f"{run_id}: {e!r}")
            continue
        if allowed_cards is not None and _is_modded(runs, choices, allowed_cards):
            stats.n_modded += 1
            continue
        yield run_id, source, data, runs, choices


def parse_dir(paths: Iterable[Path], include_multiplayer: bool = False,
              allowed_cards: Optional[Set[str]] = None,
              ) -> Tuple[pd.DataFrame, pd.DataFrame, ParseStats]:
    """Parse everything into in-memory DataFrames (fine for small inputs)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    stats = ParseStats()
    run_rows: List[dict] = []
    choice_rows: List[dict] = []
    for runs, choices in iter_parsed(paths, stats, include_multiplayer, allowed_cards):
        run_rows.extend(runs)
        choice_rows.extend(choices)
    return (pd.DataFrame(run_rows, columns=RUN_COLUMNS),
            pd.DataFrame(choice_rows, columns=CHOICE_COLUMNS), stats)


def write_parquet(paths: Iterable[Path], out: Path,
                  include_multiplayer: bool = False,
                  allowed_cards: Optional[Set[str]] = None,
                  chunk_runs: int = 5000) -> ParseStats:
    """Stream runs into ``out/runs.parquet`` and ``out/card_choices.parquet``."""
    out.mkdir(parents=True, exist_ok=True)
    stats = ParseStats()
    run_buf: List[dict] = []
    choice_buf: List[dict] = []
    n_buffered = 0
    with pq.ParquetWriter(out / "runs.parquet", RUN_SCHEMA) as run_w, \
            pq.ParquetWriter(out / "card_choices.parquet", CHOICE_SCHEMA) as choice_w:

        def flush() -> None:
            run_w.write_table(pa.Table.from_pylist(run_buf, schema=RUN_SCHEMA))
            choice_w.write_table(pa.Table.from_pylist(choice_buf, schema=CHOICE_SCHEMA))
            run_buf.clear()
            choice_buf.clear()

        for runs, choices in iter_parsed(paths, stats, include_multiplayer, allowed_cards):
            run_buf.extend(runs)
            choice_buf.extend(choices)
            n_buffered += 1
            if n_buffered % chunk_runs == 0:
                flush()
                log.info("parsed %d runs", n_buffered)
        flush()
    return stats


def summarize(runs: pd.DataFrame, choices: pd.DataFrame, stats: ParseStats) -> str:
    n_runs = runs["run_id"].nunique()
    per_run = runs.drop_duplicates("run_id")
    by_source = per_run.groupby("source")["win"].agg(["size", "mean"])
    lines = [
        f"Runs parsed:       {n_runs}  ("
        + ", ".join(f"{s}: {int(r['size'])}" for s, r in by_source.iterrows()) + ")",
        f"Skipped:           {stats.n_errors} unreadable, "
        f"{stats.n_multiplayer} multiplayer, {stats.n_duplicates} duplicates, "
        f"{stats.n_modded} modded",
        f"Player-run rows:   {len(runs)}",
        f"Wins:              {int(per_run['win'].sum())}/{n_runs} "
        f"({per_run['win'].mean():.1%})" if n_runs else "Wins:              n/a",
    ]
    if len(by_source) > 1:
        lines.append("Win rate by source: " + ", ".join(
            f"{s} {r['mean']:.1%}" for s, r in by_source.iterrows()))
    lines += [
        f"Abandoned:         {int(per_run['was_abandoned'].sum())}",
        f"Builds:            {dict(Counter(per_run['build_id']).most_common(5))}",
        "",
        "By character (player-runs, win rate, mean ascension):",
    ]
    by_char = (runs.groupby("character")
               .agg(runs=("run_id", "size"), win_rate=("win", "mean"),
                    asc=("ascension", "mean"))
               .sort_values("runs", ascending=False))
    for char, r in by_char.iterrows():
        lines.append(f"  {char:<12} {int(r['runs']):>7}  {r['win_rate']:6.1%}  A{r['asc']:.1f}")

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
            lines.append(f"  {card:<24} {n:>7} / {offered[card]:<7} ({n / offered[card]:.0%})")
    if stats.errors:
        lines += ["", f"Unreadable (first {len(stats.errors)}):"] + [
            f"  {e}" for e in stats.errors]
    return "\n".join(lines)


SUMMARY_CHOICE_COLUMNS = ["run_id", "player_id", "floor", "card_id",
                          "was_picked", "offer_skipped"]


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", type=Path, nargs="*", default=[Path("data/raw")],
                    help="directories or files (.run, .jsonl.gz); default data/raw")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--include-multiplayer", action="store_true",
                    help="keep co-op runs (excluded by default)")
    ap.add_argument("--card-allowlist", type=Path, default=DEFAULT_CARD_ALLOWLIST,
                    help="official card IDs; runs with other cards count as modded")
    ap.add_argument("--allow-mods", action="store_true",
                    help="keep runs with modded characters or cards")
    ap.add_argument("-v", "--verbose", action="store_true", help="log progress")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    allowed = None if args.allow_mods else load_card_allowlist(args.card_allowlist)
    stats = write_parquet(args.inputs, args.out, args.include_multiplayer, allowed)
    runs = pd.read_parquet(args.out / "runs.parquet")
    choices = pd.read_parquet(args.out / "card_choices.parquet",
                              columns=SUMMARY_CHOICE_COLUMNS)
    print(summarize(runs, choices, stats))
    print(f"\nWrote {args.out / 'runs.parquet'} and {args.out / 'card_choices.parquet'}")


if __name__ == "__main__":
    main()
