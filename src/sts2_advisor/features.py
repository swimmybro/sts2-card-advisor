"""Build card-pick training rows from raw runs.

Produces ``offers.parquet`` with one row per candidate in each post-combat
card reward, plus one ``SKIP`` row per reward. Every feature describes the
state *before* the pick: the deck is rebuilt floor by floor by replaying the
logged gains, removals and transforms from the starting deck.

Usage::

    python -m sts2_advisor.features data/raw data/external \
        --out data/processed/offers.parquet
"""

from __future__ import annotations

import argparse
import bisect
import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sts2_advisor.cards import TRAITS
from sts2_advisor.parse import (
    DEFAULT_CARD_ALLOWLIST, ParseStats, _strip, iter_kept_runs, load_card_allowlist,
)

log = logging.getLogger(__name__)

DEFAULT_CARD_META = Path("configs/cards.csv")
REWARD_ROOMS = {"monster", "elite", "boss"}
SKIP = "SKIP"
# 'OstyAttack' -> 'deck_osty_attack': how many deck cards have each trait.
TRAIT_COLUMNS = {t: "deck_" + re.sub(r"(?<!^)(?=[A-Z])", "_", t).lower() for t in TRAITS}

OFFER_SCHEMA = pa.schema([
    # identity / grouping
    ("offer_id", pa.string()), ("run_id", pa.string()), ("source", pa.string()),
    ("group", pa.string()),
    # run context
    ("character", pa.string()), ("ascension", pa.int64()), ("build_id", pa.string()),
    ("act", pa.int64()), ("floor", pa.int64()), ("room_type", pa.string()),
    ("hp", pa.int64()), ("max_hp", pa.int64()), ("hp_frac", pa.float64()),
    ("gold", pa.int64()), ("n_relics", pa.int64()),
    ("relics_held", pa.string()),  # '|'-separated relic ids; expanded at training time
    # deck before the pick
    ("deck_size", pa.int64()), ("deck_attacks", pa.int64()),
    ("deck_skills", pa.int64()), ("deck_powers", pa.int64()),
    ("deck_curses", pa.int64()), ("deck_basics", pa.int64()),
    ("deck_avg_cost", pa.float64()),
    *[(col, pa.int64()) for col in TRAIT_COLUMNS.values()],
    # deckbuilding measures (see deck_measures)
    ("deck_upgraded", pa.int64()), ("deck_upgrade_density", pa.float64()),
    ("deck_block_density", pa.float64()), ("deck_cycle_time", pa.float64()),
    ("deck_redraw_time", pa.float64()),
    # candidate
    ("n_offered", pa.int64()), ("candidate", pa.string()), ("is_skip", pa.bool_()),
    ("cand_type", pa.string()), ("cand_rarity", pa.string()),
    ("cand_cost", pa.int64()), ("cand_upgraded", pa.bool_()),
    ("cand_copies_in_deck", pa.int64()),
    ("cand_n_traits", pa.int64()), ("cand_synergy", pa.int64()),
    # how taking the option changes the deck measures (0 for SKIP)
    ("cand_d_upgrade_density", pa.float64()), ("cand_d_block_density", pa.float64()),
    ("cand_d_cycle_time", pa.float64()), ("cand_d_redraw_time", pa.float64()),
    # labels
    ("picked", pa.bool_()), ("run_won", pa.bool_()), ("floors_reached", pa.int64()),
])
OFFER_COLUMNS = OFFER_SCHEMA.names
# The deck before each pick, stored once per reward in a side file (repeating it
# on every option's row made the table far too big to load): '|'-separated,
# repeats = copies. Rows from reward_rows() also carry it as ``deck_cards``.
DECK_SCHEMA = pa.schema([("offer_id", pa.string()), ("deck_cards", pa.string())])


def decks_path(offers_path: Path) -> Path:
    """offers.parquet -> offers_decks.parquet"""
    offers_path = Path(offers_path)
    return offers_path.with_name(offers_path.stem + "_decks.parquet")


class RunRejected(ValueError):
    """A run whose data can't be trusted; ``reason`` is a short label for stats."""
    reason = "rejected"


class DeckReplayError(RunRejected):
    """The logged deck changes don't add up to the final deck."""
    reason = "deck replay failed"


class BadHpError(RunRejected):
    """HP outside [0, max_hp] (edited saves)."""
    reason = "impossible HP"


@dataclass
class FeatureStats:
    parse: ParseStats
    n_runs: int = 0
    n_offers: int = 0
    n_rows: int = 0
    n_multi_pick: int = 0
    rejected: Counter = field(default_factory=Counter)


def load_card_meta(path: Path) -> Dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: {"type": r["type"], "rarity": r["rarity"],
                          "cost": int(r["cost"]),
                          "traits": frozenset(filter(None, (r.get("traits") or "").split("|"))),
                          "draw": int(r.get("draw") or 0),
                          "defensive": r.get("defensive") == "True"}
                for r in csv.DictReader(f)}


def _upgrade_changes(stats: Dict[str, Any]) -> int:
    """Net change in the number of upgraded cards at one map point."""
    def up(card: Dict[str, Any]) -> bool:
        return (card.get("current_upgrade_level") or 0) > 0

    n = len(stats.get("upgraded_cards", [])) - len(stats.get("downgraded_cards", []))
    n += sum(up(c) for c in stats.get("cards_gained", []))
    n -= sum(up(c) for c in stats.get("cards_removed", []))
    for t in stats.get("cards_transformed", []):
        n += up(t["final_card"]) - up(t["original_card"])
    return n


# Cards drawn per turn without draw effects.
HAND_SIZE = 5


def _deck_totals(deck: Counter, meta: Dict[str, dict]) -> Dict[str, int]:
    """Counts behind the deckbuilding measures."""
    size = sum(deck.values())
    defensive = draw = staying = 0
    for card, n in deck.items():
        m = meta.get(card, {})
        defensive += n * m.get("defensive", False)
        draw += n * m.get("draw", 0)
        # powers and Exhaust cards don't come back after a reshuffle
        staying += n * (m.get("type") != "Power" and "Exhaust" not in m.get("traits", ()))
    return {"size": size, "defensive": defensive, "draw": draw, "staying": staying}


def deck_measures(t: Dict[str, int], n_upgraded: int) -> Dict[str, Any]:
    """Deckbuilding measures from Baalorlord's 'Core Deckbuilding Concepts':

    * upgrade density: upgraded cards / deck size
    * block density: cards that reduce incoming damage / deck size
    * cycle time: turns to draw the whole deck, (size - card draw) / 5
    * redraw time: the same after the first shuffle, without powers and Exhaust cards
    """
    size = t["size"]
    return {
        "deck_upgraded": n_upgraded,
        "deck_upgrade_density": min(n_upgraded, size) / size if size else None,
        "deck_block_density": t["defensive"] / size if size else None,
        "deck_cycle_time": max(size - t["draw"], 0) / HAND_SIZE,
        "deck_redraw_time": max(t["staying"] - t["draw"], 0) / HAND_SIZE,
    }


def option_deltas(t: Dict[str, int], n_upgraded: int, card: Optional[str], upgraded: bool,
                  meta: Dict[str, dict]) -> Dict[str, float]:
    """How taking ``card`` changes the deck measures (all 0 for SKIP)."""
    keys = ("upgrade_density", "block_density", "cycle_time", "redraw_time")
    if card is None:
        return {f"cand_d_{k}": 0.0 for k in keys}
    m = meta.get(card, {})
    after = dict(t, size=t["size"] + 1, defensive=t["defensive"] + m.get("defensive", False),
                 draw=t["draw"] + m.get("draw", 0),
                 staying=t["staying"] + (m.get("type") != "Power"
                                         and "Exhaust" not in m.get("traits", ())))
    before, now = deck_measures(t, n_upgraded), deck_measures(after, n_upgraded + upgraded)
    return {f"cand_d_{k}": (now[f"deck_{k}"] or 0) - (before[f"deck_{k}"] or 0) for k in keys}


def _deck_changes(stats: Dict[str, Any]) -> Counter:
    """Net card-count change at one map point (keyed by stripped card id)."""
    delta: Counter = Counter()
    for c in stats.get("cards_gained", []):
        delta[_strip(c["id"])] += 1
    for c in stats.get("cards_removed", []):
        delta[_strip(c["id"])] -= 1
    for t in stats.get("cards_transformed", []):
        delta[_strip(t["original_card"]["id"])] -= 1
        delta[_strip(t["final_card"]["id"])] += 1
    return delta


def starting_deck(data: Dict[str, Any]) -> Counter:
    """Final deck minus every logged change = the deck the run started with."""
    player = data["players"][0]
    deck = Counter(_strip(c["id"]) for c in player.get("deck", []))
    for act in data["map_point_history"]:
        for point in act:
            for s in point.get("player_stats", []):
                deck.subtract(_deck_changes(s))
    if any(n < 0 for n in deck.values()):
        bad = sorted(k for k, n in deck.items() if n < 0)
        raise DeckReplayError(f"negative starting count for {bad}")
    return +deck


def _deck_features(deck: Counter, meta: Dict[str, dict]) -> Dict[str, Any]:
    by_type: Counter = Counter()
    traits: Counter = Counter()
    n_basic = cost_total = n_costed = 0
    for card, n in deck.items():
        m = meta.get(card, {})
        by_type[m.get("type")] += n
        for t in m.get("traits", ()):
            traits[t] += n
        n_basic += n if m.get("rarity") == "Basic" else 0
        if m.get("cost") is not None and m["cost"] >= 0:  # -1 = unplayable / X
            cost_total += m["cost"] * n
            n_costed += n
    return {
        "deck_size": sum(deck.values()),
        "deck_attacks": by_type["Attack"],
        "deck_skills": by_type["Skill"],
        "deck_powers": by_type["Power"],
        "deck_curses": by_type["Curse"] + by_type["Status"],
        "deck_basics": n_basic,
        "deck_avg_cost": cost_total / n_costed if n_costed else None,
        **{col: traits[t] for t, col in TRAIT_COLUMNS.items()},
    }


def reward_rows(base: Dict[str, Any], candidates: List[Dict[str, Any]],
                deck: Counter, meta: Dict[str, dict]) -> List[dict]:
    """Rows for one card reward: each candidate plus a ``SKIP`` row.

    ``base`` holds the run state (character, ascension, act, floor,
    room_type, hp, max_hp, gold, n_relics, ``deck_upgraded`` (upgraded cards
    in the deck), plus any id/label fields). ``candidates`` are
    ``{"id", "upgraded", "picked"}`` dicts; ``picked`` may be omitted when
    scoring a live reward. ``deck`` is the deck *before* the pick. Shared by
    the feature builder and the API so both compute features the same way.
    """
    hp, max_hp = base.get("hp"), base.get("max_hp")
    n_up = base.get("deck_upgraded") or 0
    totals = _deck_totals(deck, meta)
    base = dict(base, n_offered=len(candidates),
                hp_frac=hp / max_hp if hp is not None and max_hp else None,
                deck_cards=deck_string(deck),
                **_deck_features(deck, meta), **deck_measures(totals, n_up))
    rows = []
    for c in candidates:
        m = meta.get(c["id"], {})
        traits = m.get("traits", frozenset())
        rows.append(dict(
            base, candidate=c["id"], is_skip=False,
            cand_type=m.get("type"), cand_rarity=m.get("rarity"), cand_cost=m.get("cost"),
            cand_upgraded=bool(c.get("upgraded")), cand_copies_in_deck=deck[c["id"]],
            cand_n_traits=len(traits),
            # deck cards sharing each of the candidate's traits, summed
            cand_synergy=sum(base[TRAIT_COLUMNS[t]] for t in traits if t in TRAIT_COLUMNS),
            **option_deltas(totals, n_up, c["id"], bool(c.get("upgraded")), meta),
            picked=bool(c.get("picked")),
        ))
    rows.append(dict(
        base, candidate=SKIP, is_skip=True, cand_type=None, cand_rarity=None,
        cand_cost=None, cand_upgraded=None, cand_copies_in_deck=None,
        cand_n_traits=None, cand_synergy=None,
        **option_deltas(totals, n_up, None, False, meta),
        picked=not any(c.get("picked") for c in candidates),
    ))
    return rows


def deck_string(deck: Counter) -> str:
    """Counter({'B': 2, 'A': 1}) -> 'A|B|B' (sorted; repeats = copies)."""
    return "|".join(card for card in sorted(deck) for _ in range(deck[card]) if card)


def relic_fields(relic_ids: Iterable[str]) -> Dict[str, Any]:
    """``n_relics`` and ``relics_held`` for the relics held at a reward."""
    ids = sorted(set(relic_ids))
    return {"n_relics": len(ids), "relics_held": "|".join(ids)}


def _group(data: Dict[str, Any], run_id: str) -> str:
    """Train/test grouping key: the player when known, otherwise the run.

    Your own runs are grouped per run too, so a split spreads them across
    train and test; hold them out explicitly to evaluate on your play.
    """
    return data.get("player_token") or f"run:{run_id}"


def offer_rows(data: Dict[str, Any], run_id: str, source: str,
               meta: Dict[str, dict], fstats: Optional[FeatureStats] = None
               ) -> List[dict]:
    """Candidate rows (+ SKIP) for every card reward in a single-player run.

    Rewards with more than one pick (e.g. two reward screens logged together)
    are left out. Raises :class:`RunRejected` for untrustworthy runs.
    """
    if len(data["players"]) != 1:
        return []
    player = data["players"][0]
    pid = player["id"]
    deck = starting_deck(data)
    relics = sorted((r.get("floor_added_to_deck", 0), _strip(r["id"]))
                    for r in player.get("relics", []))
    relic_floors = [f for f, _ in relics]
    n_floors = sum(len(act) for act in data["map_point_history"])
    run_ctx = {
        "run_id": run_id, "source": source, "group": _group(data, run_id),
        "character": _strip(player.get("character")),
        "ascension": data.get("ascension"), "build_id": data.get("build_id"),
        "run_won": None if data.get("was_abandoned") else bool(data["win"]),
        "floors_reached": n_floors,
    }

    rows: List[dict] = []
    floor = 0
    n_upgraded = 0  # upgraded cards in the deck (matches the final deck in ~98.5% of runs)
    for act_idx, act in enumerate(data["map_point_history"], start=1):
        for point in act:
            floor += 1
            stats = next((s for s in point.get("player_stats", [])
                          if s.get("player_id") == pid), {})
            hp, max_hp = stats.get("current_hp"), stats.get("max_hp")
            if hp is not None and max_hp is not None and not 0 <= hp <= max_hp:
                raise BadHpError(f"floor {floor}: hp {hp} / max {max_hp}")
            offer = stats.get("card_choices") or []
            if not offer or point.get("map_point_type") not in REWARD_ROOMS:
                pass
            elif sum(bool(c.get("was_picked")) for c in offer) > 1:
                if fstats is not None:
                    fstats.n_multi_pick += 1
            else:
                base = dict(
                    run_ctx,
                    offer_id=f"{run_id}:{floor}", act=act_idx, floor=floor,
                    room_type=point["map_point_type"], hp=hp, max_hp=max_hp,
                    gold=stats.get("current_gold"), deck_upgraded=n_upgraded,
                    # relics gained on earlier floors only
                    **relic_fields(r for _, r in relics[:bisect.bisect_left(relic_floors, floor)]),
                )
                candidates = [
                    {"id": _strip(c["card"]["id"]),
                     "upgraded": c["card"].get("current_upgrade_level", 0) > 0,
                     "picked": bool(c.get("was_picked"))}
                    for c in offer
                ]
                rows.extend(reward_rows(base, candidates, deck, meta))
            # Changes at this point happen at/after the reward, so apply them last.
            deck.update(_deck_changes(stats))
            n_upgraded = max(n_upgraded + _upgrade_changes(stats), 0)
    return rows


def build_offers(paths: Iterable[Path], out: Path, meta: Dict[str, dict],
                 allowed_cards: Optional[Set[str]] = None,
                 chunk_runs: int = 5000) -> FeatureStats:
    """Stream offer rows for every kept solo run into ``out`` (Parquet), and each
    reward's deck into :func:`decks_path` ``(out)``."""
    stats = FeatureStats(parse=ParseStats())
    out.parent.mkdir(parents=True, exist_ok=True)
    buf: List[dict] = []

    def flush() -> None:
        writer.write_table(pa.Table.from_pylist(buf, schema=OFFER_SCHEMA))
        deck_writer.write_table(pa.Table.from_pylist(
            [{"offer_id": r["offer_id"], "deck_cards": r["deck_cards"]}
             for r in buf if r["is_skip"]], schema=DECK_SCHEMA))  # one row per reward
        buf.clear()

    with pq.ParquetWriter(out, OFFER_SCHEMA) as writer, \
            pq.ParquetWriter(decks_path(out), DECK_SCHEMA) as deck_writer:
        for run_id, source, data, _, _ in iter_kept_runs(
                paths, stats.parse, include_multiplayer=False,
                allowed_cards=allowed_cards):
            multi_before = stats.n_multi_pick
            try:
                rows = offer_rows(data, run_id, source, meta, stats)
            except RunRejected as e:
                stats.n_multi_pick = multi_before
                stats.rejected[e.reason] += 1
                log.debug("rejected %s: %s", run_id, e)
                continue
            stats.n_runs += 1
            stats.n_rows += len(rows)
            stats.n_offers += sum(r["is_skip"] for r in rows)
            buf.extend(rows)
            if stats.n_runs % chunk_runs == 0:
                flush()
                log.info("built %d runs", stats.n_runs)
        flush()
    return stats


def split_by_group(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split where each ``group`` (player or run) lands on one side only."""
    groups = df["group"].unique()
    rng = np.random.default_rng(seed)
    test_groups = set(rng.choice(groups, size=max(1, round(len(groups) * test_frac)),
                                 replace=False))
    is_test = df["group"].isin(test_groups)
    return df[~is_test], df[is_test]


def summarize(offers: pd.DataFrame, stats: FeatureStats) -> str:
    p = stats.parse
    skips = offers[offers["is_skip"]]
    cards = offers[~offers["is_skip"]]
    rejected = ", ".join(f"{n} {why}" for why, n in sorted(stats.rejected.items())) or "none"
    lines = [
        f"Runs used:         {stats.n_runs}",
        f"Runs dropped:      parser: {p.n_errors} unreadable, {p.n_multiplayer} multiplayer, "
        f"{p.n_duplicates} duplicates, {p.n_modded} modded; features: {rejected}",
        f"Card rewards:      {stats.n_offers}  ({stats.n_rows} rows incl. SKIP; "
        f"{stats.n_multi_pick} multi-pick rewards left out)",
        f"Skip rate:         {skips['picked'].mean():.1%}",
        f"Split groups:      {offers['group'].nunique()}  "
        f"(players where known, otherwise runs)",
        f"Unknown card meta: {cards['cand_type'].isna().mean():.2%} of candidate rows",
        "",
        "Rewards by character:",
    ]
    for ch, n in skips["character"].value_counts().items():
        lines.append(f"  {ch:<12} {n:>7}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", type=Path, nargs="*", default=[Path("data/raw")])
    ap.add_argument("--out", type=Path, default=Path("data/processed/offers.parquet"))
    ap.add_argument("--card-meta", type=Path, default=DEFAULT_CARD_META)
    ap.add_argument("--card-allowlist", type=Path, default=DEFAULT_CARD_ALLOWLIST)
    ap.add_argument("--allow-mods", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    allowed = None if args.allow_mods else load_card_allowlist(args.card_allowlist)
    stats = build_offers(args.inputs, args.out, load_card_meta(args.card_meta), allowed)
    offers = pd.read_parquet(args.out, columns=["is_skip", "picked", "character",
                                                "group", "cand_type"])
    print(summarize(offers, stats))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
