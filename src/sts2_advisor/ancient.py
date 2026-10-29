"""Start-of-act choices: which of an ancient's relics gives the best chance of winning.

Each act opens with an ancient (Neow in act 1; Tezcatara, Pael, Orobas, Darv,
Nonupeipe, Tanx or Vakuu later) offering a few relics, one of which you take.
Run histories record every option and the one picked, so this works like
the card advisor: a pick model learns what players take, a win model learns
which choice wins, and the advice is the highest win chance, with ties
broken by what players take most.

Usage::

    python -m sts2_advisor.ancient build    # data/processed/ancients.parquet
    python -m sts2_advisor.ancient train    # models/ancient/
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sts2_advisor.advise import _suggest, normalize_id
from sts2_advisor.features import (
    DEFAULT_CARD_META, TRAIT_COLUMNS, RunRejected, _deck_changes, _deck_features, _deck_totals,
    _group, _upgrade_changes, deck_measures, load_card_meta, split_by_group, starting_deck,
)
from sts2_advisor.parse import (
    DEFAULT_CARD_ALLOWLIST, ParseStats, _strip, iter_kept_runs, load_card_allowlist,
)
from sts2_advisor.win import (
    DEFAULT_RELIABILITY, adjust_win_chances, auc, fit_temperature, near_best,
    pick_probabilities,
)

log = logging.getLogger(__name__)

DEFAULT_DATA = Path("data/processed/ancients.parquet")
DEFAULT_MODEL_DIR = Path("models/ancient")
# The game's own ancients; mods add others, which are left out.
OFFICIAL_ANCIENTS = {"NEOW", "TEZCATARA", "PAEL", "OROBAS", "DARV", "NONUPEIPE", "TANX", "VAKUU"}

DECK_FEATURES = ["deck_size", "deck_attacks", "deck_skills", "deck_powers", "deck_curses",
                 "deck_basics", "deck_avg_cost", *TRAIT_COLUMNS.values(), "deck_upgraded",
                 "deck_upgrade_density", "deck_block_density", "deck_cycle_time",
                 "deck_redraw_time"]
NUMERIC = ["ascension", "act", "floor", "hp", "max_hp", "hp_frac", "gold", "n_relics",
           "n_options", *DECK_FEATURES]
CATEGORICAL = ["character", "ancient", "option"]
FEATURES = NUMERIC + CATEGORICAL

SCHEMA = pa.schema([("choice_id", pa.string()), ("group", pa.string()),
                    ("run_id", pa.string()), ("source", pa.string()),
                    *[(c, pa.string()) for c in CATEGORICAL],
                    *[(c, pa.float64()) for c in NUMERIC],
                    ("picked", pa.bool_()), ("run_won", pa.float64())])


# ----------------------------------------------------------------------- data

def choice_rows(base: Dict[str, Any], options: Sequence[str], deck: Counter,
                n_upgraded: int, meta: Dict[str, dict],
                picked: Optional[str] = None) -> List[dict]:
    """Rows for one ancient's choice, one per option. ``base`` holds the run
    state before the choice (character, ascension, act, floor, ancient, hp,
    max_hp, gold, n_relics). Shared by the data builder and live advice."""
    hp, max_hp = base.get("hp"), base.get("max_hp")
    state = dict(base, n_options=len(options),
                 hp_frac=hp / max_hp if hp is not None and max_hp else None,
                 **_deck_features(deck, meta),
                 **deck_measures(_deck_totals(deck, meta), n_upgraded))
    return [dict(state, option=o, picked=o == picked) for o in options]


def ancient_rows(data: Dict[str, Any], run_id: str, source: str,
                 meta: Dict[str, dict]) -> List[dict]:
    """Option rows for every ancient's choice in a single-player run."""
    if len(data["players"]) != 1:
        return []
    player = data["players"][0]
    pid = player["id"]
    deck = starting_deck(data)
    relic_floors = sorted(r.get("floor_added_to_deck", 0) for r in player.get("relics", []))
    run_ctx = {"run_id": run_id, "source": source, "group": _group(data, run_id),
               "character": _strip(player.get("character")),
               "ascension": data.get("ascension"),
               "run_won": None if data.get("was_abandoned") else float(bool(data["win"]))}
    rows: List[dict] = []
    floor, n_upgraded = 0, 0
    before: Dict[str, Any] = {}  # the previous floor's stats: the state going in
    for act_idx, act in enumerate(data["map_point_history"], start=1):
        for point in act:
            floor += 1
            stats = next((s for s in point.get("player_stats", [])
                          if s.get("player_id") == pid), {})
            offer = stats.get("ancient_choice") or []
            ancient = _strip(((point.get("rooms") or [{}])[0]).get("model_id"))
            n_chosen = sum(bool(o.get("was_chosen")) for o in offer)
            if 2 <= len(offer) <= 4 and n_chosen == 1 and ancient in OFFICIAL_ANCIENTS:
                base = dict(run_ctx, choice_id=f"{run_id}:{floor}", act=act_idx, floor=floor,
                            ancient=ancient, hp=before.get("current_hp"),
                            max_hp=before.get("max_hp"), gold=before.get("current_gold"),
                            n_relics=bisect.bisect_left(relic_floors, floor))
                options = [o["TextKey"] for o in offer]
                chosen = next(o["TextKey"] for o in offer if o.get("was_chosen"))
                if len(set(options)) == len(options):  # a few odd runs repeat one relic
                    rows.extend(choice_rows(base, options, deck, n_upgraded, meta, chosen))
            deck.update(_deck_changes(stats))
            n_upgraded = max(n_upgraded + _upgrade_changes(stats), 0)
            before = stats
    return rows


def build(paths: Iterable[Path], out: Path, meta: Dict[str, dict],
          allowed_cards: Optional[Set[str]] = None, chunk_runs: int = 5000) -> Dict[str, int]:
    """Stream every kept solo run's ancient choices into ``out`` (Parquet)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stats, counts, buf = ParseStats(), Counter(), []

    def flush() -> None:
        df = pd.DataFrame(buf).reindex(columns=SCHEMA.names)
        df[NUMERIC + ["run_won"]] = df[NUMERIC + ["run_won"]].astype(float)
        writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
        buf.clear()

    with pq.ParquetWriter(out, SCHEMA) as writer:
        for run_id, source, data, _, _ in iter_kept_runs(paths, stats, allowed_cards=allowed_cards):
            try:
                rows = ancient_rows(data, run_id, source, meta)
            except RunRejected:
                counts["rejected"] += 1
                continue
            counts["runs"] += 1
            counts["choices"] += len({r["choice_id"] for r in rows})
            buf.extend(rows)
            if counts["runs"] % chunk_runs == 0:
                flush()
                log.info("built %d runs", counts["runs"])
        if buf:
            flush()
    return dict(counts)


# ---------------------------------------------------------------------- model

def to_matrix(df: pd.DataFrame, categories: Dict[str, List[str]]) -> pd.DataFrame:
    X = df[NUMERIC].astype(float).copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=categories[c])
    return X


def _predict(booster: lgb.Booster, categories: Dict[str, List[str]], df: pd.DataFrame) -> np.ndarray:
    return booster.predict(to_matrix(df, categories), num_iteration=booster.best_iteration or None)


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of a choice next to each other (needed for ranking groups)."""
    return df.sort_values(["choice_id", "option"], kind="stable").reset_index(drop=True)


def train_models(train: pd.DataFrame, valid: pd.DataFrame, seed: int = 0
                 ) -> Tuple[lgb.Booster, lgb.Booster, Dict[str, List[str]]]:
    """A pick model (what players take) and a win model (taken option -> run won)."""
    categories = {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL}
    common = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                  feature_fraction=0.9, verbose=-1, seed=seed)
    callbacks = [lgb.early_stopping(50, verbose=False)]

    def sizes(d):
        return d.groupby("choice_id", sort=False).size().to_numpy()

    pick = lgb.train(
        dict(common, objective="lambdarank", metric="ndcg", eval_at=[1]),
        lgb.Dataset(to_matrix(train, categories), label=train["picked"].astype(int),
                    group=sizes(train)),
        num_boost_round=2000, callbacks=callbacks,
        valid_sets=[lgb.Dataset(to_matrix(valid, categories),
                                label=valid["picked"].astype(int), group=sizes(valid))])
    tr = train[train["picked"] & train["run_won"].notna()]
    va = valid[valid["picked"] & valid["run_won"].notna()]
    win = lgb.train(
        dict(common, objective="binary", metric="binary_logloss"),
        lgb.Dataset(to_matrix(tr, categories), label=tr["run_won"].astype(int)),
        num_boost_round=3000, callbacks=callbacks,
        valid_sets=[lgb.Dataset(to_matrix(va, categories), label=va["run_won"].astype(int))])
    return pick, win, categories


def score(rows: pd.DataFrame, pick: lgb.Booster, win: lgb.Booster,
          categories: Dict[str, List[str]], rel: Dict[str, Any]) -> pd.DataFrame:
    """Pick and win scores plus the suggestion, for rows keyed by ``choice_id``."""
    rows = rows.copy()
    ids = rows["choice_id"].to_numpy()
    rows["score"] = _predict(pick, categories, rows)
    rows["raw_win"] = _predict(win, categories, rows)
    rows["pick_chance"] = pick_probabilities(ids, rows["score"], rel["temperature"])
    rows["win"], _ = adjust_win_chances(ids, rows["raw_win"], rows["pick_chance"], rel["shrink_k"])
    rows["low_confidence"] = rows["pick_chance"] < rel["low_confidence_below"]
    close = pd.Series(near_best(ids, rows["win"], rel["tie_margin"]), index=rows.index)
    rows["tied"] = close & (close.groupby(ids).transform("sum") > 1)
    rows["known"] = rows["option"].isin(categories["option"])
    return _suggest(rows.assign(offer_id=ids), has_win=True).drop(columns="offer_id")


def matched_gap(d: pd.DataFrame, col: str, min_each: int = 10) -> Tuple[float, int]:
    """Win rate when ``col`` (e.g. 'took the suggestion') vs not, compared
    within situations (character, ascension, act, ancient) and averaged."""
    s = d.groupby(["stratum", col])["won"].agg(["sum", "count"]).unstack(col).dropna()
    s = s[(s[("count", True)] >= min_each) & (s[("count", False)] >= min_each)]
    if s.empty:
        return float("nan"), 0
    gap = s[("sum", True)] / s[("count", True)] - s[("sum", False)] / s[("count", False)]
    w = s[("count", True)] + s[("count", False)]
    return float((gap * w).sum() / w.sum()), int(w.sum())


def evaluate(df: pd.DataFrame, scored: pd.DataFrame) -> Dict[str, Any]:
    """Pick accuracy, win AUC, and whether following the advice goes with winning."""
    taken = scored[scored["picked"]].set_index("choice_id")
    top_pick = scored.loc[scored.groupby("choice_id")["score"].idxmax()].set_index("choice_id")
    suggested = scored[scored["suggested"]].set_index("choice_id")
    labelled = taken[taken["run_won"].notna()]
    out = {
        "n_choices": int(len(taken)),
        "pick_top1": float((top_pick["option"] == taken.loc[top_pick.index, "option"]).mean()),
        "win_auc": auc(labelled["run_won"].astype(bool).to_numpy(), labelled["raw_win"].to_numpy()),
        "agrees_with_player": float((suggested["option"] == taken.loc[suggested.index, "option"]).mean()),
        "tie_rate": float(suggested["suggestion_source"].eq("win_tie_popular").mean()),
    }
    d = labelled.assign(
        won=labelled["run_won"].astype(bool),
        followed=labelled["option"] == suggested.loc[labelled.index, "option"],
        took_popular=labelled["option"] == top_pick.loc[labelled.index, "option"],
        stratum=(labelled["character"] + "|" + (labelled["ascension"] // 3).astype(str) + "|"
                 + labelled["act"].astype(str) + "|" + labelled["ancient"]))
    for col in ("followed", "took_popular"):
        gap, n = matched_gap(d, col)
        out[f"matched_gap_{col}"], out[f"matched_n_{col}"] = gap, n
    return out


def train(data: Path = DEFAULT_DATA, model_dir: Path = DEFAULT_MODEL_DIR,
          seed: int = 0) -> Dict[str, Any]:
    df = pd.read_parquet(data)
    repeats = df.duplicated(["choice_id", "option"], keep=False)  # see ancient_rows
    df = df[~df["choice_id"].isin(df.loc[repeats, "choice_id"])]
    mine = df[df["source"] == "mine"]
    rest = df.drop(mine.index)
    train_df, test = split_by_group(rest, 0.2, seed)  # same fractions as the card models
    train_df, valid = split_by_group(train_df, 0.1, seed + 1)
    train_df, valid, test, mine = map(_sorted, (train_df, valid, test, mine))
    pick, win, categories = train_models(train_df, valid, seed)
    rel = dict(DEFAULT_RELIABILITY,
               temperature=fit_temperature(valid.rename(columns={"choice_id": "offer_id"}),
                                           _predict(pick, categories, valid)))
    metrics = {"best_iteration": {"pick": pick.best_iteration, "win": win.best_iteration},
               "reliability": rel}
    for name, d in (("test", test), ("mine", mine)):
        if len(d):
            metrics[name] = evaluate(d, score(d, pick, win, categories, rel))
    model_dir.mkdir(parents=True, exist_ok=True)
    pick.save_model(str(model_dir / "pick.txt"))
    win.save_model(str(model_dir / "win.txt"))
    # which ancient offers each option, to name the ancient from the options alone
    offered_by = (train_df.groupby(["option", "ancient"]).size().rename("n").reset_index()
                  .sort_values("n", ascending=False).drop_duplicates("option"))
    (model_dir / "meta.json").write_text(json.dumps({
        "categories": categories, "reliability": rel,
        "offered_by": dict(zip(offered_by["option"], offered_by["ancient"])),
        "metrics": metrics}, indent=2))
    return metrics


# --------------------------------------------------------------------- advice

class AncientAdvisor:
    """Advice on a live ancient's choice."""

    def __init__(self, model_dir: Optional[Path] = None, card_meta: Optional[Path] = None):
        self.model_dir = Path(model_dir or os.environ.get("ANCIENT_MODEL_DIR", DEFAULT_MODEL_DIR))
        if not (self.model_dir / "win.txt").exists():
            raise RuntimeError(f"No start-of-act model at {self.model_dir}. Train one with "
                               "`python -m sts2_advisor.ancient build` then `... train`.")
        m = json.loads((self.model_dir / "meta.json").read_text())
        self.categories, self.reliability = m["categories"], m["reliability"]
        self.offered_by: Dict[str, str] = m["offered_by"]
        self.pick = lgb.Booster(model_file=str(self.model_dir / "pick.txt"))
        self.win = lgb.Booster(model_file=str(self.model_dir / "win.txt"))
        self.meta = load_card_meta(Path(card_meta or os.environ.get("CARD_META",
                                                                    DEFAULT_CARD_META)))

    @property
    def options(self) -> List[str]:
        """Every option the models know."""
        return list(self.categories["option"])

    def ancient_for(self, options: Sequence[str]) -> Optional[str]:
        """The ancient that offers these options (most of them, at least)."""
        votes = Counter(self.offered_by[o] for o in options if o in self.offered_by)
        return votes.most_common(1)[0][0] if votes else None

    def rank(self, state: Dict[str, Any], deck: List[Tuple[str, bool]], relics: List[str],
             options: Sequence[str], ancient: Optional[str] = None) -> pd.DataFrame:
        """Score one live choice. ``state``: character, ascension, act, floor,
        hp, max_hp, gold; ``deck`` ``(card id, upgraded)`` pairs; ``options``
        relic ids. Returns the options, the suggestion first, then by win chance."""
        base = dict(state, ancient=ancient or self.ancient_for(options), n_relics=len(relics),
                    choice_id="live")
        rows = pd.DataFrame(choice_rows(base, list(options), Counter(c for c, _ in deck),
                                        sum(up for _, up in deck), self.meta))
        rows = score(rows, self.pick, self.win, self.categories, self.reliability)
        return rows.sort_values(["suggested", "win"], ascending=False)


def normalize_option(name: str) -> str:
    """'Arcane Scroll' / 'RELIC.ARCANE_SCROLL' -> 'ARCANE_SCROLL'."""
    return normalize_id(name, "relic")


# ----------------------------------------------------------------------- main

def report(metrics: Dict[str, Any]) -> str:
    lines = []
    for name, title in (("test", "Community test set"), ("mine", "Your runs")):
        m = metrics.get(name)
        if not m:
            continue
        lines += [f"{title}: {m['n_choices']} choices",
                  f"  Pick model names the player's choice: {m['pick_top1']:.1%}",
                  f"  Win model AUC (taken option -> run won): {m['win_auc']:.3f}",
                  f"  Advice agrees with the player: {m['agrees_with_player']:.1%} "
                  f"(ties broken by popularity: {m['tie_rate']:.1%})",
                  f"  Matched win-rate gap, took the advice vs not: "
                  f"{m['matched_gap_followed']:+.1%} ({m['matched_n_followed']} choices)",
                  f"  Matched win-rate gap, took the popular option vs not: "
                  f"{m['matched_gap_took_popular']:+.1%} ({m['matched_n_took_popular']} choices)"]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="extract ancient choices from run histories")
    b.add_argument("inputs", nargs="*", type=Path, default=[Path("data/raw"), Path("data/external")])
    b.add_argument("--out", type=Path, default=DEFAULT_DATA)
    t = sub.add_parser("train", help="train the pick and win models")
    t.add_argument("--data", type=Path, default=DEFAULT_DATA)
    t.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    if args.cmd == "build":
        counts = build(args.inputs, args.out, load_card_meta(DEFAULT_CARD_META),
                       load_card_allowlist(DEFAULT_CARD_ALLOWLIST))
        print(f"{counts.get('choices', 0)} choices from {counts.get('runs', 0)} runs -> {args.out}")
    else:
        print(report(train(args.data, args.model_dir)))
        print(f"\nSaved models to {args.model_dir}")


if __name__ == "__main__":
    main()
