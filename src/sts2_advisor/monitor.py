"""Check whether the models still fit recent runs, and flag when to retrain.

Compares recent runs (your new runs, or a newer community export) with the
data the models were trained on: unlisted cards (likely new ones after a
patch), new game versions, shifts in pick rates, pick-model accuracy and
win-model calibration.

    python -m sts2_advisor.monitor data/external/runs-v0.112.0.jsonl.gz
    python -m sts2_advisor.monitor data/raw --since-model   # your runs since training

Each check reports ok / drift / low data; the verdict says whether to retrain.
``--fail-on-drift`` exits with status 1 on drift (for scheduled jobs).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from sts2_advisor.advise import Advisor
from sts2_advisor.features import RunRejected, offer_rows
from sts2_advisor.parse import (
    DEFAULT_CARD_ALLOWLIST, OFFICIAL_CHARACTERS, ParseStats, iter_kept_runs,
    load_card_allowlist,
)
from sts2_advisor.train import evaluate
from sts2_advisor.win import auc, outcome_rows

log = logging.getLogger(__name__)

OK, DRIFT, LOW_DATA, INFO = "ok", "drift", "low data", "info"


@dataclass
class Check:
    name: str
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecentRuns:
    n_runs: int = 0
    n_modded_characters: int = 0
    builds: Counter = field(default_factory=Counter)
    unlisted_card_runs: Counter = field(default_factory=Counter)  # card -> runs containing it
    n_runs_with_unlisted: int = 0
    offers: pd.DataFrame = field(default_factory=pd.DataFrame)
    sources: Counter = field(default_factory=Counter)


def collect_recent(paths: List[Path], advisor: Advisor, allowed: set,
                   since: Optional[int] = None) -> RecentRuns:
    """Solo runs with official characters started at or after ``since``.

    Runs with cards outside the allowlist are counted (to spot new cards) but
    left out of the offer rows, like in training.
    """
    recent = RecentRuns()
    rows: List[dict] = []
    for run_id, source, data, runs, choices in iter_kept_runs(paths, ParseStats()):
        if since is not None and (data.get("start_time") or 0) < since:
            continue
        if any(r["character"] not in OFFICIAL_CHARACTERS for r in runs):
            recent.n_modded_characters += 1
            continue
        recent.n_runs += 1
        recent.sources[source] += 1
        recent.builds[data.get("build_id")] += 1
        cards = {c["card_id"] for c in choices} | set(runs[0]["deck"].split("|"))
        unlisted = cards - allowed - {None, ""}
        if unlisted:
            recent.n_runs_with_unlisted += 1
            recent.unlisted_card_runs.update(unlisted)
            continue
        try:
            rows.extend(offer_rows(data, run_id, source, advisor.meta))
        except RunRejected:
            continue
    recent.offers = pd.DataFrame(rows)
    return recent


def population(sources: Counter) -> str:
    """Compare like with like: your runs against your earlier runs, else the community."""
    return "mine" if sources and set(sources) == {"mine"} else "codex"


def reference_stats(offers_path: Path, pop: str, exclude_runs: set) -> Dict[str, Any]:
    """Pick rates, skip rate and builds of the reference population, leaving out
    the recent runs themselves."""
    ref = pd.read_parquet(offers_path, columns=["run_id", "candidate", "picked", "is_skip",
                                               "source", "build_id"],
                          filters=[("source", "==", pop)])
    ref = ref[~ref["run_id"].isin(exclude_runs)]
    cards = ref[~ref["is_skip"]]
    trained_on = pd.read_parquet(offers_path, columns=["build_id"],
                                 filters=[("source", "==", "codex")])
    return {
        "population": pop,
        "n_rows": int(len(ref)),
        "pick_rate": cards.groupby("candidate")["picked"].agg(["mean", "size"]),
        "skip_rate": float(ref.loc[ref["is_skip"], "picked"].mean()) if len(ref) else float("nan"),
        # versions the models were trained on (community data), whatever the population
        "builds": set(trained_on["build_id"].dropna().unique()),
    }


# ---------------------------------------------------------------------- checks

def check_runs(recent: RecentRuns, t: Dict[str, Any]) -> Check:
    if recent.n_runs < t["min_runs"]:
        return Check("Recent runs", LOW_DATA,
                     f"{recent.n_runs} solo runs (need {t['min_runs']} for a reliable check)",
                     {"n_runs": recent.n_runs})
    return Check("Recent runs", OK, f"{recent.n_runs} solo runs, "
                 f"{recent.offers['offer_id'].nunique() if len(recent.offers) else 0} card rewards",
                 {"n_runs": recent.n_runs})


def check_new_cards(recent: RecentRuns, t: Dict[str, Any]) -> Check:
    n = max(recent.n_runs, 1)
    common = {c: k for c, k in recent.unlisted_card_runs.most_common()
              if k / n >= t["new_card_run_share"]}
    share = recent.n_runs_with_unlisted / n
    details = {"runs_with_unlisted_cards": share,
               "likely_new_cards": {c: round(k / n, 4) for c, k in common.items()}}
    if common:
        return Check("New cards", DRIFT,
                     f"{len(common)} unlisted card(s) in >= {t['new_card_run_share']:.0%} of runs, "
                     f"probably new: {', '.join(list(common)[:8])}. Add real ones to "
                     "configs/official_cards.txt and refresh configs/cards.csv", details)
    return Check("New cards", OK, f"{share:.1%} of runs have unlisted cards, none common "
                 "(mods, not new cards)", details)


def check_builds(recent: RecentRuns, ref: Dict[str, Any]) -> Check:
    new = {b: k for b, k in recent.builds.items() if b not in ref["builds"]}
    details = {"recent_builds": dict(recent.builds), "new_builds": new}
    if new:
        return Check("Game version", DRIFT,
                     f"{sum(new.values())} run(s) on versions the models weren't trained on: "
                     f"{', '.join(sorted(new))}", details)
    return Check("Game version", OK,
                 f"all on versions the models were trained on ({', '.join(sorted(recent.builds))})",
                 details)


def check_pick_rates(recent: RecentRuns, ref: Dict[str, Any], t: Dict[str, Any]) -> Check:
    o = recent.offers
    if o.empty or o["offer_id"].nunique() < t["min_rewards"]:
        return Check("Pick rates", LOW_DATA, "too few card rewards to compare pick rates")
    if ref["n_rows"] == 0:
        return Check("Pick rates", LOW_DATA, "no earlier runs to compare pick rates with")
    cards = o[~o["is_skip"]]
    stats = cards.groupby("candidate")["picked"].agg(["mean", "size"])
    stats = stats.join(ref["pick_rate"].add_prefix("ref_"), how="inner")
    # both sides need enough offers for a pick rate to mean anything
    stats = stats[(stats["size"] >= t["min_card_offers"])
                  & (stats["ref_size"] >= t["min_card_offers"])]
    stats["shift"] = stats["mean"] - stats["ref_mean"]
    shifted = stats[stats["shift"].abs() >= t["pick_rate_shift"]].sort_values(
        "shift", key=abs, ascending=False)
    skip = float(o.loc[o["is_skip"], "picked"].mean())
    skip_shift = skip - ref["skip_rate"]
    details = {"cards_compared": int(len(stats)),
               "shifted": {c: round(float(s), 4) for c, s in shifted["shift"].head(10).items()},
               "skip_rate": skip, "reference_skip_rate": ref["skip_rate"]}
    msg = (f"{len(shifted)} of {len(stats)} cards moved >= {t['pick_rate_shift']:.0%} "
           f"(skip rate {ref['skip_rate']:.1%} -> {skip:.1%})")
    if len(shifted):
        msg += ": " + ", ".join(f"{c} {s:+.0%}" for c, s in shifted["shift"].head(5).items())
    drifted = len(shifted) >= t["shifted_cards"] or abs(skip_shift) >= t["skip_rate_shift"]
    if ref["population"] == "mine":
        # The models learn from community runs, so retraining can't follow a
        # change in your own style: report it, but don't call it drift.
        return Check("Pick rates (yours)", INFO,
                     msg + (" - your picks changed" if drifted else ""), details)
    return Check("Pick rates", DRIFT if drifted else OK, msg, details)


def _reference_metric(metrics: Dict[str, Any], sources: Counter, keys: List[str]) -> Any:
    """The training-time test result for the same population ('mine' or 'test')."""
    v: Any = metrics.get("mine" if population(sources) == "mine" else "test", {})
    for k in keys:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def _calibration_gap(bins: Optional[List[Dict[str, float]]]) -> Optional[float]:
    """Mean predicted minus actual win rate, from saved calibration bins."""
    if not bins:
        return None
    n = sum(b["n"] for b in bins)
    return sum(b["n"] * (b["predicted"] - b["actual"]) for b in bins) / n


def check_pick_accuracy(recent: RecentRuns, scored: pd.DataFrame, advisor: Advisor,
                        t: Dict[str, Any]) -> Check:
    if scored.empty or scored["offer_id"].nunique() < t["min_rewards"]:
        return Check("Pick-model accuracy", LOW_DATA, "too few card rewards to measure accuracy")
    m = evaluate(scored, scored["score"].to_numpy())
    metrics = json.loads((advisor.model_dir / "metrics.json").read_text())
    ref = _reference_metric(metrics, recent.sources, ["model", "top1"])
    n = m["n_offers"]
    se = math.sqrt(m["top1"] * (1 - m["top1"]) / n)  # binomial standard error
    details = {"top1": m["top1"], "reference_top1": ref, "n_rewards": n, "std_error": se}
    if ref is None:
        return Check("Pick-model accuracy", INFO, f"top-1 {m['top1']:.1%} on {n} rewards", details)
    drop = ref - m["top1"]
    msg = f"top-1 {m['top1']:.1%} on {n} rewards vs {ref:.1%} at training (±{2 * se:.1%})"
    # Flag only drops that are both meaningful and larger than sampling noise.
    return Check("Pick-model accuracy", DRIFT if drop >= max(t["top1_drop"], 2 * se) else OK,
                 msg, details)


def check_win_model(recent: RecentRuns, scored: pd.DataFrame, advisor: Advisor,
                    t: Dict[str, Any]) -> Check:
    if advisor.win is None:
        return Check("Win-model calibration", INFO, "no win model loaded")
    out = outcome_rows(scored) if not scored.empty else scored
    if len(out) < t["min_rewards"]:
        return Check("Win-model calibration", LOW_DATA, "too few finished-run picks to check")
    y = out["run_won"].astype(bool).to_numpy()
    p = out["raw_win"].to_numpy()
    gap = float(p.mean() - y.mean())
    a = auc(y, p)
    metrics = json.loads((advisor.win_model_dir / "metrics.json").read_text())
    ref_auc = _reference_metric(metrics, recent.sources, ["outcome", "auc"])
    # The same population can already be off at training time (e.g. your runs,
    # if you win less than the community); drift is the change from that.
    ref_gap = _calibration_gap(_reference_metric(metrics, recent.sources,
                                                 ["outcome", "calibration"])) or 0.0
    details = {"mean_predicted": float(p.mean()), "actual_win_rate": float(y.mean()),
               "gap": gap, "reference_gap": ref_gap, "auc": a, "reference_auc": ref_auc,
               "n_picks": int(len(out))}
    msg = (f"predicted {p.mean():.1%} vs actual {y.mean():.1%} win rate "
           f"(off by {gap:+.1%}; {ref_gap:+.1%} at training); AUC {a:.3f}"
           + (f" vs {ref_auc:.3f} at training" if ref_auc is not None else ""))
    se = math.sqrt(max(y.mean() * (1 - y.mean()), 1e-9) / len(out))
    drifted = (abs(gap - ref_gap) >= max(t["calibration_gap"], 2 * se)
               or (ref_auc is not None and ref_auc - a >= t["auc_drop"]))
    return Check("Win-model calibration", DRIFT if drifted else OK, msg, details)


# ------------------------------------------------------------------------ main

def run_checks(paths: List[Path], cfg: Dict[str, Any], advisor: Advisor, allowed: set,
               since: Optional[int] = None) -> Dict[str, Any]:
    t = cfg["thresholds"]
    recent = collect_recent(paths, advisor, allowed, since)
    recent_ids = set(recent.offers["run_id"]) if len(recent.offers) else set()
    ref = reference_stats(Path(cfg["reference"]["offers"]), population(recent.sources),
                          recent_ids)
    scored = advisor.score(recent.offers) if len(recent.offers) else recent.offers
    checks = [check_runs(recent, t), check_new_cards(recent, t), check_builds(recent, ref),
              check_pick_rates(recent, ref, t),
              check_pick_accuracy(recent, scored, advisor, t),
              check_win_model(recent, scored, advisor, t)]
    if checks[0].status == LOW_DATA:
        verdict = "Too little recent data to judge drift."
    elif any(c.status == DRIFT for c in checks):
        verdict = "Retrain recommended: " + "; ".join(
            c.name.lower() for c in checks if c.status == DRIFT) + "."
    else:
        verdict = "Models look fine for these runs."
    return {"inputs": [str(p) for p in paths], "since": since,
            "population": ref["population"],
            "checks": [asdict(c) for c in checks], "verdict": verdict,
            "drift": any(c.status == DRIFT for c in checks)}


def report(result: Dict[str, Any]) -> str:
    since = (f" since {datetime.fromtimestamp(result['since'], timezone.utc):%Y-%m-%d %H:%M} UTC"
             if result["since"] else "")
    against = ("your earlier runs" if result["population"] == "mine"
               else "the community data the models learned from")
    lines = [f"Monitoring {', '.join(result['inputs'])}{since}, compared with {against}", ""]
    for c in result["checks"]:
        lines.append(f"  [{c['status'].upper():>8}] {c['name']}: {c['message']}")
    lines += ["", result["verdict"]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", type=Path, nargs="+", help="run files or folders (.run, .jsonl.gz)")
    ap.add_argument("--config", type=Path, default=Path("configs/monitor.yaml"))
    ap.add_argument("--card-allowlist", type=Path, default=DEFAULT_CARD_ALLOWLIST)
    when = ap.add_mutually_exclusive_group()
    when.add_argument("--since", help="only runs started on/after this date (YYYY-MM-DD)")
    when.add_argument("--since-model", action="store_true",
                      help="only runs started after the pick model was trained")
    ap.add_argument("--json", type=Path, help="also write the full result as JSON")
    ap.add_argument("--fail-on-drift", action="store_true", help="exit with status 1 on drift")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    advisor = Advisor()
    since = None
    if args.since:
        since = int(datetime.strptime(args.since, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc).timestamp())
    elif args.since_model:
        since = int((advisor.model_dir / "model.txt").stat().st_mtime)
    result = run_checks(args.inputs, cfg, advisor,
                        load_card_allowlist(args.card_allowlist), since)
    print(report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=float))
    if args.fail_on_drift and result["drift"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
