"""Retrain both models in one command and promote them only if they're no worse.

Steps:

1. Rebuild ``offers.parquet`` from your runs and the community exports.
2. Train a new pick model and win model into ``models/staging``.
3. Score the new and the current models on the same test set.
4. If neither new model is worse (within ``tolerance``), move the current
   pair to ``models/archive/<time>/`` and the new pair into place.

    python -m sts2_advisor.pipeline                  # everything
    python -m sts2_advisor.pipeline --skip-features  # reuse offers.parquet
    python -m sts2_advisor.pipeline --force          # promote regardless

The current models may have trained on runs that are now in the test set
(if the data changed), which slightly favours keeping them: the safe side.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from sts2_advisor.features import DEFAULT_CARD_META, build_offers, load_card_meta
from sts2_advisor.parse import DEFAULT_CARD_ALLOWLIST, load_card_allowlist
from sts2_advisor.train import evaluate, load_model, load_offers, predict, split_data
from sts2_advisor.train import run as train_pick
from sts2_advisor.win import deck_sensitivity, outcome_metrics
from sts2_advisor.win import run as train_win

log = logging.getLogger(__name__)

PAIR = ("card_pick", "win_prob")


def _load_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def score_pair(parts: Dict[str, Any], pick_dir: Path, win_dir: Path,
               meta: Optional[Dict[str, dict]] = None) -> Optional[Dict[str, float]]:
    """Test-set metrics for one pick/win model pair; None if the pair doesn't exist."""
    if not ((pick_dir / "model.txt").exists() and (win_dir / "model.txt").exists()):
        return None
    pick, win = load_model(pick_dir), load_model(win_dir)
    out = {"pick_top1": evaluate(parts["test"], predict(*pick, parts["test"]))["top1"]}
    if not parts["mine"].empty:
        out["pick_top1_mine"] = evaluate(parts["mine"], predict(*pick, parts["mine"]))["top1"]
    w = outcome_metrics(parts["test"], *win)
    out.update(win_auc=w["auc"], win_log_loss=w["log_loss"])
    if meta is not None:
        s = deck_sensitivity(parts["test"], *win, meta)
        out.update(deck_top_changed=s["top_pick_changed"],
                   deck_relative_change=s["mean_relative_change"])
    return out


def decide(current: Optional[Dict[str, float]], candidate: Dict[str, float],
           tolerance: Dict[str, float], force: bool = False) -> Dict[str, Any]:
    """Promote when there's nothing to replace, when forced, or when neither model got worse."""
    if current is None:
        return {"promote": True, "reason": "no current models"}
    if force:
        return {"promote": True, "reason": "forced"}
    worse = []
    if candidate["pick_top1"] < current["pick_top1"] - tolerance["pick_top1"]:
        worse.append(f"pick top-1 {candidate['pick_top1']:.2%} < {current['pick_top1']:.2%}")
    if candidate["win_auc"] < current["win_auc"] - tolerance["win_auc"]:
        worse.append(f"win AUC {candidate['win_auc']:.3f} < {current['win_auc']:.3f}")
    if worse:
        return {"promote": False, "reason": "new models are worse: " + "; ".join(worse)}
    return {"promote": True, "reason": "new models are at least as good"}


def promote(staging: Path, current: Path, archive: Path) -> Optional[Path]:
    """Archive the current pair (if any) and move the staged pair into place."""
    archived = None
    if any((current / name).exists() for name in PAIR):
        archived = archive / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived.mkdir(parents=True)
        for name in PAIR:
            if (current / name).exists():
                shutil.move(str(current / name), str(archived / name))
    current.mkdir(parents=True, exist_ok=True)
    for name in PAIR:
        shutil.move(str(staging / name), str(current / name))
    rel = current / "win_prob" / "reliability.json"
    if rel.exists():  # point the reliability settings at the promoted pick model
        data = json.loads(rel.read_text())
        data["pick_model_dir"] = str(current / "card_pick")
        rel.write_text(json.dumps(data, indent=2))
    return archived


def _in_fresh_process(step: str, fn, *args, **kwargs):
    """Run a memory-heavy step in its own process, so everything it allocated is
    returned to the OS before the next step (Python keeps freed memory otherwise)."""
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
        try:
            return ex.submit(fn, *args, **kwargs).result()
        except BrokenProcessPool as e:
            raise RuntimeError(f"{step} was killed before finishing, most likely by running "
                               "out of memory") from e


def _build_features(inputs: List[str], offers: Path, card_meta: Path,
                    card_allowlist: Path) -> Dict[str, int]:
    stats = build_offers([Path(p) for p in inputs], offers, load_card_meta(card_meta),
                         load_card_allowlist(card_allowlist))
    return {"runs": stats.n_runs, "rewards": stats.n_offers}


def run_pipeline(cfg: Dict[str, Any], skip_features: bool = False, force: bool = False,
                 card_meta: Path = DEFAULT_CARD_META,
                 card_allowlist: Path = DEFAULT_CARD_ALLOWLIST) -> Dict[str, Any]:
    started = time.time()
    m = cfg["models"]
    current, staging, archive = Path(m["current"]), Path(m["staging"]), Path(m["archive"])
    offers = Path(cfg["offers"])
    result: Dict[str, Any] = {"steps": {}}

    if not skip_features:
        t = time.time()
        counts = _in_fresh_process("features", _build_features, cfg["inputs"], offers,
                                   card_meta, card_allowlist)
        result["steps"]["features"] = {"seconds": round(time.time() - t), **counts}

    pick_cfg = copy.deepcopy(_load_yaml(cfg["train_config"]))
    pick_cfg["data"]["offers"] = str(offers)
    pick_cfg["output"]["model_dir"] = str(staging / "card_pick")
    win_cfg = copy.deepcopy(_load_yaml(cfg["win_config"]))
    win_cfg["data"]["offers"] = str(offers)
    win_cfg["output"]["model_dir"] = str(staging / "win_prob")
    if win_cfg["split"] != pick_cfg["split"]:
        raise ValueError("train and win configs must use the same split to be comparable")

    shutil.rmtree(staging, ignore_errors=True)
    t = time.time()
    _in_fresh_process("pick model training", train_pick, pick_cfg)
    result["steps"]["pick_model"] = {"seconds": round(time.time() - t)}
    t = time.time()
    _in_fresh_process("win model training", train_win, win_cfg,
                      pick_model=staging / "card_pick", card_meta=card_meta)
    result["steps"]["win_model"] = {"seconds": round(time.time() - t)}

    t = time.time()
    parts = split_data(load_offers(offers, with_decks=True), pick_cfg)
    meta = load_card_meta(card_meta)
    result["current"] = score_pair(parts, current / "card_pick", current / "win_prob", meta)
    result["candidate"] = score_pair(parts, staging / "card_pick", staging / "win_prob", meta)
    del parts
    result["steps"]["compare"] = {"seconds": round(time.time() - t)}

    result["decision"] = decide(result["current"], result["candidate"], cfg["tolerance"], force)
    if result["decision"]["promote"]:
        archived = promote(staging, current, archive)
        result["decision"]["archived_to"] = str(archived) if archived else None
    else:
        result["decision"]["kept_candidate_in"] = str(staging)
    result["seconds"] = round(time.time() - started)
    return result


def report(result: Dict[str, Any]) -> str:
    lines = ["Steps: " + ", ".join(f"{k} {v['seconds']}s" for k, v in result["steps"].items())]
    f = result["steps"].get("features")
    if f:
        lines.append(f"Features: {f['runs']} runs, {f['rewards']} card rewards")
    cur, new = result["current"], result["candidate"]
    rows = [("Pick top-1 (community test)", "pick_top1", "{:.2%}"),
            ("Pick top-1 (your runs)", "pick_top1_mine", "{:.2%}"),
            ("Win AUC (community test)", "win_auc", "{:.4f}"),
            ("Win log loss (lower is better)", "win_log_loss", "{:.4f}"),
            ("Deck swap changes top pick", "deck_top_changed", "{:.1%}"),
            ("Deck swap: relative change", "deck_relative_change", "{:.2%}")]
    lines += ["", f"  {'':<32} {'current':>9} {'new':>9}"]
    for label, key, fmt in rows:
        if key not in new:
            continue
        old = fmt.format(cur[key]) if cur and key in cur else "-"
        lines.append(f"  {label:<32} {old:>9} {fmt.format(new[key]):>9}")
    d = result["decision"]
    lines += ["", ("Promoted" if d["promote"] else "Kept current models") + f": {d['reason']}."]
    if d.get("archived_to"):
        lines.append(f"Previous models archived to {d['archived_to']}")
    if d.get("kept_candidate_in"):
        lines.append(f"New models left in {d['kept_candidate_in']} for inspection "
                     "(rerun with --force to promote them)")
    lines.append(f"Total {result['seconds'] // 60} min {result['seconds'] % 60} s")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    ap.add_argument("--skip-features", action="store_true", help="reuse offers.parquet")
    ap.add_argument("--force", action="store_true", help="promote the new models regardless")
    ap.add_argument("--json", type=Path, help="also write the result as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    result = run_pipeline(yaml.safe_load(args.config.read_text()),
                          skip_features=args.skip_features, force=args.force)
    print(report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
