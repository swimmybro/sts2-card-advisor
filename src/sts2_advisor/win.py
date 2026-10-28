"""Win-probability model: rank a reward's options by predicted chance of winning.

Each row of ``offers.parquet`` describes one option at a reward (a card or
SKIP) together with the state just before the pick. The model is trained on
the options players actually *picked*, labelled with whether the run was won
(abandoned runs are left out). To rank a reward, every option's row is
scored: "if you took this, how likely is the run to be won?"

This learns from outcomes rather than imitating choices, but it is still
correlational: stronger players both win more and pick differently.

Usage::

    python -m sts2_advisor.win --config configs/win.yaml --pick-model models/card_pick
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from sts2_advisor.train import (
    CATEGORICAL_FEATURES, fit_categories, load_model, load_offers, log_to_mlflow,
    predict, save_model, split_data, to_matrix, top_picks,
)

log = logging.getLogger(__name__)


def outcome_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The option each player actually picked, for runs that were finished."""
    return df[df["picked"].to_numpy() & df["run_won"].notna().to_numpy()]


# ------------------------------------------------------------------------ model

def train_win_model(train: pd.DataFrame, valid: pd.DataFrame, cfg: Dict[str, Any]
                    ) -> Tuple[lgb.Booster, Dict[str, List[str]]]:
    m = dict(cfg["model"])
    rounds = m.pop("num_boost_round")
    early = m.pop("early_stopping_rounds")
    params = dict(m, verbose=-1, seed=cfg["split"]["seed"])
    tr, va = outcome_rows(train), outcome_rows(valid)
    categories = fit_categories(tr)
    dtrain = lgb.Dataset(to_matrix(tr, categories), label=tr["run_won"].astype(int),
                         categorical_feature=CATEGORICAL_FEATURES)
    dvalid = lgb.Dataset(to_matrix(va, categories), label=va["run_won"].astype(int),
                         reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(early, verbose=False),
                                   lgb.log_evaluation(100)])
    return booster, categories


# ---------------------------------------------------------------------- metrics

def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum formula (ties get mid-ranks)."""
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if not n_pos or not n_neg:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    """Predicted vs actual win rate in equal-width probability bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                        "n": int(mask.sum()), "predicted": float(p[mask].mean()),
                        "actual": float(y[mask].mean())})
    return out


def outcome_metrics(df: pd.DataFrame, booster: lgb.Booster,
                    categories: Dict[str, List[str]]) -> Dict[str, Any]:
    """How well predicted win chances match outcomes, on the options actually picked."""
    rows = outcome_rows(df)
    y = rows["run_won"].astype(bool).to_numpy()
    p = np.clip(predict(booster, categories, rows), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(rows)), "base_rate": float(y.mean()), "auc": auc(y, p),
        "log_loss": float(-np.mean(y * np.log(p) + (~y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "calibration": calibration(y, p),
    }


def ranking_metrics(df: pd.DataFrame, win_scores: np.ndarray,
                    pick_scores: Optional[np.ndarray] = None,
                    n_cards: int = 10) -> Dict[str, Any]:
    """How the win model's top option relates to players' picks (and the pick model's)."""
    top = top_picks(df, win_scores)
    spread = (pd.DataFrame({"offer_id": df["offer_id"].to_numpy(), "p": win_scores})
              .groupby("offer_id", sort=False)["p"].agg(lambda s: s.max() - s.min()))
    labelled = top[top["run_won"].notna()]
    followed = labelled["picked"].astype(bool)
    won = labelled["run_won"].astype(bool)
    actual = df[df["picked"].to_numpy()].set_index("offer_id")["candidate"]
    share_win = top["candidate"].value_counts(normalize=True)
    share_actual = actual.value_counts(normalize=True)
    shift = share_win.subtract(share_actual, fill_value=0).sort_values()
    out = {
        "agrees_with_player": float(top["picked"].mean()),
        "suggests_skip": float(top["is_skip"].mean()),
        "players_skip": float(df[df["picked"].to_numpy()]["is_skip"].mean()),
        "median_spread": float(spread.median()),
        "win_if_followed": float(won[followed].mean()) if followed.any() else float("nan"),
        "win_if_not": float(won[~followed].mean()) if (~followed).any() else float("nan"),
        "favoured_over_players": {k: round(float(v), 4)
                                  for k, v in shift[::-1].head(n_cards).items() if v > 0},
        "less_than_players": {k: round(float(v), 4)
                              for k, v in shift.head(n_cards).items() if v < 0},
    }
    if pick_scores is not None:
        pick_top = top_picks(df, pick_scores)["candidate"]
        out["agrees_with_pick_model"] = float((top["candidate"] == pick_top).mean())
    return out


# ------------------------------------------------------------------------- main

def run(cfg: Dict[str, Any], pick_model: Optional[Path] = None) -> Dict[str, Any]:
    parts = split_data(load_offers(Path(cfg["data"]["offers"])), cfg)
    booster, categories = train_win_model(parts["train"], parts["valid"], cfg)
    metrics: Dict[str, Any] = {
        "best_iteration": booster.best_iteration,
        "n_outcomes": {k: int(len(outcome_rows(v))) for k, v in parts.items()},
    }
    pick = load_model(pick_model) if pick_model else None
    for split in ("test", "mine"):
        d = parts[split]
        if d.empty:
            continue
        win_scores = predict(booster, categories, d)
        pick_scores = predict(*pick, d) if pick else None
        metrics[split] = {"outcome": outcome_metrics(d, booster, categories),
                          "ranking": ranking_metrics(d, win_scores, pick_scores)}
    gain = booster.feature_importance("gain")
    metrics["feature_importance"] = dict(sorted(
        zip(booster.feature_name(), (gain / gain.sum()).round(4).tolist()),
        key=lambda kv: -kv[1]))

    model_dir = Path(cfg["output"]["model_dir"])
    save_model(booster, categories, metrics, model_dir)
    if cfg.get("mlflow", {}).get("enabled"):
        flat = {"best_iteration": metrics["best_iteration"]}
        for split in ("test", "mine"):
            for group in ("outcome", "ranking"):
                for k, v in metrics.get(split, {}).get(group, {}).items():
                    if isinstance(v, (int, float)):
                        flat[f"{split}.{group}.{k}"] = v
        log_to_mlflow(cfg, flat, model_dir)
    return metrics


def report(metrics: Dict[str, Any]) -> str:
    n = metrics["n_outcomes"]
    lines = [f"Picks with a known outcome: train {n['train']}, valid {n['valid']}, "
             f"test {n['test']}, mine {n['mine']}   (best iteration: {metrics['best_iteration']})"]
    for split, title in (("test", "Community test set"), ("mine", "Your runs (held out)")):
        if split not in metrics:
            continue
        o, r = metrics[split]["outcome"], metrics[split]["ranking"]
        lines += ["", f"{title}:",
                  f"  Outcome prediction: AUC {o['auc']:.3f}, log loss {o['log_loss']:.3f}, "
                  f"Brier {o['brier']:.3f} (base win rate {o['base_rate']:.1%})",
                  f"  Top option agrees with the player's pick: {r['agrees_with_player']:.1%}"]
        if "agrees_with_pick_model" in r:
            lines.append(f"  Top option agrees with the pick model:    "
                         f"{r['agrees_with_pick_model']:.1%}")
        lines += [f"  Suggests SKIP {r['suggests_skip']:.1%} of the time "
                  f"(players skip {r['players_skip']:.1%})",
                  f"  Median gap between best and worst option: {r['median_spread']:.1%}",
                  f"  Run won when player followed it: {r['win_if_followed']:.1%}, "
                  f"when not: {r['win_if_not']:.1%}"]
        if split == "test":
            lines += ["  Calibration (predicted -> actual win rate):"] + [
                f"    {b['lo']:.0%}-{b['hi']:.0%}: {b['predicted']:6.1%} -> {b['actual']:6.1%}"
                f"  ({b['n']} picks)" for b in o["calibration"]]
            lines += ["  Picked as top option more often than players pick it: " + ", ".join(
                          f"{k} (+{v:.1%})" for k, v in r["favoured_over_players"].items()),
                      "  Picked as top option less often than players pick it: " + ", ".join(
                          f"{k} ({v:.1%})" for k, v in r["less_than_players"].items())]
    lines += ["", "Top features (share of gain):"]
    for name, share in list(metrics["feature_importance"].items())[:10]:
        lines.append(f"  {name:<22} {share:6.1%}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/win.yaml"))
    ap.add_argument("--pick-model", type=Path, metavar="MODEL_DIR",
                    help="pick model to compare top options against")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    metrics = run(cfg, pick_model=args.pick_model)
    print(report(metrics))
    print(f"\nSaved model to {cfg['output']['model_dir']}")


if __name__ == "__main__":
    main()
