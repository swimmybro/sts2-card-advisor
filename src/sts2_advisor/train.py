"""Train the card-pick model and compare it with simple baselines.

Each card reward is a group of candidate rows (the offered cards plus SKIP);
exactly one row has ``picked = True``. Every method scores the rows and is
judged on whether its top-scored row is the one the player picked.

Usage::

    python -m sts2_advisor.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from sts2_advisor.features import split_by_group

log = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "ascension", "act", "floor", "hp_frac", "gold", "n_relics",
    "deck_size", "deck_attacks", "deck_skills", "deck_powers", "deck_curses",
    "n_offered", "cand_cost", "cand_upgraded", "cand_copies_in_deck", "is_skip",
]
CATEGORICAL_FEATURES = ["character", "candidate", "cand_type", "cand_rarity", "room_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------- data

def load_offers(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Rows of one reward must be contiguous for the ranker's group sizes.
    return df.sort_values(["offer_id", "is_skip"], kind="stable").reset_index(drop=True)


def split_data(df: pd.DataFrame, cfg: Dict[str, Any]
               ) -> Dict[str, pd.DataFrame]:
    """Return train / valid / test (community) and mine (your runs) frames."""
    s = cfg["split"]
    mine = df[df["source"] == "mine"] if s.get("hold_out_mine", True) else df.iloc[:0]
    rest = df.drop(mine.index)
    train, test = split_by_group(rest, s["test_frac"], s["seed"])
    train, valid = split_by_group(train, s["valid_frac"], s["seed"] + 1)
    return {"train": train, "valid": valid, "test": test, "mine": mine}


def fit_categories(train: pd.DataFrame) -> Dict[str, List[str]]:
    return {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}


def to_matrix(df: pd.DataFrame, categories: Dict[str, List[str]]) -> pd.DataFrame:
    """Feature matrix with categories fixed to the training set (unseen -> NaN)."""
    X = df[FEATURES].copy()
    for c in NUMERIC_FEATURES:
        X[c] = X[c].astype("float64")
    for c in CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("offer_id", sort=False).size().to_numpy()


# ---------------------------------------------------------------------- metrics

def evaluate(df: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
    """Top-1 accuracy, mean reciprocal rank, and skip precision / recall."""
    d = pd.DataFrame({"offer_id": df["offer_id"].to_numpy(),
                      "picked": df["picked"].to_numpy(),
                      "is_skip": df["is_skip"].to_numpy(), "score": scores})
    top = d.loc[d.groupby("offer_id", sort=False)["score"].idxmax()]
    rank = d.groupby("offer_id", sort=False)["score"].rank(ascending=False, method="first")
    actual_skip = d[d["picked"]].set_index("offer_id")["is_skip"]
    pred_skip = top.set_index("offer_id")["is_skip"].reindex(actual_skip.index)
    tp = (pred_skip & actual_skip).sum()
    return {
        "top1": float(top["picked"].mean()),
        "mrr": float((1.0 / rank[d["picked"]]).mean()),
        "skip_precision": float(tp / pred_skip.sum()) if pred_skip.sum() else 0.0,
        "skip_recall": float(tp / actual_skip.sum()) if actual_skip.sum() else 0.0,
        "n_offers": int(len(top)),
    }


def evaluate_by_character(df: pd.DataFrame, scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for ch, idx in df.groupby("character").indices.items():
        out[ch] = evaluate(df.iloc[idx], scores[idx])
    return out


# -------------------------------------------------------------------- baselines

def random_scores(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(len(df))


def pick_rate_scores(train: pd.DataFrame, df: pd.DataFrame,
                     by: Tuple[str, ...] = ()) -> np.ndarray:
    """Score = how often this candidate was picked when offered in training.

    With ``by=("character",)`` rates are per character, falling back to the
    global rate for unseen pairs. Unseen candidates score 0.
    """
    keys = list(by) + ["candidate"]
    global_rate = train.groupby("candidate")["picked"].mean()
    scores = df["candidate"].map(global_rate)
    if by:
        rate = train.groupby(keys)["picked"].mean().rename("rate")
        scores = df[keys].join(rate, on=keys)["rate"].fillna(scores)
    return scores.fillna(0.0).to_numpy()


# ------------------------------------------------------------------------ model

def train_ranker(train: pd.DataFrame, valid: pd.DataFrame, cfg: Dict[str, Any]
                 ) -> Tuple[lgb.Booster, Dict[str, List[str]]]:
    m = dict(cfg["model"])
    rounds = m.pop("num_boost_round")
    early = m.pop("early_stopping_rounds")
    params = dict(m, metric="ndcg", eval_at=[1], verbose=-1, seed=cfg["split"]["seed"])
    categories = fit_categories(train)
    dtrain = lgb.Dataset(to_matrix(train, categories), label=train["picked"].astype(int),
                         group=group_sizes(train), categorical_feature=CATEGORICAL_FEATURES)
    dvalid = lgb.Dataset(to_matrix(valid, categories), label=valid["picked"].astype(int),
                         group=group_sizes(valid), reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(early, verbose=False),
                                   lgb.log_evaluation(100)])
    return booster, categories


def predict(booster: lgb.Booster, categories: Dict[str, List[str]],
            df: pd.DataFrame) -> np.ndarray:
    return booster.predict(to_matrix(df, categories), num_iteration=booster.best_iteration)


def save_model(booster: lgb.Booster, categories: Dict[str, List[str]],
               metrics: Dict[str, Any], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "model.txt"), num_iteration=booster.best_iteration)
    (model_dir / "categories.json").write_text(json.dumps(categories))
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


# ------------------------------------------------------------------------- main

def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    df = load_offers(Path(cfg["data"]["offers"]))
    parts = split_data(df, cfg)
    train, valid = parts["train"], parts["valid"]
    log.info("offers: %s", {k: v["offer_id"].nunique() for k, v in parts.items()})

    booster, categories = train_ranker(train, valid, cfg)
    fit = pd.concat([train, valid])  # baselines may use all training data
    methods = {
        "random": lambda d: random_scores(d, cfg["split"]["seed"]),
        "pick_rate": lambda d: pick_rate_scores(fit, d),
        "pick_rate_by_character": lambda d: pick_rate_scores(fit, d, ("character",)),
        "model": lambda d: predict(booster, categories, d),
    }
    metrics: Dict[str, Any] = {"best_iteration": booster.best_iteration,
                               "n_offers": {k: int(v["offer_id"].nunique())
                                            for k, v in parts.items()}}
    for split in ("test", "mine"):
        d = parts[split]
        if d.empty:
            continue
        metrics[split] = {name: evaluate(d, f(d)) for name, f in methods.items()}
    metrics["test_by_character"] = evaluate_by_character(
        parts["test"], predict(booster, categories, parts["test"]))
    gain = booster.feature_importance("gain")
    metrics["feature_importance"] = dict(sorted(
        zip(booster.feature_name(), (gain / gain.sum()).round(4).tolist()),
        key=lambda kv: -kv[1]))

    save_model(booster, categories, metrics, Path(cfg["output"]["model_dir"]))
    if cfg.get("mlflow", {}).get("enabled"):
        _log_mlflow(cfg, metrics, Path(cfg["output"]["model_dir"]))
    return metrics


def _log_mlflow(cfg: Dict[str, Any], metrics: Dict[str, Any], model_dir: Path) -> None:
    """Log to a SQLite-backed MLflow store (newer MLflow rejects plain folders)."""
    import mlflow

    m = cfg["mlflow"]
    db = Path(m["db_path"])
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db.resolve()}")
    if mlflow.get_experiment_by_name(m["experiment"]) is None:
        mlflow.create_experiment(m["experiment"],
                                 artifact_location=Path(m["artifact_dir"]).resolve().as_uri())
    mlflow.set_experiment(m["experiment"])
    with mlflow.start_run():
        mlflow.log_params({f"model.{k}": v for k, v in cfg["model"].items()})
        mlflow.log_params({f"split.{k}": v for k, v in cfg["split"].items()})
        mlflow.log_metric("best_iteration", metrics["best_iteration"])
        for split in ("test", "mine"):
            for method, vals in metrics.get(split, {}).items():
                for k, v in vals.items():
                    if k != "n_offers":
                        mlflow.log_metric(f"{split}.{method}.{k}", v)
        mlflow.log_artifacts(str(model_dir))


def report(metrics: Dict[str, Any]) -> str:
    n = metrics["n_offers"]
    lines = [f"Rewards: train {n['train']}, valid {n['valid']}, test {n['test']}, "
             f"mine {n['mine']}   (best iteration: {metrics['best_iteration']})"]
    for split, title in (("test", "Community test set"), ("mine", "Your runs (held out)")):
        if split not in metrics:
            continue
        lines += ["", f"{title}:",
                  f"  {'method':<24} {'top-1':>7} {'MRR':>7} {'skip prec':>10} {'skip rec':>9}"]
        for method, m in metrics[split].items():
            lines.append(f"  {method:<24} {m['top1']:7.1%} {m['mrr']:7.3f} "
                         f"{m['skip_precision']:10.1%} {m['skip_recall']:9.1%}")
    lines += ["", "Model top-1 by character (community test):"]
    for ch, m in sorted(metrics["test_by_character"].items()):
        lines.append(f"  {ch:<12} {m['top1']:6.1%}  ({m['n_offers']} rewards)")
    lines += ["", "Top features (share of gain):"]
    for name, share in list(metrics["feature_importance"].items())[:8]:
        lines.append(f"  {name:<22} {share:6.1%}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    metrics = run(cfg)
    print(report(metrics))
    print(f"\nSaved model to {cfg['output']['model_dir']}")


if __name__ == "__main__":
    main()
