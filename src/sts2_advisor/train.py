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
import pyarrow as pa
import pyarrow.compute as pc
import yaml

from sts2_advisor.features import TRAIT_COLUMNS, decks_path, split_by_group

log = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "ascension", "act", "floor", "hp_frac", "gold", "n_relics",
    "deck_size", "deck_attacks", "deck_skills", "deck_powers", "deck_curses",
    "deck_basics", "deck_avg_cost", *TRAIT_COLUMNS.values(),
    "deck_upgraded", "deck_upgrade_density", "deck_block_density", "deck_cycle_time",
    "deck_redraw_time",
    "n_offered", "cand_cost", "cand_upgraded", "cand_copies_in_deck",
    "cand_n_traits", "cand_synergy",
    "cand_d_upgrade_density", "cand_d_block_density", "cand_d_cycle_time",
    "cand_d_redraw_time", "is_skip",
]
CATEGORICAL_FEATURES = ["character", "candidate", "cand_type", "cand_rarity", "room_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Relics get one 0/1 column each (relic_<ID>) for the most common ones in training.
MAX_RELIC_FEATURES = 60
# The pick model's view of each option, an input to the win model (add_pick_features).
PICK_FEATURES = ["pick_rel", "pick_prob"]
EVAL_SPLITS = {"test": "Community test set",
               "test_won": "Community test set, winning runs only",
               "mine": "Your runs (held out)"}


# --------------------------------------------------------------------------- data

LOAD_COLUMNS = FEATURES + ["relics_held", "offer_id", "group", "source", "picked", "run_won"]


class DeckStore:
    """Every reward's deck, parsed once into integer arrays.

    ``cards[indptr[i]:indptr[i + 1]]`` are the card numbers (into ``vocab``)
    of reward ``i``'s deck. Holding decks this way takes ~100 MB for 2M+
    rewards, where per-row deck strings took many GB.
    """

    def __init__(self, decks: List[str]):
        lists = pc.split_pattern(pa.array(decks, pa.string()), "|")
        self.indptr = lists.offsets.to_numpy().astype(np.int64)
        encoded = pc.dictionary_encode(pc.list_flatten(lists))  # no Python strings
        self.vocab = np.asarray(encoded.dictionary.to_pylist(), dtype=object)
        self.cards = encoded.indices.to_numpy().astype(np.int32)

    def __deepcopy__(self, memo):
        # pandas deep-copies df.attrs on every filter; the store is never modified,
        # so share it instead of duplicating its arrays for each subset.
        return self

    def _tokens(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """For rows with these deck numbers: (row index, card number) per deck card."""
        starts, ends = self.indptr[ids], self.indptr[ids + 1]
        lens = ends - starts
        rows = np.repeat(np.arange(len(ids)), lens)
        pos = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)
        return rows, self.cards[pos]

    def counts(self, ids: np.ndarray, tracked: List[str]) -> np.ndarray:
        """(rows x tracked cards) copies of each tracked card in each row's deck."""
        col_of = pd.Series(np.arange(len(tracked)), index=tracked).reindex(self.vocab)
        col_of = col_of.fillna(-1).astype(int).to_numpy()
        out = np.zeros((len(ids), len(tracked)), dtype=np.float32)
        rows, cards = self._tokens(np.asarray(ids))
        col = col_of[cards]
        np.add.at(out, (rows[col >= 0], col[col >= 0]), 1.0)
        return out

    def frequency(self, ids: np.ndarray) -> pd.Series:
        """How many of these rows' decks contain each card (copies counted)."""
        _, cards = self._tokens(np.asarray(ids))
        freq = pd.Series(np.bincount(cards, minlength=len(self.vocab)), index=self.vocab)
        return freq[freq.index != ""]

    def strings(self, ids: np.ndarray) -> List[str]:
        return ["|".join(self.vocab[self.cards[self.indptr[i]:self.indptr[i + 1]]]) for i in ids]


def load_offers(path: Path, with_decks: bool = False) -> pd.DataFrame:
    """Load only what training needs, in compact types (~6x less memory).

    ``offer_id`` and ``group`` become integer codes; they're only used to
    tell rewards and players apart. With ``with_decks``, every reward's deck
    is read from the side file (if there is one) and kept as a
    :class:`DeckStore` in ``df.attrs["decks"]``, indexed by the ``offer_id`` code.
    """
    df = pd.read_parquet(path, columns=LOAD_COLUMNS)
    offer_ids = None
    for c in ("offer_id", "group"):
        # sort=True keeps the codes in string order, so row order and the
        # seeded train/test split don't depend on the file's row order.
        codes, uniques = pd.factorize(df[c], sort=True)
        df[c] = codes.astype("int32")
        if c == "offer_id":
            offer_ids = uniques
    for c in CATEGORICAL_FEATURES + ["source", "relics_held"]:
        df[c] = df[c].astype("string[pyarrow]")
    numeric = [c for c in NUMERIC_FEATURES if c != "is_skip"]  # is_skip stays bool
    df[numeric] = df[numeric].astype("float32")
    df["run_won"] = df["run_won"].astype("boolean")
    # Rows of one reward must be contiguous for the ranker's group sizes.
    df = df.sort_values(["offer_id", "is_skip"], kind="stable").reset_index(drop=True)
    if with_decks and decks_path(path).exists():
        decks = pd.read_parquet(decks_path(path))
        by_code = pd.Series(decks["deck_cards"].to_numpy(),
                            index=pd.Index(offer_ids).get_indexer(decks["offer_id"]))
        by_code = by_code[by_code.index >= 0]
        ordered = by_code.reindex(range(len(offer_ids))).fillna("")
        df.attrs["decks"] = DeckStore(ordered.tolist())
    return df


def split_data(df: pd.DataFrame, cfg: Dict[str, Any]
               ) -> Dict[str, pd.DataFrame]:
    """Return train / valid / test (community) and mine (your runs) frames.

    ``train_filter`` narrows train and valid only, so models trained with
    different filters are scored on exactly the same test sets.
    """
    s = cfg["split"]
    mine = df[df["source"] == "mine"] if s.get("hold_out_mine", True) else df.iloc[:0]
    rest = df.drop(mine.index)
    train, test = split_by_group(rest, s["test_frac"], s["seed"])
    train, valid = split_by_group(train, s["valid_frac"], s["seed"] + 1)
    f = cfg.get("train_filter") or {}
    train, valid = apply_train_filter(train, f), apply_train_filter(valid, f)
    return {"train": train, "valid": valid, "test": test, "mine": mine}


def apply_train_filter(df: pd.DataFrame, f: Dict[str, Any]) -> pd.DataFrame:
    """Keep only rewards from won runs and/or runs at or above an ascension."""
    keep = pd.Series(True, index=df.index)
    if f.get("won_only"):
        keep &= df["run_won"].fillna(False).astype(bool)
    if f.get("min_ascension") is not None:
        keep &= df["ascension"] >= f["min_ascension"]
    return df[keep]


def _split_pipe(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """('A|B', '', 'C') -> ids ['A','B','C'] and their row indices [0,0,2]."""
    lists = pc.split_pattern(pa.array(values.fillna("").astype(str), pa.string()), "|")
    flat = pc.list_flatten(lists).to_numpy(zero_copy_only=False)
    rows = pc.list_parent_indices(lists).to_numpy()
    keep = flat != ""
    return flat[keep], rows[keep]


def _distinct(values: pd.Series) -> Tuple[np.ndarray, pd.Series]:
    """Row codes into a list of distinct values (deck strings repeat across a
    reward's rows, so parsing each distinct one once is much cheaper)."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.cat.remove_unused_categories()  # a chunk uses few of the decks
        return values.cat.codes.to_numpy(), pd.Series(values.cat.categories.astype(str))
    codes, uniq = pd.factorize(values.fillna("").astype(str))
    return codes, pd.Series(uniq)


def deck_card_counts(decks: pd.Series, tracked: List[str]) -> np.ndarray:
    """(rows x tracked cards) copies of each tracked card in each row's deck."""
    codes, uniq = _distinct(decks)
    ids, rows = _split_pipe(uniq)
    col = pd.Categorical(ids, categories=tracked).codes
    per_deck = np.zeros((len(uniq) + 1, len(tracked)), dtype=np.float32)  # last row: missing
    np.add.at(per_deck, (rows[col >= 0], col[col >= 0]), 1.0)
    return per_deck[np.where(codes >= 0, codes, len(uniq))]


def fit_categories(train: pd.DataFrame, n_deck_cards: int = 0) -> Dict[str, List[str]]:
    """Categories, the most common relics and (optionally) the ``n_deck_cards``
    most common deck cards, all from the training rows."""
    cats = {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}
    relics = pd.Series(_split_pipe(train["relics_held"])[0]).value_counts()
    cats["relics"] = sorted(relics.head(MAX_RELIC_FEATURES).index.tolist())
    if n_deck_cards:
        if "deck_cards" in train:  # deck strings on the rows (small inputs, tests)
            codes, uniq = _distinct(train["deck_cards"])
            uses = np.bincount(codes[codes >= 0], minlength=len(uniq))  # rows per deck
            ids, rows = _split_pipe(uniq)
            freq = pd.Series(uses[rows]).groupby(ids).sum()
        else:
            freq = _deck_store(train).frequency(train["offer_id"].to_numpy())
        freq = freq.sort_values(ascending=False, kind="stable")
        cats["deck_cards"] = sorted(freq.head(n_deck_cards).index.tolist())
    return cats


def _deck_store(df: pd.DataFrame) -> DeckStore:
    store = df.attrs.get("decks")
    if store is None:
        raise ValueError("this model uses deck cards: pass rows with 'deck_cards', or "
                         "load offers with load_offers(..., with_decks=True)")
    return store


def to_matrix(df: pd.DataFrame, categories: Dict[str, List[str]]) -> pd.DataFrame:
    """Feature matrix with categories fixed to the training set (unseen -> NaN),
    plus a 0/1 column per tracked relic and, if the model tracks deck cards,
    a copy count per tracked card (``deck_has_<CARD>``)."""
    X = df[FEATURES].copy()
    for c in NUMERIC_FEATURES:
        X[c] = X[c].astype("float32")
    for c in CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    extra = []
    tracked = categories.get("relics", [])
    if tracked:
        ids, rows = _split_pipe(df["relics_held"])
        col = pd.Categorical(ids, categories=tracked).codes
        flags = np.zeros((len(df), len(tracked)), dtype=np.float32)
        flags[rows[col >= 0], col[col >= 0]] = 1.0
        extra.append(pd.DataFrame(flags, index=X.index, columns=[f"relic_{r}" for r in tracked]))
    if categories.get("pick_features"):
        missing = [c for c in PICK_FEATURES if c not in df]
        if missing:
            raise ValueError("this model uses the pick model's scores: add them with "
                             "add_pick_features() first")
        extra.append(df[PICK_FEATURES].astype("float32"))
    cards = categories.get("deck_cards", [])
    if cards:
        counts = (deck_card_counts(df["deck_cards"], cards) if "deck_cards" in df
                  else _deck_store(df).counts(df["offer_id"].to_numpy(), cards))
        extra.append(pd.DataFrame(counts, index=X.index,
                                  columns=[f"deck_has_{c}" for c in cards]))
    return pd.concat([X, *extra], axis=1) if extra else X


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("offer_id", sort=False).size().to_numpy()


# ---------------------------------------------------------------------- metrics

def top_picks(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """One row per reward: the top-scored candidate and whether it was the pick."""
    d = pd.DataFrame({"offer_id": df["offer_id"].to_numpy(),
                      "candidate": df["candidate"].to_numpy(),
                      "picked": df["picked"].to_numpy(),
                      "is_skip": df["is_skip"].to_numpy(),
                      "run_won": df["run_won"].to_numpy(), "score": scores})
    return d.loc[d.groupby("offer_id", sort=False)["score"].idxmax()].set_index("offer_id")


def evaluate(df: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
    """Top-1 accuracy, mean reciprocal rank, skip precision / recall, and
    the win rate of runs when the player's pick agreed / disagreed with ours.

    The win rates are correlations, not proof that a pick causes wins.
    """
    top = top_picks(df, scores)
    d = pd.DataFrame({"offer_id": df["offer_id"].to_numpy(),
                      "picked": df["picked"].to_numpy(), "score": scores})
    rank = d.groupby("offer_id", sort=False)["score"].rank(ascending=False, method="first")
    actual_skip = df[df["picked"].to_numpy()].set_index("offer_id")["is_skip"]
    pred_skip = top["is_skip"].reindex(actual_skip.index)
    tp = (pred_skip & actual_skip).sum()
    labelled = top[top["run_won"].notna()]
    won = labelled["run_won"].astype(bool)
    followed = labelled["picked"].astype(bool)
    return {
        "top1": float(top["picked"].mean()),
        "mrr": float((1.0 / rank[d["picked"]]).mean()),
        "skip_precision": float(tp / pred_skip.sum()) if pred_skip.sum() else 0.0,
        "skip_recall": float(tp / actual_skip.sum()) if actual_skip.sum() else 0.0,
        "win_if_followed": float(won[followed].mean()) if followed.any() else float("nan"),
        "win_if_not": float(won[~followed].mean()) if (~followed).any() else float("nan"),
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
    # Select the booster's own columns so models trained on fewer features still
    # work; score in chunks so wide feature matrices don't all sit in memory.
    names = booster.feature_name()
    chunk = 500_000
    out = [booster.predict(to_matrix(df.iloc[i:i + chunk], categories)[names],
                           num_iteration=booster.best_iteration)
           for i in range(0, len(df), chunk)]
    return np.concatenate(out) if out else np.zeros(0)


def save_model(booster: lgb.Booster, categories: Dict[str, List[str]],
               metrics: Dict[str, Any], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "model.txt"), num_iteration=booster.best_iteration)
    (model_dir / "categories.json").write_text(json.dumps(categories))
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_model(model_dir: Path) -> Tuple[lgb.Booster, Dict[str, List[str]]]:
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    categories = json.loads((model_dir / "categories.json").read_text())
    return booster, categories


def compare_top_picks(df: pd.DataFrame, ours: np.ndarray, theirs: np.ndarray,
                      n_cards: int = 8) -> Dict[str, Any]:
    """How often two models' top picks differ, and which picks shift most."""
    a, b = top_picks(df, ours), top_picks(df, theirs)
    share_a = a["candidate"].value_counts(normalize=True)
    share_b = b["candidate"].value_counts(normalize=True)
    shift = share_a.subtract(share_b, fill_value=0).sort_values()
    return {
        "disagree": float((a["candidate"] != b["candidate"]).mean()),
        "more_often": {k: round(float(v), 4) for k, v in shift[::-1].head(n_cards).items() if v > 0},
        "less_often": {k: round(float(v), 4) for k, v in shift.head(n_cards).items() if v < 0},
    }


# ------------------------------------------------- pick scores for the win model

def add_pick_features(df: pd.DataFrame) -> pd.DataFrame:
    """From ``pick_score`` (the pick model's score for each option), add each
    option's score relative to its reward's best (``pick_rel``) and its share of
    the pick model's preference within the reward (``pick_prob``)."""
    ids = df["offer_id"].to_numpy()
    s = pd.Series(df["pick_score"].to_numpy(dtype=np.float64), index=df.index)
    best = s.groupby(ids).transform("max")
    e = np.exp(s - best)
    df = df.assign(pick_rel=(s - best).astype("float32"),
                   pick_prob=(e / e.groupby(ids).transform("sum")).astype("float32"))
    return df


def oof_pick_scores(cfg: Dict[str, Any], final_pick_dir: Path, out: Path,
                    folds: int = 3) -> Dict[str, Any]:
    """Pick-model scores for every row, for use as a win-model input.

    Training rows are scored by pick models trained on the *other* folds of
    the training runs (out-of-fold), so their scores are as good as on unseen
    runs, not better. All other rows get the final pick model's score, which
    is what the API uses. Saved in :func:`load_offers` row order.
    """
    df = load_offers(Path(cfg["data"]["offers"]))
    parts = split_data(df, cfg)
    scores = np.full(len(df), np.nan, dtype=np.float32)
    final = load_model(final_pick_dir)
    others = df.index.difference(parts["train"].index)
    scores[others] = predict(*final, df.loc[others])

    train, valid = parts["train"], parts["valid"]
    groups = train["group"].unique()
    rng = np.random.default_rng(cfg["split"]["seed"] + 2)
    fold_of = dict(zip(rng.permutation(groups), np.arange(len(groups)) % folds))
    fold = train["group"].map(fold_of).to_numpy()
    for k in range(folds):
        booster, cats = train_ranker(train[fold != k], valid, cfg)
        held = train[fold == k]
        scores[held.index] = predict(booster, cats, held)
        log.info("pick-score fold %d/%d done", k + 1, folds)
        del booster
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pick_score": scores}).to_parquet(out)
    return {"rows": int(len(df)), "folds": folds}


def attach_pick_scores(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Add saved pick scores (same row order as :func:`load_offers`) and the
    features derived from them."""
    s = pd.read_parquet(path)["pick_score"].to_numpy()
    if len(s) != len(df):
        raise ValueError(f"{path} has {len(s)} rows but the offers have {len(df)}: "
                         "rebuild pick scores after rebuilding features")
    return add_pick_features(df.assign(pick_score=s))


# ------------------------------------------------------------------------- main

def run(cfg: Dict[str, Any], compare_with: Optional[Path] = None) -> Dict[str, Any]:
    df = load_offers(Path(cfg["data"]["offers"]))
    parts = split_data(df, cfg)
    del df  # the splits hold their own copies
    parts["test_won"] = parts["test"][parts["test"]["run_won"].fillna(False).astype(bool)]
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
                               "train_filter": cfg.get("train_filter") or {},
                               "n_offers": {k: int(v["offer_id"].nunique())
                                            for k, v in parts.items()}}
    for split in EVAL_SPLITS:
        d = parts[split]
        if d.empty:
            continue
        metrics[split] = {name: evaluate(d, f(d)) for name, f in methods.items()}
    test_scores = predict(booster, categories, parts["test"])
    metrics["test_by_character"] = evaluate_by_character(parts["test"], test_scores)
    if compare_with is not None:
        other, other_cats = load_model(compare_with)
        metrics["comparison"] = dict(
            compare_top_picks(parts["test"], test_scores,
                              predict(other, other_cats, parts["test"])),
            against=str(compare_with))
    gain = booster.feature_importance("gain")
    metrics["feature_importance"] = dict(sorted(
        zip(booster.feature_name(), (gain / max(gain.sum(), 1e-12)).round(4).tolist()),
        key=lambda kv: -kv[1]))

    save_model(booster, categories, metrics, Path(cfg["output"]["model_dir"]))
    if cfg.get("mlflow", {}).get("enabled"):
        _log_mlflow(cfg, metrics, Path(cfg["output"]["model_dir"]))
    return metrics


def _log_mlflow(cfg: Dict[str, Any], metrics: Dict[str, Any], model_dir: Path) -> None:
    flat = {"best_iteration": metrics["best_iteration"]}
    for split in EVAL_SPLITS:
        for method, vals in metrics.get(split, {}).items():
            for k, v in vals.items():
                if k != "n_offers":
                    flat[f"{split}.{method}.{k}"] = v
    if "comparison" in metrics:
        flat["comparison.disagree"] = metrics["comparison"]["disagree"]
    log_to_mlflow(cfg, flat, model_dir)


def log_to_mlflow(cfg: Dict[str, Any], flat_metrics: Dict[str, float], model_dir: Path) -> None:
    """Log config params, flat metrics and the model files to a SQLite-backed
    MLflow store (newer MLflow rejects plain folders)."""
    import mlflow

    m = cfg["mlflow"]
    db = Path(m["db_path"])
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db.resolve()}")
    if mlflow.get_experiment_by_name(m["experiment"]) is None:
        mlflow.create_experiment(m["experiment"],
                                 artifact_location=Path(m["artifact_dir"]).resolve().as_uri())
    mlflow.set_experiment(m["experiment"])
    with mlflow.start_run(run_name=m.get("run_name")):
        mlflow.log_params({f"model.{k}": v for k, v in cfg["model"].items()})
        mlflow.log_params({f"split.{k}": v for k, v in cfg["split"].items()})
        mlflow.log_params({f"train_filter.{k}": v
                           for k, v in (cfg.get("train_filter") or {}).items()})
        for k, v in flat_metrics.items():
            if v is not None and not np.isnan(v):
                mlflow.log_metric(k, v)
        mlflow.log_artifacts(str(model_dir))


def report(metrics: Dict[str, Any]) -> str:
    n = metrics["n_offers"]
    f = metrics.get("train_filter") or {}
    lines = [f"Rewards: train {n['train']}, valid {n['valid']}, test {n['test']} "
             f"({n['test_won']} from won runs), mine {n['mine']}   "
             f"(best iteration: {metrics['best_iteration']})",
             f"Training filter: {f if any(v for v in f.values()) else 'none (all runs)'}"]
    for split, title in EVAL_SPLITS.items():
        if split not in metrics:
            continue
        lines += ["", f"{title}:",
                  f"  {'method':<24} {'top-1':>7} {'MRR':>7} {'skip prec':>10} "
                  f"{'skip rec':>9} {'win: followed':>14} {'win: not':>9}"]
        for method, m in metrics[split].items():
            # Every run in test_won was won, so its win columns carry no information.
            wins = (f"{'—':>14} {'—':>9}" if split == "test_won" else
                    f"{m['win_if_followed']:14.1%} {m['win_if_not']:9.1%}")
            lines.append(f"  {method:<24} {m['top1']:7.1%} {m['mrr']:7.3f} "
                         f"{m['skip_precision']:10.1%} {m['skip_recall']:9.1%} {wins}")
    if "comparison" in metrics:
        c = metrics["comparison"]
        lines += ["", f"Compared with {c['against']} (community test set):",
                  f"  Top picks differ on {c['disagree']:.1%} of rewards",
                  "  Picked more often by this model: " + ", ".join(
                      f"{k} (+{v:.1%})" for k, v in c["more_often"].items()),
                  "  Picked less often by this model: " + ", ".join(
                      f"{k} ({v:.1%})" for k, v in c["less_often"].items())]
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
    ap.add_argument("--compare-with", type=Path, metavar="MODEL_DIR",
                    help="a saved model to compare top picks against on the test set")
    ap.add_argument("--oof-scores", type=Path, metavar="PARQUET",
                    help="instead of training, write out-of-fold pick scores for the win "
                         "model, using the trained model in output.model_dir for other rows")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    if args.oof_scores:
        oof_pick_scores(cfg, Path(cfg["output"]["model_dir"]), args.oof_scores, args.folds)
        print(f"Wrote pick scores to {args.oof_scores}")
        return
    metrics = run(cfg, compare_with=args.compare_with)
    print(report(metrics))
    print(f"\nSaved model to {cfg['output']['model_dir']}")


if __name__ == "__main__":
    main()
