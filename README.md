# sts2-card-advisor
MLOps project: predicts Slay the Spire 2 run outcomes and recommends card picks from my run history.

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Getting run data

- **Your own runs:** copy your `.run` files into `data/raw/`.
- **Community runs (optional):** download a per-version export from [Spire Codex](https://spire-codex.com/docs) into `data/external/`. `/api/exports/runs/manifest` lists the available versions with sizes and SHA-256 checksums. For example:

  ```bash
  curl -fL -o data/external/runs-v0.110.1.jsonl.gz https://spire-codex.com/exports/runs-v0.110.1.jsonl.gz
  ```

Both directories are gitignored.

## Parsing run files

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.parse data/raw data/external --out data/processed
```

Pass any mix of directories or files; `.run` and `.jsonl.gz` files are both read. This prints a summary (win rates by character, most-picked cards, etc.) and writes two Parquet files:

| File | One row per | Contents |
| --- | --- | --- |
| `data/processed/runs.parquet` | run | source, outcome, character, ascension, floors reached, killed by, final deck |
| `data/processed/card_choices.parquet` | card offered | act, floor, room type, card, whether it was picked, whether the offer was skipped |

Every row has a `source` column: `mine` for `.run` files, `codex` for exports. Codex runs also carry `player_token`, a pseudonymous ID for the submitting account.

- Duplicate runs (same `seed` and `start_time`) are kept only once. The first copy wins, so list `data/raw` before `data/external`.
- Multiplayer (co-op) runs are excluded by default. Pass `--include-multiplayer` to keep them; each player then gets their own row, keyed by `player_id`.
- Modded runs are excluded by default: any run with a non-official character, or with a card not listed in [`configs/official_cards.txt`](configs/official_cards.txt). Pass `--allow-mods` to keep them. The list comes from Spire Codex's `/api/cards` plus a few real cards the catalog lacks. After a game patch, check the summary's "modded" count: if it jumps, a new card is probably missing from the list.
- Unreadable or malformed runs are skipped with a warning and counted in the summary.
- Pass `-v` to log progress on large exports.

## Building card-pick features

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.features data/raw data/external
```

This writes `data/processed/offers.parquet`: one row per card in each post-combat card reward (monster, elite and boss rooms), plus one `SKIP` row per reward. Exactly one row per reward (`offer_id`) has `picked = True`.

| Column group | Columns |
| --- | --- |
| Run context | `character`, `ascension`, `build_id`, `act`, `floor`, `room_type`, `hp`, `max_hp`, `hp_frac`, `gold`, `n_relics`, `relics_held` (`\|`-separated IDs) |
| Deck before the pick | `deck_size`, `deck_attacks`, `deck_skills`, `deck_powers`, `deck_curses`, `deck_basics`, `deck_avg_cost`, plus one count per card trait (`deck_exhaust`, `deck_block`, `deck_strength`, …) |
| Candidate | `candidate`, `is_skip`, `cand_type`, `cand_rarity`, `cand_cost`, `cand_upgraded`, `cand_copies_in_deck`, `cand_n_traits`, `cand_synergy`, `n_offered` |
| Labels | `picked`; `run_won` (blank for abandoned runs); `floors_reached` |
| Splitting | `group`: the player's `player_token` when known, otherwise the run |

How it works and what it leaves out:

- **Deck rebuild:** the deck at each reward comes from replaying every logged gain, removal and transform from the starting deck. The starting deck is the final deck with all those changes undone. Runs where this doesn't add up are dropped.
- **Dropped runs:** also any run with impossible HP (edited saves).
- **Left-out rewards:** rewards with more than one pick, which are usually two reward screens logged together.
- **Card metadata:** type, rarity, cost and traits come from [`configs/cards.csv`](configs/cards.csv), a snapshot of Spire Codex's `/api/cards`. Refresh it after a game patch with `PYTHONPATH=src .venv/bin/python -m sts2_advisor.cards`.
- **Card traits** (see [`cards.py`](src/sts2_advisor/cards.py)) describe each card's role:
  - keywords: Exhaust, Ethereal, Retain, Innate, Sly
  - tags: Strike, Defend, Shiv, Minion, OstyAttack
  - effects applied: Strength, Vulnerable, Weak, Poison, Dexterity, Focus, Doom, Plating
  - cards it creates: Soul, Sovereign Blade
  - what it does: Block, Draw, Energy, HP loss, hits all enemies, X-cost

  `cand_synergy` counts how many deck cards share a trait with the candidate.
- **Relics:** `relics_held` lists the relics held *before* each reward. Training turns the 60 most common relics into individual 0/1 columns (`relic_<ID>`).
- **Splitting:** use `features.split_by_group` for train/test splits, so no run (or player) lands on both sides.

## Training the card-pick model

LightGBM needs OpenMP. On macOS, install it first with `brew install libomp`.

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.train --config configs/train.yaml
```

This trains a LightGBM ranker on `offers.parquet`. Within each reward it scores every candidate, including `SKIP`; the highest score is the suggested pick. Settings live in [`configs/train.yaml`](configs/train.yaml).

- **Split:** community runs are split 70/10/20 into train, early-stopping validation and test, with no run on both sides. Your own runs are never trained on; they're reported as a separate test set.
- **Baselines:** the model is compared with `random` and with each card's historical pick rate, both overall and per character.
- **Metrics:**
  - top-1: how often the top-scored row is the actual pick.
  - MRR: mean reciprocal rank of the actual pick.
  - Skip precision and recall.
- **Outputs:** saved to `models/card_pick/`, which holds `model.txt`, `categories.json` and `metrics.json`. Each run is also logged to MLflow: run details go to a SQLite database at `mlruns/mlflow.db`, and saved files to `mlruns/artifacts/`. Browse runs with `.venv/bin/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

The model learns which card players *pick*, not which pick *wins*. It matches community choices, which are not necessarily the best ones.

### Training on winning runs only

`train_filter` in the config limits which rewards the model learns from: `won_only`, and/or `min_ascension`. It applies to the training and validation data only, so models trained with different filters are scored on identical test sets. [`configs/train_winners.yaml`](configs/train_winners.yaml) trains on won runs only and compares the result with the default model:

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.train --config configs/train_winners.yaml --compare-with models/card_pick
```

The report adds:

- **Winning runs only:** a test set limited to won runs. It shows how well each method matches winners' picks.
- **`win: followed` / `win: not`:** how often runs were won when the player's pick agreed or disagreed with the method's top choice. Treat this as a hint, not proof. Strong players win more *and* pick differently, so the gap mixes skill with pick quality.
- **Comparison:** with `--compare-with`, how often the two models' top picks differ, and which cards each favours.

## Win-probability model

The pick model copies what players choose. The win model instead ranks options by how often taking them was followed by a win.

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.win --config configs/win.yaml --pick-model models/card_pick
```

- **Training data:** the option each player actually picked at each reward, labelled with whether the run was won. Abandoned runs are left out. It uses the same features as the pick model: the state before the pick plus the picked option's details.
- **Ranking a reward:** every option, including `SKIP`, is scored with the same model ("if you take this, how likely is a win?").
- **Same test sets:** [`configs/win.yaml`](configs/win.yaml) uses the same split as `train.yaml`.
- **Report:**
  - outcome prediction: AUC, log loss, Brier score, and calibration (predicted vs actual win rate)
  - how often its top option matches the player's pick and the pick model's
  - how often it suggests skipping, and how big the gap is between a reward's best and worst options
  - which cards it favours more or less than players do
- **Output:** saved to `models/win_prob/` and logged to MLflow under the `win-prob` experiment.

It learns from outcomes, but it's still correlational. Stronger players both win more and pick differently, so a card can look good partly because good players take it.

### Reliability check

A win chance for an option players rarely take rests on few, unusual runs. With `--pick-model`, the win model adds a check, configured under `reliability` in `configs/win.yaml`:

- **Pick chance:** how likely players are to take each option, from the pick model's scores within the reward. A scaling factor (`temperature`) is fitted on the validation runs so these match how often players actually took each option.
- **Shrinkage:** each option's win chance is pulled toward the reward's typical outcome, meaning the average win chance weighted by how often players take each option. An option keeps `p / (p + shrink_k)` of its difference from that average, where `p` is its pick chance. With `shrink_k: 0.05`, an option taken 50% of the time keeps 91% of its difference, and one taken 2% of the time keeps 29%.
- **Flags and ties:** options taken less than `low_confidence_below` (5%) of the time are flagged. Options within `tie_margin` (1 point) of the best are reported as tied.

The settings are saved to `models/win_prob/reliability.json`, so the API applies exactly what was evaluated. The report shows the ranking metrics before and after the check.

## Card advisor API

Serve the trained model over HTTP:

```bash
.venv/bin/uvicorn --factory sts2_advisor.serve:create_app --app-dir src --port 8000
```

`MODEL_DIR` (default `models/card_pick`) and `CARD_META` (default `configs/cards.csv`) choose the pick model and card list. `WIN_MODEL_DIR` (default `models/win_prob`) points to the optional win model. If it exists, each option also gets a `win_chance`, and the response gets a `win_suggestion`. If the win model was trained with `--pick-model`, the reliability check is applied:
- each option also gets `pick_chance`, `raw_win_chance` (before the adjustment) and `low_confidence`
- `win_chance` is the adjusted value
- `win_tied` lists the options tied for best, and is empty when one option is clearly best Interactive docs are at http://127.0.0.1:8000/docs.

- `GET /health`: confirms the server is up and shows which model is loaded.
- `POST /rank`: send the current state and the offered cards; get them back ranked, plus `SKIP`:

```bash
curl -s -X POST http://127.0.0.1:8000/rank -H 'Content-Type: application/json' -d '{
  "character": "Ironclad", "ascension": 8, "act": 1, "floor": 6, "room_type": "monster",
  "hp": 58, "max_hp": 80, "gold": 140,
  "deck": ["Strike Ironclad", "Strike Ironclad", "Defend Ironclad", "Bash+", "Pommel Strike"],
  "relics": ["Burning Blood"],
  "offered": ["Shrug It Off", "Anger", "Headbutt"]
}'
```

Card names are forgiving: `"Shrug It Off"`, `"shrug_it_off"` and `"CARD.SHRUG_IT_OFF"` all work, and a trailing `+` marks an upgraded card. `deck` is the deck *before* the pick, and `relics` is every relic you hold. Features are computed with the same code as the training data (`features.reward_rows`).

The response's `suggestion` is the top-ranked option. `ranking` lists every option with its model score; higher is better, and scores are only meaningful relative to each other. Cards the model never saw in training, often typos, get `known: false` and are listed in `unknown_cards`. Treat their scores with suspicion.

## Tests

```bash
.venv/bin/python -m pytest
```

## Data credits

Community run data comes from [Spire Codex](https://spire-codex.com), a fan project not affiliated with Mega Crit. Slay the Spire 2 is © Mega Crit Games.
