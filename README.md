# sts2-card-advisor
MLOps project: predicts Slay the Spire 2 run outcomes and recommends card picks from my run history.

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Parsing run files

Put your `.run` files in `data/raw/`, then run:

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.parse data/raw --out data/processed
```

This prints a summary (win rates by character, most-picked cards, etc.) and writes two CSVs:

| File | One row per | Contents |
| --- | --- | --- |
| `data/processed/runs.csv` | run | outcome, character, ascension, floors reached, killed by, final deck |
| `data/processed/card_choices.csv` | card offered | act, floor, room type, card, whether it was picked, whether the offer was skipped |

Multiplayer (co-op) runs are excluded by default. Pass `--include-multiplayer` to keep them; each player then gets their own row, keyed by `player_id`.

Unreadable or malformed files are skipped with a warning and listed in the summary.

## Tests

```bash
.venv/bin/python -m pytest
```
