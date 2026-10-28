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

## Tests

```bash
.venv/bin/python -m pytest
```

## Data credits

Community run data comes from [Spire Codex](https://spire-codex.com), a fan project not affiliated with Mega Crit. Slay the Spire 2 is © Mega Crit Games.
