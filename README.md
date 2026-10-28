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
| Run context | `character`, `ascension`, `build_id`, `act`, `floor`, `room_type`, `hp`, `max_hp`, `hp_frac`, `gold`, `n_relics` |
| Deck before the pick | `deck_size`, `deck_attacks`, `deck_skills`, `deck_powers`, `deck_curses` |
| Candidate | `candidate`, `is_skip`, `cand_type`, `cand_rarity`, `cand_cost`, `cand_upgraded`, `cand_copies_in_deck`, `n_offered` |
| Labels | `picked`; `run_won` (blank for abandoned runs); `floors_reached` |
| Splitting | `group`: the player's `player_token` when known, otherwise the run |

How it works and what it leaves out:

- **Deck rebuild:** the deck at each reward comes from replaying every logged gain, removal and transform from the starting deck. The starting deck is the final deck with all those changes undone. Runs where this doesn't add up are dropped.
- **Dropped runs:** also any run with impossible HP (edited saves).
- **Left-out rewards:** rewards with more than one pick, which are usually two reward screens logged together.
- **Card metadata:** type, rarity and cost come from [`configs/cards.csv`](configs/cards.csv), a snapshot of Spire Codex's `/api/cards`.
- **Splitting:** use `features.split_by_group` for train/test splits, so no run (or player) lands on both sides.

## Tests

```bash
.venv/bin/python -m pytest
```

## Data credits

Community run data comes from [Spire Codex](https://spire-codex.com), a fan project not affiliated with Mega Crit. Slay the Spire 2 is © Mega Crit Games.
