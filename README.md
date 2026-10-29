# sts2-card-advisor
MLOps project: recommends which Slay the Spire 2 card to take (or whether to skip), ranking the options by predicted chance of winning the run given your current deck. It learns from about 220,000 community runs.

## Setup

Requires Python 3.9+. On macOS, LightGBM also needs OpenMP: `brew install libomp`.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

The trained models live in `models/` and aren't in git. Build them with [Retraining in one command](#retraining-in-one-command), or copy the `models/` folder from a machine that has them.

## Everyday use

Run these from the project folder.

**1. At a card reward (during a run):** type the offered cards, with a trailing `+` for upgraded ones (the game doesn't save them). Everything else, including whether it was a normal, elite or boss fight, comes from the game's save of your run in progress.

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.suggest "Shrug It Off" Anger Headbutt+
```

- **Clear call** (2+ points ahead): take it.
- **Slight edge:** the advisor leans one way; use your judgement.
- **Close call:** the options are effectively equal on win chance, so it suggests what most players take, and your own read of the deck is as good as the advisor's.

On macOS it can also read the cards off the screen, so you don't type anything: `--screen` reads them once, and `--auto` keeps watching and advises at every reward. See [Reading the cards off the screen](#reading-the-cards-off-the-screen-macos).

See [What should I pick?](#what-should-i-pick-during-a-run) and [How much to trust the advice](#how-much-to-trust-the-advice).

**2. After a run:** keep the watcher running in a second terminal. It copies each finished run into `data/raw/` and prints a review of your picks. Or review the latest run on demand:

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.watch
```

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.watch --review latest
```

In reviews, focus on the clear calls where you disagreed.

**3. After a game patch:** download the new community export into `data/external/`. Check whether the models still fit, and retrain if the monitor recommends it (about 15 minutes; new models replace the current ones only if they're at least as good):

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.monitor data/external/<new export>.jsonl.gz
```

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.pipeline
```

**Shortcut:** add an alias to your shell's startup file so `pick Anger Inflame Headbutt` works from anywhere. For bash this is `~/.bash_profile`; for zsh, `~/.zshrc`. Run once:

```bash
echo 'alias pick="cd ~/Documents/Projects/sts2-card-advisor && PYTHONPATH=src .venv/bin/python -m sts2_advisor.suggest"' >> ~/.bash_profile && source ~/.bash_profile
```

Adjust the folder path if your copy lives elsewhere.

### On a Windows PC

The code is the same; the commands are written differently. In PowerShell, from the project folder:

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\pip install -r requirements.txt
```

```powershell
$env:PYTHONPATH = "src"; .venv\Scripts\python -m sts2_advisor.suggest "Shrug It Off" Anger Headbutt+
```

- **Models:** copy the `models\` folder (`card_pick`, `win_prob`) from your Mac. Retraining on the PC also works, but needs the data in `data\raw` and `data\external` and about 10 GB of free memory.
- **No extra install:** LightGBM's Windows build includes OpenMP, so you don't need `libomp`.
- **The game's save folder** is found automatically. The tools look in `%APPDATA%\SlayTheSpire2\steam\<steam-id>\profile*\saves\` (Windows), `~/Library/Application Support/SlayTheSpire2/…` (macOS) and `~/.local/share/SlayTheSpire2/…` (Linux). The macOS location is confirmed; the Windows one follows the same layout. If the tools say they can't find it, locate the folder holding `current_run.save` and the `history` folder, and point them at it:

  ```powershell
  $env:STS2_HISTORY_DIR = "C:\path\to\SlayTheSpire2\steam\<steam-id>\profile1\saves\history"
  ```

  To make that permanent, use `setx STS2_HISTORY_DIR "C:\path\…\history"` and open a new PowerShell window.

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
| Deck before the pick | `deck_size`, `deck_attacks`, `deck_skills`, `deck_powers`, `deck_curses`, `deck_basics`, `deck_avg_cost`, one count per card trait (`deck_exhaust`, `deck_block`, `deck_strength`, …), and deckbuilding measures: `deck_upgraded`, `deck_upgrade_density`, `deck_block_density`, `deck_cycle_time`, `deck_redraw_time` |
| Candidate | `candidate`, `is_skip`, `cand_type`, `cand_rarity`, `cand_cost`, `cand_upgraded`, `cand_copies_in_deck`, `cand_n_traits`, `cand_synergy`, `n_offered`, and how taking it changes the deck measures: `cand_d_upgrade_density`, `cand_d_block_density`, `cand_d_cycle_time`, `cand_d_redraw_time` (all 0 for `SKIP`) |
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
- **Deckbuilding measures**, after Baalorlord's [Core Deckbuilding Concepts](https://sts2.untapped.gg/en/articles/core-deckbuilding-concepts-in-slay-the-spire):
  - **Upgrade density:** upgraded cards ÷ deck size. The deck rebuild tracks logged upgrades, downgrades, and upgraded cards gained, removed or transformed; the count matches the final deck in 98.5% of runs.
  - **Block density:** cards that reduce incoming damage ÷ deck size. A card counts if it blocks, or applies Weak, Intangible, Plating or Buffer (the `defensive` column in `cards.csv`).
  - **Cycle time:** (deck size − cards drawn by draw effects) ÷ 5, the turns needed to see the whole deck.
  - **Redraw time:** the same for later shuffles, leaving out powers and Exhaust cards, which don't come back.
  - **Each option's change** to these measures: for example, a card that draws 2 shortens the cycle, while a filler card lengthens it.

  In the community data, upgrade density and cycle time both strongly predict winning, independently of each other. Block density shows no 33% sweet spot.
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

- **Training data:** the option each player actually picked at each reward, labelled with whether the run was won. Abandoned runs are left out. It uses the pick model's features (the state before the pick plus the picked option's details), plus the deck's actual cards: one copy-count column (`deck_has_<CARD>`) for each of the 250 most common cards in training decks (`features.deck_cards` in `configs/win.yaml`). Those columns let it learn card synergies, such as a card winning more often when the deck already holds a particular other card.
- **Ranking a reward:** every option, including `SKIP`, is scored with the same model ("if you take this, how likely is a win?").
- **Same test sets:** [`configs/win.yaml`](configs/win.yaml) uses the same split as `train.yaml`.
- **Report:**
  - outcome prediction: AUC, log loss, Brier score, and calibration (predicted vs actual win rate)
  - how often its top option matches the player's pick and the pick model's
  - how often it suggests skipping, and how big the gap is between a reward's best and worst options
  - which cards it favours more or less than players do
  - **deck sensitivity:** test rewards are scored again with a different real deck from a reward of the same character and act, keeping everything else the same. It reports how often the top option changes, and how much the options' win chances move relative to each other. A model that ignores the deck scores near zero.
- **The pick model's view as an input** (`features.pick_scores` in `configs/win.yaml`): players already pick synergy cards for their decks, and the pick model has learned that. Each option gets `pick_rel` (its pick score minus the reward's best) and `pick_prob` (its share of the pick model's preference within the reward), and the win model learns from outcomes how far to trust them.
  - **Avoiding over-trust:** a pick model scores its own training runs too well, so training runs are scored *out-of-fold*. The training runs are split into 3 parts, and each part is scored by a pick model trained on the other two. All other rows use the final pick model, the one the API uses.
  - **Producing the scores:** the pipeline does it automatically. By hand, train the pick model, then run `python -m sts2_advisor.train --oof-scores pick_scores.parquet`, and pass `--pick-scores pick_scores.parquet --pick-model models/card_pick` to `sts2_advisor.win`.
- **Output:** saved to `models/win_prob/` and logged to MLflow under the `win-prob` experiment.

It learns from outcomes, but it's still correlational. Stronger players both win more and pick differently, so a card can look good partly because good players take it.

### Reliability check

A win chance for an option players rarely take rests on few, unusual runs. With `--pick-model`, the win model adds a check, configured under `reliability` in `configs/win.yaml`:

- **Pick chance:** how likely players are to take each option, from the pick model's scores within the reward. A scaling factor (`temperature`) is fitted on the validation runs so these match how often players actually took each option.
- **Shrinkage:** each option's win chance is pulled toward the reward's typical outcome, meaning the average win chance weighted by how often players take each option. An option keeps `p / (p + shrink_k)` of its difference from that average, where `p` is its pick chance. With `shrink_k: 0.05`, an option taken 50% of the time keeps 91% of its difference, and one taken 2% of the time keeps 29%.
- **Flags and ties:** options taken less than `low_confidence_below` (5%) of the time are flagged. Options within `tie_margin` (1 point) of the best are reported as tied.

The settings are saved to `models/win_prob/reliability.json`, so the API applies exactly what was evaluated. The report shows the ranking metrics before and after the check.

### How much to trust the advice

A trust check (October 2026, current models) used the community test runs, which the models never trained on: 423,867 card rewards from 34,039 finished runs. It compared the win rate when players took the advisor's suggestion against the win rate when they didn't. Comparisons were made **within the same situation** (character, ascension band, act, floor band, HP band) and then averaged. The 95% ranges come from resampling whole runs, since a run's rewards share one outcome.

| Took… | Win-rate gain vs not taking it (matched) |
| --- | --- |
| The advisor's suggestion | **+4.9 pts** [4.6, 5.2] |
| The popular choice (pick model's top option) | +3.5 pts [3.2, 3.7] |
| The advisor's pick, where it disagreed with the popular choice (clear calls only) | +2.9 pts [2.1, 3.5] over taking the popular one |

The overall win rate is 39%. The gain grows with how clear the advisor says the call is, meaning the suggestion's win chance minus the next-best option's:

| Predicted margin | under 1 pt (tie) | 1–2 pts | 2–4 pts | 4–8 pts | 8+ pts |
| --- | --- | --- | --- | --- | --- |
| Observed gain from following it | +2.7 | +3.9 | +5.1 | +7.6 | +11.9 |

Following its skip suggestions showed +5.8 points, and its card suggestions +4.5.

**Reading this:**
- **Trust the clear calls,** especially those with a margin of 2 points or more.
- **Treat ties as close calls,** where your own read of the deck is as good as the advisor's. The popular choice that breaks them still showed +2.7.

**Caveats:**
- **Observational, not a controlled test.** The data has no player IDs, so player skill can't be matched, and part of the gain likely reflects stronger players making the same picks. Two things suggest the advice still carries real information: it beats the popular choice head to head, and the gain rises steadily with the predicted margin.
- **Observed gains run larger than predicted margins.** That's consistent with some skill effect, so read the margins as how clear a call is, not as exact promises.
- **Your own runs weren't checked:** about 3,500 rewards is too few for this matching.

## What should I pick? (during a run)

At a card reward, type the offered cards (a trailing `+` means upgraded):

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.suggest "Fear" "Drain Power" "Defy+"
```

Everything else comes from the game's save of your run in progress (`current_run.save`, next to the run history): character, ascension, act, floor, HP, gold, every card in your deck with its upgrades, your relics, and, while the reward screen is open, whether it was a normal, elite or boss fight. `--elite` / `--boss` are only needed if you ask after leaving the reward screen.

```
Necrobinder A5, act 2 floor 30 (monster), HP 51/73, 417 gold, 25-card deck, 10 relics

  Take: DEFY+  (clear call: +3.7 pts of win chance over the next option)

  option               win chance  players take
  DEFY+                     51.8%           71%
  FEAR                      48.1%            3%  rarely taken here: less reliable
  SKIP                      47.6%           24%
  DRAIN_POWER               45.5%            3%  rarely taken here: less reliable
```

- **The suggestion** follows the same rule as the API. It's a **clear call** at 2+ points over the next option, a **slight edge** below that, or a **close call** when options are tied, in which case it's what players take most. See [How much to trust the advice](#how-much-to-trust-the-advice).
- **Unknown cards** are listed but never suggested. Card names the models don't know (often typos) are flagged.
- **Stale saves:** if the save is more than 3 hours old, it asks whether that's really the run you're playing.
- **Floor:** read from the save. It may be one off, depending on exactly when the game saved, which barely matters; `--floor` overrides it.
- **Finding the save:** next to the history folder that `watch.py` finds, or set `--save` / `$STS2_SAVE`.
- **Co-op saves** need `--player N`.
- **The offered cards aren't in the save:** the game creates them when the reward screen opens, so you type them, or let the advisor read them off the screen (below).

### Reading the cards off the screen (macOS)

Instead of typing the cards, the advisor can read them from a screenshot, using the text recognition built into macOS.

One-time setup:

```bash
.venv/bin/pip install -r requirements-mac.txt
```

Then allow **Screen Recording** for your terminal app, in System Settings → Privacy & Security → Screen Recording, and restart the terminal.

- **`--screen`:** read the cards now. Run it while the three cards are showing, not the list of rewards:

  ```bash
  PYTHONPATH=src .venv/bin/python -m sts2_advisor.suggest --screen
  ```

- **`--auto`:** keep it running in a terminal. Whenever the save shows an open reward screen, it takes a screenshot every couple of seconds until it reads the same cards twice in a row. Then it prints the advice, pops up a notification ("Take INFLAME"), and moves on to the next reward:

  ```bash
  PYTHONPATH=src .venv/bin/python -m sts2_advisor.suggest --auto
  ```

- **`--image FILE`:** read the cards from a screenshot you've already taken. Useful for checking what it reads.

**How it works:** the recognised text is matched against the card names your character can be offered (its own cards plus colourless ones). Small reading errors are tolerated, such as "Shrug lt Off" for Shrug It Off. Card descriptions and buttons don't match any card name, so they're ignored. The reward's cards are the names in a single row below the **"Choose a Card"** title, left to right. A card counts as upgraded if its name is drawn in green or ends in `+`. The small `+` is often misread as "t", "-" or ":", so a card name followed by one of those characters also counts. Without that title it reads nothing. That matters after a save-and-quit mid-fight, when the save looks like a reward is open but the cards on screen are your hand.

`--auto` gives advice once per card choice, so a fight that gives two card rewards gets advice for both.

**When it can't read them**, you'll see "No card names found on screen". Either the card choice isn't open yet, or Screen Recording isn't allowed. Typing the cards always works.

**Limits:** only the main display is read (use `--display 2` for another), and it works on macOS only.

### Start-of-act choices (ancients)

Every act opens with an ancient offering relics, one of which you take: Neow in act 1, then one of Tezcatara, Pael, Orobas, Darv, Nonupeipe, Tanx or Vakuu. Run histories record every relic offered and the one picked, so this works like the card advisor. A pick model learns what players take, and a win model learns which choice goes with winning, given your character, ascension, act, HP, gold, relics and deck. The advice is the highest win chance, with near-ties broken by what players take most. Which ancient you're at is worked out from the relics offered.

Build the data and train (about 10 minutes, then seconds):

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.ancient build
PYTHONPATH=src .venv/bin/python -m sts2_advisor.ancient train
```

Ask at the ancient's screen by typing the relics, or by reading them off the screen:

```bash
pick --ancient Kaleidoscope "Arcane Scroll" "Dowsing Rod"
pick --ancient --screen
```

`pick --auto` covers these too once the models are trained. While you're in an act's first room, it looks for relic names on screen and advises once per choice.

**How much to trust it** (community test set, 78,766 choices from players the models never saw):

| | |
|---|---|
| Pick model names the player's choice | 53.7% |
| Win model AUC | 0.748 |
| Advice agrees with the player | 35.6% |
| Win rate, took the advice vs not (matched on character, ascension band, act and ancient) | **+6.6 pts** |
| The same for the popular choice | +0.0 pts |

The gain rises with the predicted margin: +3.0 pts for near-ties, +3.8 for margins of 1–3 pts, +7.8 for 3–6 pts and +13.1 above 6 pts. It holds in every act (act 1 +4.6, act 2 +9.1, act 3 +7.3) and at every ascension band. As with cards, this is correlation: the matching doesn't cover HP or deck, so treat the advice as a strong hint rather than proof. Compared with players, it takes Booming Conch, Phial Holster, Neow's Torment and Neow's Sacrifice more often, and Arcane Scroll, Hefty Tablet and Kaleidoscope less often.

**Not yet checked on a real ancient screen.** If `--screen` or `--auto` misses the relics, typing them always works, and a screenshot of the screen helps tune the reading.

## Card advisor API

Serve the trained model over HTTP:

```bash
.venv/bin/uvicorn --factory sts2_advisor.serve:create_app --app-dir src --port 8000
```

`MODEL_DIR` (default `models/card_pick`) and `CARD_META` (default `configs/cards.csv`) choose the pick model and card list. `WIN_MODEL_DIR` (default `models/win_prob`) points to the win model, which drives the recommendation. Without it, the API falls back to the pick model. If the win model was trained with `--pick-model`, the reliability check is applied:
- each option also gets `pick_chance`, `raw_win_chance` (before the adjustment) and `low_confidence`
- `win_chance` is the adjusted value
- `win_tied` lists the options tied for best, and is empty when one option is clearly best

Interactive docs are at http://127.0.0.1:8000/docs.

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

The response:
- **`suggestion`:** the recommended option, listed first in `ranking`. `suggestion_source` says how it was chosen:
  - **`win_model`:** the option with the highest win chance.
  - **`win_tie_popular`:** when two or more options are tied on win chance (within the tie margin, listed in `win_tied`), the tied option players take most. In close calls, win chances can't separate the options, but players' picks often reflect card synergies. For example, with poison in a Silent deck, Snakebite, Backflip and Accuracy are within a point on win chance, and players reliably take Snakebite.
  - **`pick_model`:** no win model is loaded, so it's what players take most.
- **`popular_choice`:** what players most often take in this spot, from the pick model.
- **`ranking`:** every option, best first by win chance, with its `win_chance` and its pick-model `score` (higher means players take it more; only meaningful relative to the other options).
- **Unknown cards:** cards the models never saw in training, often typos, get `known: false` and are listed in `unknown_cards`. Treat their numbers with suspicion.

### Running in Docker

The image holds the API code and card list. Your trained models are mounted in at run time:

```bash
docker build -t sts2-card-advisor .
```

```bash
docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" sts2-card-advisor
```

- `models/card_pick` is required. `models/win_prob` is optional and adds win chances.
- The image installs only [`requirements-serve.txt`](requirements-serve.txt), what the API needs at runtime. [`requirements.txt`](requirements.txt) adds MLflow and the test tools for training and development.
- The image runs as a non-root user and has a health check on `/health`.
- CI builds the image on every push, starts it with tiny demo models trained on synthetic data ([`tests/make_demo_models.py`](tests/make_demo_models.py)), and checks `/health` and `/rank`.

## Watching your runs

`watch.py` copies new runs from the game's history folder into `data/raw` and reviews your card picks in each one:

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.watch
```

- **Default:** checks every 30 seconds (`--interval`) until you stop it.
- **`--once`:** copies new runs, reviews them, and exits.
- **`--review latest`:** reviews a run already in `data/raw`. Instead of `latest`, you can pass a run ID or a file path.
- **Finding the history folder:** it's found automatically under `SlayTheSpire2/steam/<steam-id>/profile*/saves/history`, in the game's data folder on macOS, Windows or Linux (see [On a Windows PC](#on-a-windows-pc)). If you have several profiles, the most recently used one is chosen. Set `STS2_HISTORY_DIR` or pass `--history` to choose it yourself.
- **What gets copied:** only `.run` files not already in `data/raw`. Backups (`.run.backup`) are skipped, and so are files the game is still writing; they're picked up on the next check.

For each solo run, the review lists every card reward with:
- the options offered
- your pick
- the best option, chosen by the same rule as the API's `suggestion` and marked "(tie)" when the popular choice broke a tie, with the win chance of your pick against it. Picking any of the tied options counts as taking the best.
- the popular option (what players most often take)

It ends with how often you took the best and the popular option, and the rewards where the best option's win chance most exceeded your pick's. Co-op runs are copied but not reviewed.

Your own runs are held out from training, so the models are judging picks they haven't seen. Scoring uses the same code as the API ([`advise.py`](src/sts2_advisor/advise.py)).

## Monitoring for drift

`monitor.py` checks whether the models still fit recent runs, for example after a game patch.

Check a newer community export:

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.monitor data/external/runs-v0.112.0.jsonl.gz
```

Check your runs played since the models were trained:

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.monitor data/raw --since-model
```

Pass run files or folders (`.run` or `.jsonl.gz`). `--since YYYY-MM-DD` or `--since-model` limits the check to runs started after a date, or after the pick model was trained.

Recent runs are compared like with like. Community runs are compared with the community data the models learned from. Your own runs are compared with your earlier runs. Each check reports `ok`, `drift`, `low data` or `info`:

| Check | Flags drift when |
| --- | --- |
| New cards | a card missing from the allowlist appears in ≥ 1% of runs. That's probably a real new card, since mod cards each appear in only a few runs. Add it to `configs/official_cards.txt` and refresh `configs/cards.csv`. |
| Game version | runs are on a version the models weren't trained on |
| Pick rates | ≥ 5 cards' pick rates move ≥ 10 points (with ≥ 100 offers on both sides), or the skip rate moves ≥ 5 points. For your own runs this is `info` only: retraining on community data can't follow your play style. |
| Pick-model accuracy | top-1 drops ≥ 3 points below the training-time test result for the same group (community or your runs), and by more than chance variation alone could explain |
| Win-model calibration | how far average predicted win chance is from the actual win rate changes by ≥ 5 points compared with training time, or AUC drops ≥ 0.05 |

Thresholds are in [`configs/monitor.yaml`](configs/monitor.yaml). The report ends with a verdict: "models look fine", "retrain recommended" (with reasons), or "too little recent data". `--json FILE` saves the full result, and `--fail-on-drift` exits with status 1 on drift, for scheduled jobs.

The win model's chances are calibrated to community players. For a player who wins less often than average, they run high: on the author's runs, it predicts 37% against an actual 15%. That's stable rather than drift, and the ranking still works (AUC 0.79).

## Retraining in one command

When the monitor recommends retraining (after downloading a new export into `data/external/`):

```bash
PYTHONPATH=src .venv/bin/python -m sts2_advisor.pipeline
```

1. **Features:** rebuilds `data/processed/offers.parquet` from `data/raw` and `data/external`. `--skip-features` reuses the existing file.
2. **Training:** trains a new pick model and win model into `models/staging/`, using `configs/train.yaml` and `configs/win.yaml`. In between, it produces the out-of-fold pick scores the win model takes as input (3 extra pick models; `pick_score_folds` in the pipeline config). Each heavy step runs in its own process, so its memory is fully released before the next.
3. **Comparison:** scores the new and the current models (`models/card_pick`, `models/win_prob`) on the same test set: pick-model top-1 on community runs and on your runs, win-model AUC and log loss, and deck sensitivity.
4. **Promotion:** if neither new model is worse beyond the tolerance in [`configs/pipeline.yaml`](configs/pipeline.yaml), the current pair moves to `models/archive/<time>/` and the new pair takes its place. The models are promoted or kept together, so the win model's reliability settings always match the pick model beside it. If the new models are worse, they stay in `models/staging/` for inspection. `--force` promotes them regardless.

To restore an archived pair, move its `card_pick` and `win_prob` folders back into `models/`.

If the data has changed, the current models may have trained on some runs that are now in the test set. That slightly favours keeping the current models, which is the safe direction.

## Tests

```bash
.venv/bin/python -m pytest
```

## Data credits

Community run data comes from [Spire Codex](https://spire-codex.com), a fan project not affiliated with Mega Crit. Slay the Spire 2 is © Mega Crit Games.
