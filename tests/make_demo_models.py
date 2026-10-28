"""Train tiny pick and win models on synthetic data, for container smoke tests.

    PYTHONPATH=src:tests python tests/make_demo_models.py demo_models

Writes demo_models/card_pick and demo_models/win_prob (the layout the Docker
image expects under /app/models). In the synthetic data the card GOOD is
always taken, so /rank should suggest it.
"""

import shutil
import sys
import tempfile
from pathlib import Path

from sts2_advisor.train import run as run_pick
from sts2_advisor.win import run as run_win
from test_train import config, synthetic_offers
from test_win import win_config


def main(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic_offers().to_parquet(tmp / "offers.parquet")
        run_pick(config(tmp))
        run_win(win_config(tmp), pick_model=tmp / "model")
        out.mkdir(parents=True, exist_ok=True)
        for src, dst in (("model", "card_pick"), ("win", "win_prob")):
            shutil.rmtree(out / dst, ignore_errors=True)
            shutil.copytree(tmp / src, out / dst)
    print(f"Wrote demo models to {out}/card_pick and {out}/win_prob")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "demo_models"))
