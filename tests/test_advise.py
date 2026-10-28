import numpy as np
import pandas as pd

from sts2_advisor.advise import _suggest


def options(win, score, tied):
    return pd.DataFrame({"offer_id": [0] * len(win), "candidate": list("ABCD")[:len(win)],
                         "win": win, "score": score, "tied": tied})


def suggested(rows):
    out = _suggest(rows, has_win=True)
    return out.loc[out["suggested"], "candidate"].tolist(), out["suggestion_source"].iloc[0]


def test_clear_best_by_win_chance():
    # B wins clearly, even though players prefer A
    rows = options(win=[0.30, 0.40, 0.20], score=[2.0, 0.0, 1.0], tied=[False] * 3)
    assert suggested(rows) == (["B"], "win_model")


def test_tie_broken_by_what_players_take():
    # A and B tied on win chance; players prefer B; C (not tied) is ignored
    rows = options(win=[0.401, 0.400, 0.300], score=[0.5, 1.5, 3.0], tied=[True, True, False])
    assert suggested(rows) == (["B"], "win_tie_popular")


def test_without_win_model_uses_the_pick_model():
    rows = options(win=[np.nan] * 3, score=[0.1, 0.9, 0.5], tied=[False] * 3)
    out = _suggest(rows, has_win=False)
    assert out.loc[out["suggested"], "candidate"].tolist() == ["B"]
    assert set(out["suggestion_source"]) == {"pick_model"}


def test_unknown_cards_are_never_suggested():
    # C has the best win chance but the models have never seen it (likely a typo)
    rows = options(win=[0.30, 0.35, 0.90], score=[1.0, 0.0, 2.0], tied=[False] * 3)
    rows["known"] = [True, True, False]
    assert suggested(rows) == (["B"], "win_model")


def test_one_suggestion_per_reward():
    two = pd.concat([options([0.3, 0.4], [1, 0], [False, False]),
                     options([0.5, 0.5], [0, 1], [True, True]).assign(offer_id=1)],
                    ignore_index=True)
    out = _suggest(two, has_win=True)
    assert out.groupby("offer_id")["suggested"].sum().tolist() == [1, 1]
    assert out.loc[out["suggested"], "candidate"].tolist() == ["B", "B"]
