import sys

import pytest

from sts2_advisor import screen
from sts2_advisor.screen import TextBox, card_pool, find_offered, match_card

POOL = ["SHRUG_IT_OFF", "ANGER", "HEADBUTT", "FIGHT_ME", "ASCENDERS_BANE", "BASH", "SHIV"]


@pytest.mark.parametrize("text, expected", [
    ("Shrug It Off", ("SHRUG_IT_OFF", False)),
    ("Shrug lt Off", ("SHRUG_IT_OFF", False)),      # 'I' read as 'l'
    ("Headbutt+", ("HEADBUTT", True)),
    ("Fight Me!", ("FIGHT_ME", False)),
    ("1 Anger", ("ANGER", False)),                   # energy cost read alongside
    ("Ascender's Bane", ("ASCENDERS_BANE", False)),
    # the small '+' is often misread
    ("Headbuttt", ("HEADBUTT", True)),
    ("Anger-", ("ANGER", True)),
    ("Shrug It Off:", ("SHRUG_IT_OFF", True)),
])
def test_match_card(text, expected):
    assert match_card(text, POOL)[:2] == expected


@pytest.mark.parametrize("text", ["Deal 6 damage.", "Gain 8 Block. Draw 1 card.", "Skip",
                                  "Add a Shiv to your hand", ""])
def test_descriptions_and_buttons_are_not_cards(text):
    assert match_card(text, POOL) is None


BANNER = TextBox("Choose a Card", 0.44, 0.80, 1.0)


def test_find_offered_reads_the_row_of_names_left_to_right():
    boxes = [
        BANNER,
        TextBox("Headbutt+", 0.71, 0.70, 1.0), TextBox("Shrug It Off", 0.11, 0.69, 1.0),
        TextBox("Anger", 0.41, 0.69, 1.0),
        TextBox("Deal 6 damage.", 0.41, 0.39, 1.0),
        TextBox("Bash", 0.50, 0.10, 1.0),  # a lone name elsewhere (e.g. a tooltip)
        TextBox("Skip", 0.40, 0.08, 1.0),
    ]
    assert find_offered(boxes, POOL) == [("SHRUG_IT_OFF", False), ("ANGER", False),
                                         ("HEADBUTT", True)]
    assert find_offered([BANNER, TextBox("Deal 6 damage.", 0.4, 0.4, 1.0)], POOL) == []


def test_no_banner_no_card_choice():
    """In a fight, the cards in your hand also sit in a row: without the
    'Choose a Card' title they mustn't be read as a reward."""
    hand = [TextBox("Anger", 0.30, 0.15, 1.0), TextBox("Bash", 0.45, 0.15, 1.0),
            TextBox("Headbutt", 0.60, 0.15, 1.0)]
    assert find_offered(hand, POOL) == []
    # names above the title (e.g. relic tooltips at the top) don't count either
    above = [BANNER, TextBox("Anger", 0.30, 0.90, 1.0), TextBox("Bash", 0.45, 0.90, 1.0)]
    assert find_offered(above, POOL) == []


def test_banner_read_with_small_errors():
    assert screen.find_banner([TextBox("Choose a Card", 0, 0.7, 1.0)]) is not None
    assert screen.find_banner([TextBox("Choose a Cardl", 0, 0.7, 1.0)]) is not None
    assert screen.find_banner([TextBox("Choose a Relic", 0, 0.7, 1.0)]) is None


def test_green_name_means_upgraded():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (400, 100), (150, 155, 160))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 180, 60], fill=(126, 230, 60))   # green text: upgraded
    d.rectangle([220, 20, 380, 60], fill=(245, 240, 225))  # cream text: not
    d.rectangle([220, 70, 380, 90], fill=(110, 210, 230))  # a cyan (Power) ribbon: not
    box = lambda x, y, h: TextBox("", x, y, 1.0, 0.45, h)  # noqa: E731
    assert screen.looks_upgraded(img, box(0.04, 0.38, 0.45))
    assert not screen.looks_upgraded(img, box(0.54, 0.38, 0.45))
    assert not screen.looks_upgraded(img, box(0.54, 0.08, 0.22))
    assert not screen.looks_upgraded(img, TextBox("", 0.04, 0.38, 1.0))  # no size known
    boxes = [BANNER, TextBox("Anger", 0.04, 0.38, 1.0, 0.45, 0.45),
             TextBox("Bash", 0.54, 0.38, 1.0, 0.45, 0.45)]
    assert find_offered(boxes, POOL, image=img) == [("ANGER", True), ("BASH", False)]


def test_card_pool_is_the_characters_cards_plus_colorless(tmp_path):
    csv = tmp_path / "cards.csv"
    csv.write_text("id,type,rarity,color,cost,is_x_cost,traits\n"
                   "ANGER,Attack,Common,ironclad,0,False,\n"
                   "BACKFLIP,Skill,Common,silent,1,False,Block\n"
                   "FINESSE,Skill,Uncommon,colorless,0,False,\n")
    assert card_pool("IRONCLAD", csv) == ["ANGER", "FINESSE"]
    assert card_pool(None, csv) == ["ANGER", "BACKFLIP", "FINESSE"]


def test_card_pool_from_the_real_card_list():
    pool = card_pool("SILENT")
    assert "BACKFLIP" in pool and "ANGER" not in pool  # Anger is Ironclad's


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS text recognition")
def test_read_text_from_an_image(tmp_path):
    pytest.importorskip("Vision")
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (900, 400), (20, 24, 40))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia Bold.ttf", 40)
    except OSError:
        pytest.skip("font not available")
    d.text((300, 20), "Choose a Card", font=font, fill=(250, 240, 220))
    for i, name in enumerate(["Shrug It Off", "Anger", "Headbutt+"]):
        d.text((40 + i * 290, 160), name, font=font, fill=(250, 240, 220))
    path = tmp_path / "reward.png"
    img.save(path)
    offered = find_offered(screen.read_text(path), POOL)
    assert offered == [("SHRUG_IT_OFF", False), ("ANGER", False), ("HEADBUTT", True)]


def test_ancient_options_read_off_the_screen():
    known = ["KALEIDOSCOPE", "ARCANE_SCROLL", "DOWSING_ROD", "PAELS_FLESH"]
    assert screen.match_option("Kaleidoscope", known) == "KALEIDOSCOPE"
    assert screen.match_option("Arcane Scrol: Add a random card", known) == "ARCANE_SCROLL"
    assert screen.match_option("Pael's Flesh", known) == "PAELS_FLESH"
    assert screen.match_option("Gain 100 gold.", known) is None
    boxes = [TextBox("Dowsing Rod", 0.3, 0.30, 1.0), TextBox("Kaleidoscope", 0.3, 0.60, 1.0),
             TextBox("Arcane Scroll - Obtain 1 random card", 0.3, 0.45, 1.0),
             TextBox("Kaleidoscope", 0.8, 0.95, 1.0)]  # repeated (e.g. a tooltip)
    assert screen.find_options(boxes, known) == ["KALEIDOSCOPE", "ARCANE_SCROLL", "DOWSING_ROD"]
