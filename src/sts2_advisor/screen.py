"""Read the offered cards off the screen (macOS).

The game doesn't save a card reward's options, so this takes a screenshot
and reads the card names with macOS's built-in text recognition (Vision),
then matches what it read against the known card names for your character.

Needs ``requirements-mac.txt`` and the Screen Recording permission for your
terminal app (System Settings > Privacy & Security > Screen Recording).
"""

from __future__ import annotations

import csv
import difflib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from sts2_advisor.advise import normalize_card
from sts2_advisor.features import DEFAULT_CARD_META

MIN_SIMILARITY = 0.85  # how close a read line must be to a card name to count
ROW_TOLERANCE = 0.06   # card names on the same row differ this little in height (0-1)


@dataclass
class TextBox:
    text: str
    x: float  # left edge, 0-1 from the left of the image
    y: float  # bottom edge, 0-1 from the bottom of the image
    confidence: float
    w: float = 0.0  # width and height, 0-1 of the image's
    h: float = 0.0


def capture(path: Path, display: Optional[int] = None) -> Path:
    """Screenshot the screen (``display`` 1 = main) without the shutter sound."""
    cmd = ["screencapture", "-x", "-t", "png"] + (["-D", str(display)] if display else []) + [str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(path)


def read_text(image: Path) -> List[TextBox]:
    """Every line of text macOS can read in the image."""
    import Vision  # macOS only (pyobjc-framework-Vision)
    from Foundation import NSURL

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        NSURL.fileURLWithPath_(str(image)), None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)  # card names aren't dictionary words
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"text recognition failed: {error}")
    boxes = []
    for obs in request.results() or []:
        best = obs.topCandidates_(1)[0]
        b = obs.boundingBox()
        boxes.append(TextBox(str(best.string()), b.origin.x, b.origin.y, float(best.confidence()),
                             b.size.width, b.size.height))
    return boxes


def card_pool(character: Optional[str], cards_csv: Path = DEFAULT_CARD_META) -> List[str]:
    """Card ids a reward for this character can offer: its own cards and colorless ones."""
    with open(cards_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if character is None:
        return [r["id"] for r in rows]
    wanted = {character.lower(), "colorless"}
    return [r["id"] for r in rows if r.get("color", "").lower() in wanted]


# What the small '+' after an upgraded card's name is often read as.
PLUS_LOOKALIKES = "+tT-:;*iIlrf"


def match_card(text: str, pool: Sequence[str]) -> Optional[Tuple[str, bool, float]]:
    """The card a line of read text names, if it's close enough to one:
    ``(card id, upgraded, similarity)``. Tolerates small reading errors."""
    cleaned = re.sub(r"^\d+\s*|\s*\d+$", "", text.strip())  # an energy cost read alongside
    if not cleaned or len(cleaned) > 40:
        return None
    card_id, upgraded = normalize_card(cleaned)
    upgraded = upgraded or cleaned.endswith("+")
    # 'Claw-' or 'Clawt': the name plus a misread '+' (a real name ending in
    # a letter, like 'Blast', still reads as itself)
    if cleaned[-1] in PLUS_LOOKALIKES and (not cleaned[-1].isalpha() or card_id not in pool):
        shorter = normalize_card(cleaned[:-1].rstrip())[0]
        if shorter in pool:
            return shorter, True, 1.0
    if card_id in pool:
        return card_id, upgraded, 1.0
    best = difflib.get_close_matches(card_id, pool, n=1, cutoff=MIN_SIMILARITY)
    if not best:
        return None
    return best[0], upgraded, difflib.SequenceMatcher(None, card_id, best[0]).ratio()


GREEN_SHARE = 0.08  # share of a name's pixels that are green when it's upgraded


def looks_upgraded(image, box: TextBox) -> bool:
    """Whether a card name is drawn in green, as upgraded cards' names are.
    ``image`` is a PIL image of the whole screenshot."""
    if box.w <= 0 or box.h <= 0:
        return False
    W, H = image.size
    crop = image.convert("RGB").crop((int(box.x * W), int((1 - box.y - box.h) * H),
                                      int((box.x + box.w) * W), int((1 - box.y) * H)))
    px = list(crop.getdata())
    green = sum(1 for r, g, b in px if g >= 140 and g - r >= 40 and g - b >= 40)
    return bool(px) and green / len(px) >= GREEN_SHARE


BANNER = "CHOOSE_A_CARD"  # the title above every card reward's choice


def find_banner(boxes: Sequence[TextBox]) -> Optional[TextBox]:
    """The 'Choose a Card' title, if it's on screen."""
    for b in boxes:
        name, _ = normalize_card(b.text)
        if difflib.SequenceMatcher(None, name, BANNER).ratio() >= MIN_SIMILARITY:
            return b
    return None


def find_offered(boxes: Sequence[TextBox], pool: Sequence[str], max_cards: int = 4,
                 image=None) -> List[Tuple[str, bool]]:
    """The offered cards: card names read side by side on one row, left to
    right, below the 'Choose a Card' title. Without the title there's no card
    choice on screen (in a fight, the cards in your hand also sit in a row).
    With the screenshot as ``image`` (PIL), a green name also counts as upgraded."""
    banner = find_banner(boxes)
    if banner is None:
        return []
    hits = [(b, m) for b in boxes if b.y < banner.y and (m := match_card(b.text, pool))]
    if not hits:
        return []
    # the reward's names share a row; pick the row with the most names
    rows: List[List[Tuple[TextBox, Tuple[str, bool, float]]]] = []
    for hit in sorted(hits, key=lambda h: -h[0].y):
        for row in rows:
            if abs(row[0][0].y - hit[0].y) <= ROW_TOLERANCE:
                row.append(hit)
                break
        else:
            rows.append([hit])
    row = max(rows, key=len)
    seen, offered = set(), []
    for box, (card_id, upgraded, _) in sorted(row, key=lambda h: h[0].x):
        upgraded = upgraded or (image is not None and looks_upgraded(image, box))
        if (card_id, upgraded) not in seen:
            seen.add((card_id, upgraded))
            offered.append((card_id, upgraded))
    return offered[:max_cards]


def match_option(text: str, known: Sequence[str]) -> Optional[str]:
    """The relic a line of read text names (an ancient's option), if any. The
    name may open a longer line ('Kaleidoscope: ...')."""
    text = re.split(r"[:\-–—]\s", text.strip(), maxsplit=1)[0]
    if not text or len(text) > 40:
        return None
    name = normalize_card(text)[0]
    if name in known:
        return name
    best = difflib.get_close_matches(name, known, n=1, cutoff=MIN_SIMILARITY)
    return best[0] if best else None


def find_options(boxes: Sequence[TextBox], known: Sequence[str],
                 max_options: int = 4) -> List[str]:
    """An ancient's relics on screen, top to bottom then left to right."""
    hits = [(b, m) for b in boxes if (m := match_option(b.text, known))]
    out: List[str] = []
    for _, name in sorted(hits, key=lambda h: (-round(h[0].y, 2), h[0].x)):
        if name not in out:
            out.append(name)
    return out[:max_options]


def notify(title: str, message: str) -> None:
    """A macOS notification (silently skipped if it can't be shown)."""
    script = f"display notification {_quote(message)} with title {_quote(title)}"
    subprocess.run(["osascript", "-e", script], capture_output=True)


def _quote(s: str) -> str:
    """An AppleScript string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
