import re
from typing import List

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def _split_long(piece: str, max_chars: int) -> List[str]:
    """Breaks an over-long sentence at clause boundaries, then at spaces as a last resort."""
    if len(piece) <= max_chars:
        return [piece]

    parts, current = [], ""
    for clause in _CLAUSE_END.split(piece):
        if current and len(current) + 1 + len(clause) > max_chars:
            parts.append(current)
            current = clause
        else:
            current = (current + " " + clause).strip()
    if current:
        parts.append(current)

    result = []
    for part in parts:
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            result.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            result.append(part)
    return result


def split_text(text: str, max_chars: int = 240) -> List[str]:
    """
    Splits text into chunks a speech model can handle. Every line of the input starts a new chunk,
    sentences are kept together while they fit, and very long sentences are cut at clause
    boundaries.
    """
    chunks = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line:
            continue

        current = ""
        for sentence in _SENTENCE_END.split(line):
            for piece in _split_long(sentence, max_chars):
                if current and len(current) + 1 + len(piece) > max_chars:
                    chunks.append(current)
                    current = piece
                else:
                    current = (current + " " + piece).strip()
        if current:
            chunks.append(current)
    return chunks
