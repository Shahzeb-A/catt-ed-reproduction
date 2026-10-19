"""The held-out split exactly as upstream selects it.

Upstream tashkeel_dataset.TashkeelDataset reads every line, applies tokenizer.clean_text,
drops empty lines, keeps 5 < len(line) <= max_seq_len, then keeps lines whose share of
diacritised positions is >= threshold, and finally sorts by length. On dataset/test/test.txt
of the v2 release with max_seq_len=1024 and threshold=0.6 this gives 2,324 lines.
We keep file order (and the 1-based line number) instead of sorting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import catt_upstream  # noqa: F401
import bw2ar
from metrics import strip_marks


@dataclass(frozen=True)
class Line:
    line_no: int  # 1-based, in the raw file
    gold: str  # after upstream clean_text
    model_input: str  # gold with the eight diacritics removed


def load_heldout(path: str | Path, tok, max_seq_len: int = 1024, threshold: float = 0.6) -> List[Line]:
    out: List[Line] = []
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, start=1):
            gold = tok.clean_text(raw)
            if gold == "" or not (5 < len(gold) <= max_seq_len):
                continue
            pairs = tok.split_tashkeel_from_text(gold, test_match=False)
            tags = [t for _, t in pairs]
            if (len(tags) - tags.count(tok.no_tashkeel_tag)) / len(tags) < threshold:
                continue
            inp = strip_marks(gold)
            unknown = {c for c in bw2ar.transliterate_text(inp, "ar2bw") if c not in tok.letters_map}
            if unknown:
                raise ValueError(f"line {n}: input chars outside the encoder vocabulary: {sorted(unknown)}")
            out.append(Line(n, gold, inp))
    return out
