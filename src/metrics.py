"""Independent DER / WER scorer and base letter check. Uses no upstream code.

A line is read as units: one base character plus the diacritics that follow it. Diacritics
are the eight marks U+064B..U+0652 (tanween x3, fatha, damma, kasra, shadda, sukun). Order
inside a unit is ignored (a set), so shadda+fatha == fatha+shadda, and a doubled mark counts once.

Two families of numbers come out of score_corpus:

  letters convention (the usual one): the denominator is Arabic letters only, spaces excluded.
    "no case ending" drops the last letter of every word from numerator and denominator.
  upstream convention (what CATT's compute_der.py computes): the denominator is every char
    position including spaces plus BOS and EOS per line. "no case ending" neutralises the
    letter before each space, so the last word of every line keeps its case ending.

Both are positional (letter i of ref vs letter i of hyp), which is defined only when base
letters are identical. Lines whose base letters differ are reported as altered and kept out
of the positional numbers; they are listed, never averaged away.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

MARKS = frozenset(chr(c) for c in range(0x064B, 0x0653))
SHADDA = "\u0651"

Unit = Tuple[str, frozenset]


def units(text: str) -> Tuple[List[Unit], int]:
    """Split text into (base, marks) units. Returns (units, stray_marks) where a stray mark
    is a diacritic with no letter to sit on (line start or right after a space)."""
    out: List[Unit] = []
    stray = 0
    for ch in text:
        if ch in MARKS:
            if not out or out[-1][0] == " ":
                stray += 1
                continue
            base, marks = out[-1]
            out[-1] = (base, marks | {ch})
        else:
            out.append((ch, frozenset()))
    return out, stray


def strip_marks(text: str) -> str:
    return "".join(ch for ch in text if ch not in MARKS)


def base_letters(text: str) -> str:
    return "".join(b for b, _ in units(text)[0])


def _words(us: Sequence[Unit]) -> List[List[int]]:
    """Index lists of maximal non-space runs."""
    words, cur = [], []
    for i, (b, _) in enumerate(us):
        if b == " ":
            if cur:
                words.append(cur)
                cur = []
        else:
            cur.append(i)
    if cur:
        words.append(cur)
    return words


@dataclass
class Tally:
    err: int = 0
    total: int = 0

    def rate(self) -> float:
        return 100.0 * self.err / self.total if self.total else float("nan")


@dataclass
class CorpusScore:
    lines: int = 0
    scored_lines: int = 0
    altered: List[int] = field(default_factory=list)  # indices into the corpus
    stray_marks_ref: int = 0
    stray_marks_hyp: int = 0
    multi_vowel_units_ref: int = 0  # a unit with two non-shadda marks (malformed gold)
    t: Dict[str, Tally] = field(default_factory=dict)

    def tally(self, key: str) -> Tally:
        return self.t.setdefault(key, Tally())

    def as_dict(self) -> dict:
        return {
            "lines": self.lines,
            "scored_lines": self.scored_lines,
            "altered_lines": len(self.altered),
            "altered_line_indices": self.altered,
            "stray_marks_ref": self.stray_marks_ref,
            "stray_marks_hyp": self.stray_marks_hyp,
            "multi_vowel_units_ref": self.multi_vowel_units_ref,
            "metrics_pct": {k: round(v.rate(), 4) for k, v in sorted(self.t.items())},
            "counts": {k: {"errors": v.err, "total": v.total} for k, v in sorted(self.t.items())},
        }


def score_corpus(refs: Sequence[str], hyps: Sequence[str], inputs: Sequence[str] | None = None) -> CorpusScore:
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} refs vs {len(hyps)} hyps")
    cs = CorpusScore(lines=len(refs))
    for idx, (ref, hyp) in enumerate(zip(refs, hyps)):
        ru, rs = units(ref)
        hu, hs = units(hyp)
        cs.stray_marks_ref += rs
        cs.stray_marks_hyp += hs
        cs.multi_vowel_units_ref += sum(1 for _, m in ru if len(m - {SHADDA}) > 1)
        rb = "".join(b for b, _ in ru)
        hb = "".join(b for b, _ in hu)
        altered = rb != hb
        if inputs is not None and strip_marks(inputs[idx]) != hb:
            altered = True
        if altered:
            cs.altered.append(idx)
            continue
        cs.scored_lines += 1

        wrong = [r[1] != h[1] for r, h in zip(ru, hu)]
        words = _words(ru)
        last_of_word = {w[-1] for w in words}
        before_space = {i - 1 for i, (b, _) in enumerate(ru) if b == " " and i > 0}
        letters = [i for i, (b, _) in enumerate(ru) if b != " "]

        # letters convention
        a = cs.tally("der_ce")
        a.total += len(letters)
        a.err += sum(wrong[i] for i in letters)
        a = cs.tally("der_noce")
        keep = [i for i in letters if i not in last_of_word]
        a.total += len(keep)
        a.err += sum(wrong[i] for i in keep)
        a = cs.tally("der_ce_gold_marked_only")
        keep = [i for i in letters if ru[i][1]]
        a.total += len(keep)
        a.err += sum(wrong[i] for i in keep)
        a = cs.tally("wer_ce")
        a.total += len(words)
        a.err += sum(any(wrong[i] for i in w) for w in words)
        a = cs.tally("wer_noce")
        a.total += len(words)
        a.err += sum(any(wrong[i] for i in w[:-1]) for w in words)

        # upstream convention (positional analogue of compute_der.py)
        n_pos = len(ru) + 2  # + BOS + EOS
        a = cs.tally("upstream_conv_der_ce")
        a.total += n_pos
        a.err += sum(wrong)
        a = cs.tally("upstream_conv_der_noce")
        a.total += n_pos
        a.err += sum(w for i, w in enumerate(wrong) if i not in before_space)
        a = cs.tally("upstream_conv_wer_ce")
        a.total += len(words)
        a.err += sum(any(wrong[i] for i in w) for w in words)
        a = cs.tally("upstream_conv_wer_noce")
        a.total += len(words)
        a.err += sum(any(wrong[i] for i in w if i not in before_space) for w in words)

        # compute_der.py scores DER as an edit distance over the label sequence, not position
        # by position; this reproduces that so the two can be compared
        for key, neutral in (("upstream_conv_der_ce_editdistance", set()), ("upstream_conv_der_noce_editdistance", before_space)):
            r = "".join(_label(u, i in neutral) for i, u in enumerate(ru))
            h = "".join(_label(u, i in neutral) for i, u in enumerate(hu))
            a = cs.tally(key)
            a.total += n_pos
            a.err += _edit_distance("<" + r + ">", "<" + h + ">")
    return cs


_LABELS: Dict[frozenset, str] = {}


def _label(u: Unit, neutral: bool) -> str:
    if u[0] == " ":
        return "_"
    marks = frozenset() if neutral else u[1]
    if marks not in _LABELS:
        _LABELS[marks] = chr(0x100 + len(_LABELS))
    return _LABELS[marks]


def _edit_distance(a: str, b: str) -> int:
    from kaldialign import edit_distance

    return int(edit_distance(a, b)["total"])


ALTERED_HEADER = "index\tline_no_in_split_file\tfirst_diff_char\tinput_len\toutput_len\tinput_sha1\n"


def altered_row(index: int, line_no: int, model_input: str, hyp: str) -> str:
    """One altered_lines.tsv row with no dataset text in it: where the base letters first differ and
    a hash of the input, so the line can be found in the split without committing the text."""
    a, b = base_letters(model_input), base_letters(hyp)
    k = next((p for p, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    sha = hashlib.sha1(model_input.encode("utf-8")).hexdigest()[:12]
    return f"{index}\t{line_no}\t{k}\t{len(a)}\t{len(b)}\t{sha}\n"
