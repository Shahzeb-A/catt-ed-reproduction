"""Which held out lines also occur in the public training split, and how the training split is sized.

    python scripts/train_overlap.py --run released_ed_heldout_cpu

Streams dataset/train/train.txt straight out of dataset.zip (nothing extracted), applies upstream
clean_text and the 5 < len <= 1024 filter, and checks the held out split against it two ways.

  whole line   the held out line's text with diacritics removed equals a training line's
               (also counted with identical diacritics)
  8 word runs  any run of 8 consecutive words of the held out line (diacritics removed) occurs
               anywhere in a training line

The diacritic ratio filter is not applied to the training side, so both checks see a superset of
what upstream training used (conservative). Subsets are then scored from the saved predictions of
<run> with upstream compute_der.py and src/metrics.py.

  no_overlap         no whole line match
  no_overlap_strict  no whole line match, at least 8 words, and no shared 8 word run
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import catt_upstream as cu  # noqa: E402
from eval_released import run_upstream_compute_der  # noqa: E402
from heldout import load_heldout  # noqa: E402
from metrics import MARKS, score_corpus, strip_marks  # noqa: E402

N = 8
_CLS = "".join(sorted(MARKS))
MARK_AFTER_LETTER = re.compile("[^ " + _CLS + "][" + _CLS + "]")


def h(s: str) -> bytes:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=10).digest()


def ngrams(words):
    return {" ".join(words[j : j + N]) for j in range(len(words) - N + 1)}


def score_subset(name: str, idx, rows, pred: Path) -> dict:
    sub = {k: [rows[k][i] for i in idx] for k in rows}
    d = pred / "subsets"
    d.mkdir(exist_ok=True)
    for k in ("ref", "hyp"):
        (d / f"{name}_{k}.txt").write_text("\n".join(sub[k]) + "\n", encoding="utf-8")
    return {
        "lines": len(idx),
        "words": sum(len(s.split()) for s in sub["input"]),
        "upstream_compute_der": {
            "with_case_ending": run_upstream_compute_der(d / f"{name}_ref.txt", d / f"{name}_hyp.txt", True),
            "without_case_ending": run_upstream_compute_der(d / f"{name}_ref.txt", d / f"{name}_hyp.txt", False),
        },
        "independent": score_corpus(sub["ref"], sub["hyp"], sub["input"]).as_dict()["metrics_pct"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="released_ed_heldout_cpu")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    args = ap.parse_args()
    pred = REPO / "predictions" / args.run
    if not (pred / "hyp.txt").exists():
        raise SystemExit(f"{pred}/hyp.txt not found, run scripts/eval_released.py first")
    tok = cu.make_tokenizer()
    lines = load_heldout(REPO / "data/ds/dataset/test/test.txt", tok)
    held_grams = set().union(*(ngrams(l.model_input.split()) for l in lines))

    exact, base, hit = set(), set(), set()
    n_raw = n_kept = n_ratio = 0
    positions = positions_ratio = words = 0
    t0 = time.perf_counter()
    with zipfile.ZipFile(REPO / "data/release/dataset.zip") as z:
        with z.open("dataset/train/train.txt") as fb:
            for raw in io.TextIOWrapper(fb, encoding="utf-8"):
                n_raw += 1
                line = tok.clean_text(raw)
                if not (5 < len(line) <= args.max_seq_len):
                    continue
                n_kept += 1
                exact.add(h(line))
                b = strip_marks(line)
                base.add(h(b))
                w = b.split()
                for j in range(len(w) - N + 1):
                    g = " ".join(w[j : j + N])
                    if g in held_grams:
                        hit.add(g)
                pos = len(b) + 2  # letters + spaces + BOS/EOS, i.e. decoder steps (approximate upstream split)
                positions += pos
                words += len(w)
                if len(MARK_AFTER_LETTER.findall(line)) / pos >= 0.6:
                    n_ratio += 1
                    positions_ratio += pos
    secs = time.perf_counter() - t0

    ov_exact = [i for i, l in enumerate(lines) if h(l.gold) in exact]
    ov_text = [i for i, l in enumerate(lines) if h(l.model_input) in base]
    ov_text_set = set(ov_text)
    no_overlap = [i for i in range(len(lines)) if i not in ov_text_set]
    shared, short = [], []
    for i in no_overlap:
        g = ngrams(lines[i].model_input.split())
        if not g:
            short.append(i)
        elif g & hit:
            shared.append(i)
    strict = [i for i in no_overlap if i not in set(shared) | set(short)]

    stats = {
        "train_raw_lines": n_raw,
        "train_lines_after_clean_and_length_filter": n_kept,
        "train_lines_after_ratio_filter_approx": n_ratio,
        "train_positions_after_length_filter": positions,
        "train_positions_after_ratio_filter_approx": positions_ratio,
        "train_words_after_length_filter": words,
        "note": "positions = letters + spaces + BOS/EOS, i.e. decoder steps per epoch; the ratio filter is approximated with a regex",
        "seconds": round(secs, 1),
    }
    (REPO / "results/train_split_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    rows = {k: (pred / f"{k}.txt").read_text(encoding="utf-8").split("\n")[:-1] for k in ("input", "ref", "hyp")}
    out = {
        "run": args.run,
        "heldout_lines": len(lines),
        "overlap_text": len(ov_text),
        "overlap_exact_diacritised": len(ov_exact),
        "no_overlap": len(no_overlap),
        "no_overlap_sharing_an_8_word_run": len(shared),
        "no_overlap_under_8_words": len(short),
        "no_overlap_strict": len(strict),
        "definitions": {
            "overlap_text": "undiacritised held out line equals an undiacritised training line (after clean_text and the length filter)",
            "overlap_exact_diacritised": "same, with identical diacritics",
            "no_overlap_strict": "no whole line match, at least 8 words, and no run of 8 consecutive words that occurs in any training line",
        },
        "scores": {
            "all": score_subset("all", list(range(len(lines))), rows, pred),
            "overlap_text": score_subset("overlap_text", ov_text, rows, pred),
            "no_overlap": score_subset("no_overlap", no_overlap, rows, pred),
            "no_overlap_strict": score_subset("no_overlap_strict", strict, rows, pred),
        },
        "indices": {
            "overlap_exact_diacritised": ov_exact,
            "overlap_text": ov_text,
            "no_overlap_sharing_an_8_word_run": shared,
            "no_overlap_under_8_words": short,
            "no_overlap_strict": strict,
        },
    }
    d = REPO / "results/heldout_overlap"
    d.mkdir(parents=True, exist_ok=True)
    (d / "overlap.json").write_text(json.dumps(out, indent=2) + "\n")
    brief = {k: v for k, v in out.items() if k not in ("indices", "definitions", "scores")}
    brief["DER_CE_upstream"] = {k: v["upstream_compute_der"]["with_case_ending"]["der_pct"] for k, v in out["scores"].items()}
    brief["DER_CE_letters"] = {k: v["independent"]["der_ce"] for k, v in out["scores"].items()}
    brief["words"] = {k: v["words"] for k, v in out["scores"].items()}
    print(json.dumps(brief, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
