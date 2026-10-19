"""Which held-out lines also occur in the public training split, and how the training split is sized.

    python scripts/train_overlap.py --run released_ed_heldout_cpu

Streams dataset/train/train.txt straight out of dataset.zip (nothing extracted), applies
upstream clean_text and the 5 < len <= 1024 filter, and hashes every line twice: the diacritised
text (exact overlap) and the text with diacritics removed (same words, any diacritics). The
diacritic-ratio filter is not applied here, so the overlap sets are a superset of what upstream
training actually used (conservative). Writes results/heldout_overlap/overlap.json and
results/train_split_stats.json, then rescores <run> on the held-out lines with no overlap.
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

import catt_upstream as cu  # noqa: E402
from heldout import load_heldout  # noqa: E402
from metrics import score_corpus, strip_marks  # noqa: E402

MARK_AFTER_LETTER = re.compile("[^ ً-ْ][ً-ْ]")
MARK = re.compile("[ً-ْ]")


def h(s: str) -> bytes:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=10).digest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="released_ed_heldout_cpu")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    args = ap.parse_args()
    tok = cu.make_tokenizer()

    exact, base = set(), set()
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
                pos = len(MARK.sub("", line)) + 2  # letters + spaces + BOS/EOS (approximate upstream split)
                positions += pos
                words += line.count(" ") + 1
                if len(MARK_AFTER_LETTER.findall(line)) / pos >= 0.6:
                    n_ratio += 1
                    positions_ratio += pos
    secs = time.perf_counter() - t0

    lines = load_heldout(REPO / "data/ds/dataset/test/test.txt", tok)
    ov_exact = [i for i, l in enumerate(lines) if h(l.gold) in exact]
    ov_base = [i for i, l in enumerate(lines) if h(l.model_input) in base]
    clean = [i for i in range(len(lines)) if i not in set(ov_base)]

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
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results/train_split_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    out = {
        "heldout_lines": len(lines),
        "overlap_exact_diacritised": len(ov_exact),
        "overlap_same_text_any_diacritics": len(ov_base),
        "no_overlap_lines": len(clean),
        "overlap_exact_indices": ov_exact,
        "overlap_text_indices": ov_base,
        "definition": "no_overlap = held-out lines whose undiacritised text never occurs as a training line (after clean_text and the length filter)",
    }
    pred = REPO / "predictions" / args.run
    if pred.exists():
        rows = {k: (pred / f"{k}.txt").read_text(encoding="utf-8").split("\n")[:-1] for k in ("input", "ref", "hyp")}
        sub = lambda k: [rows[k][i] for i in clean]  # noqa: E731
        out[f"{args.run}_on_no_overlap"] = score_corpus(sub("ref"), sub("hyp"), sub("input")).as_dict()["metrics_pct"]
        ov = lambda k: [rows[k][i] for i in ov_base]  # noqa: E731
        out[f"{args.run}_on_overlap"] = score_corpus(ov("ref"), ov("hyp"), ov("input")).as_dict()["metrics_pct"]
        out["no_overlap_words"] = sum(len(rows["input"][i].split()) for i in clean)
    d = REPO / "results/heldout_overlap"
    d.mkdir(parents=True, exist_ok=True)
    (d / "overlap.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("indices")}, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
