"""Cached decoder vs the upstream decoding loop, line by line.

    python scripts/check_equivalence.py --n 60 --seed 0

Draws a length stratified sample of the held-out split (equal draws from each length decile),
decodes it with upstream ed_pl.TashkeelModel.do_tashkeel_batch and with
cached_decode.tashkeel_batch_cached using identical batches, and writes
results/equivalence_cpu/equivalence.json with the count of byte-identical outputs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import catt_upstream as cu  # noqa: E402
from cached_decode import tashkeel_batch_cached  # noqa: E402
from heldout import load_heldout  # noqa: E402

CKPT = REPO / "data/release/best_ed_mlm_ns_epoch_178.pt"
SPLIT = REPO / "data/ds/dataset/test/test.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="a multiple of 10, equal draws per length decile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.n % 10:
        raise SystemExit("--n must be a multiple of 10 (equal draws per length decile)")
    import torch

    torch.set_num_threads(args.threads)
    model, tok = cu.load_released_ed(CKPT)
    lines = load_heldout(SPLIT, tok)
    by_len = sorted(range(len(lines)), key=lambda i: len(lines[i].model_input))
    rng = random.Random(args.seed)
    deciles = [by_len[k * len(by_len) // 10 : (k + 1) * len(by_len) // 10] for k in range(10)]
    per = args.n // 10
    picked = sorted(i for d in deciles for i in rng.sample(d, per))
    # batch neighbours by length so both decoders see the same, realistic batches
    picked.sort(key=lambda i: len(lines[i].model_input))
    texts = [lines[i].model_input for i in picked]

    t0 = time.perf_counter()
    ref = model.do_tashkeel_batch(texts, args.batch_size, False)
    t_up = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = tashkeel_batch_cached(model, texts, args.batch_size)
    t_cached = time.perf_counter() - t0

    diff = [picked[k] for k, (a, b) in enumerate(zip(ref, got)) if a != b]
    out = {
        "sample_size": len(texts),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_chars": max(len(t) for t in texts),
        "identical": len(texts) - len(diff),
        "differing_indices": diff,
        "seconds_upstream_loop": round(t_up, 1),
        "seconds_cached": round(t_cached, 1),
        "upstream_sha": cu.upstream_sha(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    }
    d = REPO / "results" / "equivalence_cpu"
    d.mkdir(parents=True, exist_ok=True)
    (d / "equivalence.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    if diff:
        sys.exit(1)


if __name__ == "__main__":
    main()
