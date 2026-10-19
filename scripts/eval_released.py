"""Evaluate a CATT ED checkpoint on the held-out split.

    python scripts/eval_released.py --config configs/eval_released_cpu.yaml

Writes results/<run_name>/{resolved_config.json, metrics.json, altered_lines.tsv} (committed)
and predictions/<run_name>/{input,ref,hyp}.txt (not committed; sha256 recorded in metrics.json).
DER/WER are computed twice: by upstream compute_der.py run unmodified as a subprocess, and by
src/metrics.py, which shares no code with upstream.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

import catt_upstream as cu  # noqa: E402
from cached_decode import tashkeel_batch_cached  # noqa: E402
from heldout import load_heldout  # noqa: E402
from metrics import base_letters, score_corpus  # noqa: E402

UPSTREAM_LINE = re.compile(r"Total (DER|WER): %([0-9.]+)|Total mismatch samples: (\d+)")


def run_upstream_compute_der(ref: Path, hyp: Path, case_ending: bool) -> dict:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONWARNINGS="ignore")
    proc = subprocess.run(
        [sys.executable, str(cu.VENDOR / "compute_der.py"), str(ref), str(hyp), "yes" if case_ending else "no"],
        capture_output=True, text=True, env=env, check=True,
    )
    out = {}
    for m in UPSTREAM_LINE.finditer(proc.stdout):
        if m.group(1):
            out[m.group(1).lower() + "_pct"] = float(m.group(2))
        else:
            out["mismatch_samples"] = int(m.group(3))
    if set(out) != {"der_pct", "wer_pct", "mismatch_samples"}:
        raise RuntimeError(f"could not parse compute_der.py output:\n{proc.stdout}\n{proc.stderr}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N lines only (smoke runs)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    import torch

    torch.set_num_threads(int(cfg["threads"]))
    torch.manual_seed(0)
    os.chdir(REPO)

    sha = {
        "checkpoint": cu.sha256_file(cfg["checkpoint"]),
        "split_file": cu.sha256_file(cfg["split_file"]),
    }
    if sha["checkpoint"] != cfg["checkpoint_sha256"]:
        raise SystemExit(f"checkpoint sha256 {sha['checkpoint']} != pinned {cfg['checkpoint_sha256']}")
    zip_path = Path(cfg["checkpoint"]).parent / "dataset.zip"
    if zip_path.exists():
        sha["dataset_zip"] = cu.sha256_file(zip_path)
        if sha["dataset_zip"] != cfg["dataset_zip_sha256"]:
            raise SystemExit("dataset.zip sha256 does not match the pinned value")

    model, tok = cu.load_released_ed(cfg["checkpoint"], device=cfg["device"])
    lines = load_heldout(cfg["split_file"], tok, cfg["max_seq_len"], cfg["threshold"])
    if args.limit:
        lines = lines[: args.limit]
    inputs = [l.model_input for l in lines]
    refs = [l.gold for l in lines]

    order = sorted(range(len(inputs)), key=lambda i: len(inputs[i])) if cfg["sort_by_length"] else list(range(len(inputs)))
    batch_in = [inputs[i] for i in order]

    # warm up kernels on two short lines (not timed, not scored)
    tashkeel_batch_cached(model, sorted(inputs, key=len)[:2], 2)

    t0 = time.perf_counter()
    if cfg["decoder"] == "cached":
        batch_out = tashkeel_batch_cached(model, batch_in, cfg["batch_size"])
    elif cfg["decoder"] == "upstream":
        batch_out = model.do_tashkeel_batch(batch_in, cfg["batch_size"], False)
    else:
        raise SystemExit(f"unknown decoder {cfg['decoder']}")
    wall = time.perf_counter() - t0
    hyps = [None] * len(inputs)
    for pos, i in enumerate(order):
        hyps[i] = batch_out[pos]

    pred_dir = REPO / "predictions" / cfg["run_name"]
    res_dir = REPO / "results" / cfg["run_name"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in (("input", inputs), ("ref", refs), ("hyp", hyps)):
        p = pred_dir / f"{name}.txt"
        p.write_text("\n".join(rows) + "\n", encoding="utf-8")
        files[name] = p

    upstream = {
        "with_case_ending": run_upstream_compute_der(files["ref"], files["hyp"], True),
        "without_case_ending": run_upstream_compute_der(files["ref"], files["hyp"], False),
    }
    indep = score_corpus(refs, hyps, inputs).as_dict()

    words = sum(len(s.split()) for s in inputs)
    chars = sum(len(s) for s in inputs)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1 << 20) if sys.platform == "darwin" else rss / 1024

    metrics = {
        "run_name": cfg["run_name"],
        "lines": len(lines),
        "words": words,
        "chars": chars,
        "upstream_compute_der": upstream,
        "independent": indep,
        "throughput": {
            "wall_seconds": round(wall, 2),
            "words_per_second": round(words / wall, 1),
            "lines_per_second": round(len(lines) / wall, 2),
            "chars_per_second": round(chars / wall, 1),
            "peak_rss_mb": round(rss_mb, 1),
            "timed": "decode only (encode+decode+detokenise), after a 2-line warmup; model load, scoring and I/O excluded",
        },
        "sha256": {**sha, **{f"predictions_{k}": cu.sha256_file(v) for k, v in files.items()}},
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "machine": platform.machine(),
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            if sys.platform == "darwin" else platform.processor(),
            "os": platform.platform(),
            "upstream_sha": cu.upstream_sha(),
            "params": cu.n_params(model),
        },
    }
    resolved = dict(cfg, limit=args.limit, config_path=str(args.config))
    (res_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n")
    (res_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    with open(res_dir / "altered_lines.tsv", "w", encoding="utf-8") as f:
        f.write("index\tline_no_in_split_file\tinput_base_letters\thyp_base_letters\n")
        for i in indep["altered_line_indices"]:
            f.write(f"{i}\t{lines[i].line_no}\t{base_letters(inputs[i])}\t{base_letters(hyps[i])}\n")

    u = upstream["with_case_ending"]
    print(json.dumps({
        "lines": len(lines), "words": words,
        "upstream DER/WER with CE": [u["der_pct"], u["wer_pct"]],
        "upstream DER/WER without CE": [upstream["without_case_ending"]["der_pct"], upstream["without_case_ending"]["wer_pct"]],
        "independent": indep["metrics_pct"],
        "altered_lines": indep["altered_lines"],
        "words_per_second": metrics["throughput"]["words_per_second"],
        "wall_seconds": metrics["throughput"]["wall_seconds"],
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
