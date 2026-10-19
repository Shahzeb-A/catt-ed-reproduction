"""Rescore saved predictions without decoding again.

    python scripts/score_predictions.py --run released_ed_heldout_cpu

Checks predictions/<run>/{input,ref,hyp}.txt against the sha256 values recorded in
results/<run>/metrics.json, then recomputes the upstream compute_der.py numbers and the
independent metrics and rewrites those two blocks plus altered_lines.tsv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import catt_upstream as cu  # noqa: E402
from eval_released import run_upstream_compute_der  # noqa: E402
from metrics import base_letters, score_corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    pred = REPO / "predictions" / args.run
    res = REPO / "results" / args.run
    m = json.loads((res / "metrics.json").read_text())
    files = {k: pred / f"{k}.txt" for k in ("input", "ref", "hyp")}
    for k, p in files.items():
        got = cu.sha256_file(p)
        if got != m["sha256"][f"predictions_{k}"]:
            raise SystemExit(f"{p} sha256 {got} does not match metrics.json")
    rows = {k: p.read_text(encoding="utf-8").split("\n")[:-1] for k, p in files.items()}
    m["upstream_compute_der"] = {
        "with_case_ending": run_upstream_compute_der(files["ref"], files["hyp"], True),
        "without_case_ending": run_upstream_compute_der(files["ref"], files["hyp"], False),
    }
    m["independent"] = score_corpus(rows["ref"], rows["hyp"], rows["input"]).as_dict()
    (res / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    cfg = json.loads((res / "resolved_config.json").read_text())
    from heldout import load_heldout

    lines = load_heldout(REPO / cfg["split_file"], cu.make_tokenizer(), cfg["max_seq_len"], cfg["threshold"])
    with open(res / "altered_lines.tsv", "w", encoding="utf-8") as f:
        f.write("index\tline_no_in_split_file\tinput_base_letters\thyp_base_letters\n")
        for i in m["independent"]["altered_line_indices"]:
            f.write(f"{i}\t{lines[i].line_no}\t{base_letters(rows['input'][i])}\t{base_letters(rows['hyp'][i])}\n")
    print(json.dumps({"upstream_compute_der": m["upstream_compute_der"], "independent": m["independent"]["metrics_pct"],
                      "altered_lines": m["independent"]["altered_lines"]}, indent=1))


if __name__ == "__main__":
    main()
