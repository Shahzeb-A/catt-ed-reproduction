"""Compare the predictions of two runs line by line.

    python scripts/compare_runs.py released_ed_heldout_cpu released_ed_heldout_cpu_upstream_decode

Both runs must have scored the same inputs (same predictions/<run>/input.txt sha256). Writes
results/equivalence_cpu/<a>__vs__<b>.json with the count of byte-identical outputs and the
indices of any line that differs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(run: str):
    m = json.loads((REPO / "results" / run / "metrics.json").read_text())
    hyp = (REPO / "predictions" / run / "hyp.txt").read_text(encoding="utf-8").split("\n")[:-1]
    return m, hyp


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    ma, ha = load(a)
    mb, hb = load(b)
    if ma["sha256"]["predictions_input"] != mb["sha256"]["predictions_input"]:
        raise SystemExit("the two runs scored different inputs")
    diff = [i for i, (x, y) in enumerate(zip(ha, hb)) if x != y]
    out = {
        "run_a": a,
        "run_b": b,
        "lines": len(ha),
        "identical": len(ha) - len(diff),
        "differing_indices": diff,
        "hyp_sha256_a": ma["sha256"]["predictions_hyp"],
        "hyp_sha256_b": mb["sha256"]["predictions_hyp"],
        "decoder_a": json.loads((REPO / "results" / a / "resolved_config.json").read_text())["decoder"],
        "decoder_b": json.loads((REPO / "results" / b / "resolved_config.json").read_text())["decoder"],
    }
    d = REPO / "results" / "equivalence_cpu"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a}__vs__{b}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
