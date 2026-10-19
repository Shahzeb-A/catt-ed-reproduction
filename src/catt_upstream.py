"""Thin, audited bridge to the upstream CATT code in vendor/catt.

Upstream modules imported through here: tashkeel_tokenizer, bw2ar, utils, xer, ed, ed_pl,
transformer. None of them performs network I/O or shell execution on import. compute_der.py
runs its whole body at import time (it reads sys.argv), so it is never imported; it is run
as a subprocess on files instead (see scripts/eval_released.run_upstream_compute_der).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor" / "catt"
PINNED_SHA = (REPO / "UPSTREAM_SHA").read_text().strip()

if str(VENDOR) not in sys.path:
    # appended (not prepended) so upstream's generic module names (utils, xer, ...)
    # can never shadow the standard library or our own modules
    sys.path.append(str(VENDOR))

import torch  # noqa: E402

from tashkeel_tokenizer import TashkeelTokenizer  # noqa: E402
import ed_pl  # noqa: E402

# Released ED checkpoint architecture, taken from upstream predict_ed.py:
#   TashkeelModel(tokenizer, max_seq_len=1024, n_layers=3, learnable_pos_emb=False)
# (ed_pl.TashkeelModel defaults: d_model=512, n_heads=16, drop_prob=0.1).
# Note: upstream train_catt.py builds n_layers=6, which does NOT match the released ED weights.
ED_ARCH = dict(max_seq_len=1024, d_model=512, n_layers=3, n_heads=16, drop_prob=0.1, learnable_pos_emb=False)


def upstream_sha() -> str:
    """SHA of the checked out upstream tree; fails loudly if it is not the pinned one."""
    head = (VENDOR / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1]
        sha = (VENDOR / ".git" / ref).read_text().strip()
    else:
        sha = head
    if sha != PINNED_SHA:
        raise RuntimeError(f"vendor/catt is at {sha}, expected pinned {PINNED_SHA}")
    return sha


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def make_tokenizer() -> TashkeelTokenizer:
    return TashkeelTokenizer()


def build_model(tokenizer: TashkeelTokenizer | None = None, arch: dict | None = None):
    tokenizer = tokenizer or make_tokenizer()
    a = dict(ED_ARCH, **(arch or {}))
    return ed_pl.TashkeelModel(
        tokenizer,
        max_seq_len=a["max_seq_len"],
        d_model=a["d_model"],
        n_layers=a["n_layers"],
        n_heads=a["n_heads"],
        drop_prob=a["drop_prob"],
        learnable_pos_emb=a["learnable_pos_emb"],
    )


def safe_load_state_dict(path: str | os.PathLike) -> dict:
    """torch.load with weights_only=True only. Never weights_only=False."""
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise TypeError(f"{path}: expected a state_dict, got {type(sd)}")
    return sd


def load_released_ed(ckpt: str | os.PathLike, device: str = "cpu", arch: dict | None = None):
    """Build the ED model and load weights strictly (every key must match, no extras)."""
    tok = make_tokenizer()
    model = build_model(tok, arch)
    sd = safe_load_state_dict(ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model, tok


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
