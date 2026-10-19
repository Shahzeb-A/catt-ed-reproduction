"""KV-cached greedy decoding for the released CATT encoder-decoder.

Upstream ed_pl.TashkeelModel.do_tashkeel_batch reruns the whole decoder over the full
prefix at every step, so decoding is quadratic in line length. Every decoder row only
sees positions <= itself (no-peek mask) and its pad mask only depends on tokens <= itself,
so caching each layer's self-attention keys/values plus the encoder keys/values computes
the same function. This module reimplements that loop with a cache, reusing the upstream
modules' own sublayers (w_q/w_k/w_v/w_concat, LayerNorm, FFN, LM head), and keeps
upstream's two behaviours that affect outputs:
  * the token fed back is forced to <NT> wherever the source char is a space;
  * masking is masked_fill(mask == 0, -10000), fully masked rows included.

scripts/check_equivalence.py compares it with the upstream loop line by line.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch

import catt_upstream  # noqa: F401  (puts vendor/catt on sys.path)
from ed_pl import get_batches, pad_seq


def _split(x: torch.Tensor, n_head: int) -> torch.Tensor:
    b, l, d = x.size()
    return x.view(b, l, n_head, d // n_head).transpose(1, 2)


def _concat(x: torch.Tensor) -> torch.Tensor:
    b, h, l, d = x.size()
    return x.transpose(1, 2).contiguous().view(b, l, h * d)


def _attention(mha, q_in: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # same arithmetic as upstream transformer.ScaleDotProductAttention
    q = _split(mha.w_q(q_in), mha.n_head)
    score = (q @ k.transpose(2, 3)) / math.sqrt(k.size(3))
    score = score.masked_fill(mask == 0, -10000)
    score = torch.softmax(score, dim=-1)
    return mha.w_concat(_concat(score @ v))


@torch.no_grad()
def greedy_decode(model, src: torch.Tensor) -> torch.Tensor:
    """src: [B, Ls] post-padded input ids (BOS ... EOS PAD...). Returns target ids [B, Ls]."""
    if model.training:
        raise RuntimeError("call model.eval() first; dropout is not applied here")
    tok = model.tokenizer
    tr = model.transformer
    dec = tr.decoder
    if not hasattr(dec.emb.pos_emb, "encoding"):
        raise NotImplementedError("only the sinusoidal position encoding of the released ED is supported")
    device = src.device
    space_id = tok.letters_map[" "]
    nt_id = tok.tashkeel_map[tok.no_tashkeel_tag]
    bos_id = tok.tashkeel_map["<BOS>"]
    B, Ls = src.shape

    src_mask = tr.make_pad_mask(src, src, tr.src_pad_idx, tr.src_pad_idx).to(device)
    enc = tr.encoder(src, src_mask)
    src_keep = src.ne(tr.src_pad_idx)  # [B, Ls]

    n_head = dec.layers[0].self_attention.n_head
    cross_kv = [
        (_split(layer.enc_dec_attention.w_k(enc), n_head), _split(layer.enc_dec_attention.w_v(enc), n_head))
        for layer in dec.layers
    ]
    cache_k: List[torch.Tensor | None] = [None] * len(dec.layers)
    cache_v: List[torch.Tensor | None] = [None] * len(dec.layers)
    pos_table = dec.emb.pos_emb.encoding.to(device)

    target = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    for i in range(Ls - 1):
        y = target[:, i : i + 1]
        x = dec.emb.tok_emb(y) + pos_table[i]
        q_keep = y.ne(tr.trg_pad_idx)  # [B, 1]
        t_mask = (target.ne(tr.trg_pad_idx).unsqueeze(1) & q_keep.unsqueeze(2)).unsqueeze(1)  # [B,1,1,i+1]
        s_mask = (src_keep.unsqueeze(1) & q_keep.unsqueeze(2)).unsqueeze(1)  # [B,1,1,Ls]
        for li, layer in enumerate(dec.layers):
            sa = layer.self_attention
            k_new = _split(sa.w_k(x), n_head)
            v_new = _split(sa.w_v(x), n_head)
            cache_k[li] = k_new if cache_k[li] is None else torch.cat([cache_k[li], k_new], dim=2)
            cache_v[li] = v_new if cache_v[li] is None else torch.cat([cache_v[li], v_new], dim=2)
            x = layer.norm1(_attention(sa, x, cache_k[li], cache_v[li], t_mask) + x)
            ck, cv = cross_kv[li]
            x = layer.norm2(_attention(layer.enc_dec_attention, x, ck, cv, s_mask) + x)
            x = layer.norm3(layer.ffn(x) + x)
        nxt = dec.linear(x)[:, -1].argmax(1)
        nxt = torch.where(src[:, i + 1] == space_id, torch.full_like(nxt, nt_id), nxt)
        target = torch.cat([target, nxt.unsqueeze(1)], dim=1)
    return target


def encode_batch(tok, texts: Sequence[str]) -> torch.Tensor:
    """Exactly upstream do_tashkeel_batch's input prep: encode(test_match=False), post-padding."""
    ids = [tok.encode(t, test_match=False)[0] for t in texts]
    return pad_seq(ids, batch_first=True, padding_value=tok.letters_map["<PAD>"], prepadding=False)


@torch.no_grad()
def tashkeel_batch_cached(model, texts: Sequence[str], batch_size: int = 16) -> List[str]:
    """Drop-in for upstream model.do_tashkeel_batch(texts, batch_size), cached decoder, same batching."""
    tok = model.tokenizer
    device = next(model.parameters()).device
    out: List[str] = []
    for mini in get_batches(list(texts), batch_size):
        src = encode_batch(tok, mini).to(device)
        out += tok.decode(src, greedy_decode(model, src))
    return out
