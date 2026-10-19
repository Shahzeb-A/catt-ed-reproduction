"""Tests that need the released checkpoint and the held-out split (skipped without them)."""
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CKPT = REPO / "data/release/best_ed_mlm_ns_epoch_178.pt"
SPLIT = REPO / "data/ds/dataset/test/test.txt"
need_assets = pytest.mark.skipif(not (CKPT.exists() and SPLIT.exists()), reason="run scripts/setup.sh first")

# the 36 base letters CATT's encoder knows (hamza forms, alef forms, ta marbuta, alef maksura, ...)
LETTERS = [chr(c) for c in range(0x0621, 0x063B)] + [chr(c) for c in range(0x0641, 0x064B)]
word = st.text(alphabet=LETTERS, min_size=1, max_size=7)
sentence = st.lists(word, min_size=1, max_size=5).map(" ".join)


@pytest.fixture(scope="module")
def model():
    import torch

    import catt_upstream as cu

    torch.set_num_threads(4)
    m, _ = cu.load_released_ed(CKPT)
    return m


@need_assets
def test_released_weights_load_strictly_and_size(model):
    import catt_upstream as cu

    assert cu.n_params(model) == 22_090_258


@need_assets
def test_heldout_selection_matches_upstream_count():
    import catt_upstream as cu
    from heldout import load_heldout

    assert cu.sha256_file(SPLIT) == "4e851ff836f0a178abb15d9f4a8bcf92748b77cc0d67030d9fae2fbe698baa12"
    lines = load_heldout(SPLIT, cu.make_tokenizer())
    assert len(lines) == 2324
    assert sum(len(l.model_input.split()) for l in lines) == 76811


@need_assets
@settings(max_examples=12, deadline=None)
@given(texts=st.lists(sentence, min_size=1, max_size=4))
def test_cached_decoder_equals_upstream_loop(model, texts):
    from cached_decode import tashkeel_batch_cached

    assert tashkeel_batch_cached(model, texts, 16) == model.do_tashkeel_batch(texts, 16, False)


@need_assets
@settings(max_examples=25, deadline=None)
@given(texts=st.lists(sentence, min_size=1, max_size=6))
def test_output_base_letters_equal_input(model, texts):
    from cached_decode import tashkeel_batch_cached
    from metrics import base_letters

    for src, out in zip(texts, tashkeel_batch_cached(model, texts, 16)):
        assert base_letters(out) == src
