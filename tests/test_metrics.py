import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import base_letters, score_corpus, strip_marks, units  # noqa: E402

FATHA, DAMMA, KASRA, SHADDA, SUKUN = "َ", "ُ", "ِ", "ّ", "ْ"
B, T, K = "ب", "ت", "ك"


def test_mark_order_inside_a_unit_is_ignored():
    a = B + SHADDA + FATHA
    b = B + FATHA + SHADDA
    assert units(a)[0] == units(b)[0]
    assert score_corpus([a], [b]).as_dict()["metrics_pct"]["der_ce"] == 0.0


def test_doubled_mark_counts_once():
    assert units(B + FATHA + FATHA)[0] == units(B + FATHA)[0]


def test_stray_marks_are_counted_not_attached():
    us, stray = units(FATHA + B + " " + DAMMA + T)
    assert stray == 2
    assert [u[0] for u in us] == [B, " ", T]


def test_base_letters_and_strip():
    s = K + FATHA + T + SUKUN + " " + B + KASRA
    assert base_letters(s) == strip_marks(s) == K + T + " " + B


def test_case_ending_conventions():
    # two words; only the last letter of each word is wrong
    ref = K + FATHA + T + DAMMA + " " + B + FATHA + T + DAMMA
    hyp = K + FATHA + T + KASRA + " " + B + FATHA + T + KASRA
    m = score_corpus([ref], [hyp]).as_dict()
    c = m["counts"]
    assert c["der_ce"] == {"errors": 2, "total": 4}
    assert c["der_noce"] == {"errors": 0, "total": 2}  # last letter of every word dropped
    assert c["wer_ce"] == {"errors": 2, "total": 2}
    assert c["wer_noce"] == {"errors": 0, "total": 2}
    # upstream convention: every position incl. the space plus BOS/EOS; only the letter
    # before a space is neutralised, so the line-final case ending still counts
    assert c["upstream_conv_der_ce"] == {"errors": 2, "total": 5 + 2}  # 4 letters + 1 space + BOS/EOS
    assert c["upstream_conv_der_noce"] == {"errors": 1, "total": 7}
    assert c["upstream_conv_wer_noce"] == {"errors": 1, "total": 2}


def test_altered_line_is_listed_and_not_scored():
    ref = K + FATHA + T
    hyp = K + FATHA + B  # base letter changed
    m = score_corpus([ref, ref], [ref, hyp]).as_dict()
    assert m["altered_lines"] == 1 and m["altered_line_indices"] == [1]
    assert m["scored_lines"] == 1


def test_altered_against_input_even_if_gold_matches():
    ref = K + FATHA + T
    m = score_corpus([ref], [ref], inputs=[K + B]).as_dict()
    assert m["altered_line_indices"] == [0]


def test_unmarked_gold_letter_counted_in_main_der_only():
    ref = K + FATHA + "ا"  # alef with no mark
    hyp = K + FATHA + "ا" + SUKUN  # model adds a mark on alef
    c = score_corpus([ref], [hyp]).as_dict()["counts"]
    assert c["der_ce"] == {"errors": 1, "total": 2}
    assert c["der_ce_gold_marked_only"] == {"errors": 0, "total": 1}
