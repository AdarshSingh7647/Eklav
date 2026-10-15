import warnings

from eklav.lf_plugin import masking
from tests.conftest import FakeTokenizer

# <prompt> <think>\n\n</think>\n\n ans </eos>
PIECES = ["PROMPT", "<think>", "\n\n", "</think>", "\n\n", "true", "<eos>", "<nl>"]


def make(labels_from):
    ids = list(range(len(PIECES)))
    labels = [-100] * labels_from + ids[labels_from:]
    return FakeTokenizer(PIECES), ids, labels


def test_answer_only_masks_everything_through_the_closing_tag_and_following_whitespace():
    tokenizer, ids, labels = make(labels_from=1)
    out = masking.answer_only_labels(tokenizer, ids, labels, "</think>")
    assert out == [-100] * 5 + [5, 6, 7]                  # only `true`, eos, newline are supervised


def test_marker_in_the_prompt_is_ignored():
    tokenizer = FakeTokenizer(["</think>", "PROMPT", "<think>", "x", "</think>", "ans"])
    ids = list(range(6))
    labels = [-100, -100, 2, 3, 4, 5]
    assert masking.answer_only_labels(tokenizer, ids, labels, "</think>") == [-100] * 5 + [5]


def test_the_last_closing_tag_wins():
    tokenizer = FakeTokenizer(["P", "</think>", "mid", "</think>", "ans"])
    ids = list(range(5))
    assert masking.answer_only_labels(tokenizer, ids, [-100, 1, 2, 3, 4], "</think>") == [-100] * 4 + [4]


def test_response_without_marker_contributes_no_loss_and_warns_once():
    masking._warned_missing_marker = False
    tokenizer = FakeTokenizer(["P", "partial", "reasoning"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert masking.answer_only_labels(tokenizer, [0, 1, 2], [-100, 1, 2], "</think>") == [-100] * 3
        masking.answer_only_labels(tokenizer, [0, 1, 2], [-100, 1, 2], "</think>")
    assert len(caught) == 1


def test_startup_directory_ships_a_sitecustomize():
    import os

    from eklav.lf_plugin import startup_dir

    assert os.path.isfile(os.path.join(startup_dir(), "sitecustomize.py"))
