from eklav import methods


def test_paper_names_cover_every_method():
    assert set(methods.PAPER_NAMES) == set(methods.METHODS)
    assert [methods.PAPER_NAMES[m] for m in methods.METHODS] == ["STD-CoT", "NAIVE-SFT", "EKLAV", "MASK-ONLY"]


def test_mask_only_trains_on_std_cot_sequence():
    assert methods.SEQUENCE_LAYOUT[methods.MASK_ONLY] == methods.SEQUENCE_LAYOUT[methods.STD_COT]


def test_only_std_cot_supervises_the_whole_response():
    assert methods.loss_mask_env(methods.STD_COT) == {}
    for m in (methods.NAIVE_SFT, methods.EKLAV, methods.MASK_ONLY):
        assert methods.loss_mask_env(m) == {"EKLAV_ANSWER_ONLY_AFTER_TOKEN": "</think>"}


def test_unknown_method_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        methods.loss_mask_env("answer-only")
