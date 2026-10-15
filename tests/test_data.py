import json

import pytest

from eklav import data
from eklav.methods import EKLAV, MASK_ONLY, NAIVE_SFT, STD_COT
from eklav.tasks import TASKS, passage_reranking as passage, table_reranking as table

TRACE = ("Okay, let's see. The query asks about tetany. The passage lists symptoms. "
         "So the passage is relevant. The rest is unrelated. Therefore the answer is true.")


def test_passage_hint_is_the_verdict_free_first_half():
    sentences = passage.split_sentences(TRACE)
    assert len(sentences) == 6
    hint = passage.hint(TRACE)
    assert hint == "Okay, let's see. The query asks about tetany. The passage lists symptoms."
    assert not passage.VERDICT_RE.search(hint)


def test_passage_hint_drops_verdict_sentences_inside_the_first_half():
    trace = "Start here. So it is true that it is relevant. More detail follows. Later part. End. Done."
    assert passage.hint(trace) == "Start here. More detail follows."


def test_passage_hint_keeps_task_framing_that_is_not_a_judgment():
    trace = "We check if the passage is relevant to the query. Then look closer. a. b. c. d."
    assert "relevant to the query" in passage.hint(trace)


def test_passage_hint_is_much_shorter_than_the_trace():
    assert len(passage.hint(TRACE)) < 0.6 * len(TRACE)


def test_table_hint_is_the_whole_trace():
    assert table.hint(TRACE) == TRACE


@pytest.mark.parametrize("task", sorted(TASKS))
def test_layouts_follow_the_paper(task):
    spec = TASKS[task].data
    ex = {"x": "INPUT", "r": TRACE, "a": "ANSWER"}
    prompt, response = data.serialize(STD_COT, ex, spec.hint, spec.hint_prompt)
    assert (prompt, response) == ("INPUT", f"<think>\n{TRACE}\n</think>\n\nANSWER")
    assert data.serialize(NAIVE_SFT, ex, spec.hint, spec.hint_prompt) == ("INPUT", "ANSWER")
    prompt, response = data.serialize(EKLAV, ex, spec.hint, spec.hint_prompt)
    assert prompt.startswith("INPUT") and spec.hint(TRACE) in prompt and response == "ANSWER"


def test_eklav_prompt_holds_only_the_hint_never_the_dropped_half():
    spec = passage.TASK.data
    prompt, _ = data.serialize(EKLAV, {"x": "X", "r": TRACE, "a": "true"}, spec.hint, spec.hint_prompt)
    assert "The rest is unrelated" not in prompt and "Therefore the answer is true" not in prompt


def test_table_eklav_prompt_holds_the_full_trace():
    spec = table.TASK.data
    prompt, _ = data.serialize(EKLAV, {"x": "X", "r": TRACE, "a": "{}"}, spec.hint, spec.hint_prompt)
    assert TRACE in prompt


def test_passage_corpus_filters_unparseable_rows(passage_raw):
    examples = passage.load_corpus(passage_raw)
    assert len(examples) == 60 and {e["a"] for e in examples} == {"true", "false"}


def test_table_corpus_drops_duplicates_and_malformed_rows(table_raw):
    examples = table.load_corpus(table_raw)
    assert len(examples) == 40
    assert json.loads(examples[0]["a"]) == {"ranked_tables": [1, 2]}


@pytest.mark.parametrize("task,raw", [("passage_reranking", "passage_raw"), ("table_reranking", "table_raw")])
def test_all_layouts_hold_out_the_same_examples(task, raw, request, tmp_path):
    raw_path, spec = request.getfixturevalue(raw), TASKS[task].data
    prompts = {}
    for layout in (STD_COT, NAIVE_SFT, EKLAV):
        paths = data.build_data(spec, layout, str(tmp_path), raw_path)
        val = json.load(open(paths["val"]))
        prompts[layout] = [r["conversations"][0]["value"].split("\n\nThink step-by-step")[0] for r in val]
        assert all(r["system"] == spec.system and r["task"] == task for r in val)
    assert prompts[STD_COT] == prompts[NAIVE_SFT] == prompts[EKLAV]
    assert len(prompts[STD_COT]) >= 1


def test_get_data_reuses_built_files_and_mask_only_shares_std_cot(table_raw, tmp_path):
    spec = table.TASK.data
    built = data.get_data(spec, STD_COT, str(tmp_path), table_raw)
    assert data.get_data(spec, MASK_ONLY, str(tmp_path)) == built      # no raw corpus needed the second time


def test_get_data_without_raw_corpus_explains_what_to_pass(tmp_path):
    with pytest.raises(ValueError, match="--raw_path"):
        data.get_data(table.TASK.data, EKLAV, str(tmp_path))


def test_tracked_passage_data_matches_the_builder():
    """The repository's shipped sharegpt data follows the layouts and the hint rule."""
    import os
    import re

    root = os.path.join(os.path.dirname(__file__), "..", "data", "passage_reranking")
    if not os.path.isdir(root):
        pytest.skip("no tracked data")
    val = {layout: json.load(open(os.path.join(root, layout, "val.json"))) for layout in (STD_COT, NAIVE_SFT, EKLAV)}
    for std, naive, eklav in list(zip(val[STD_COT], val[NAIVE_SFT], val[EKLAV]))[:300]:
        m = re.fullmatch(r"<think>\n(.*)\n</think>\n\n(true|false)", std["conversations"][1]["value"], re.S)
        x, trace, answer = std["conversations"][0]["value"], m.group(1), m.group(2)
        assert naive["conversations"] == [{"from": "human", "value": x}, {"from": "gpt", "value": answer}]
        assert eklav["conversations"][0]["value"] == passage.hint_prompt(x, passage.hint(trace))
        assert eklav["conversations"][1]["value"] == answer


def test_get_data_reassembles_tracked_chunks_without_a_raw_corpus(tmp_path):
    from eklav.chunks import split_file

    spec = table.TASK.data
    directory = tmp_path / spec.task / EKLAV
    directory.mkdir(parents=True)
    rows = [{"system": "s", "conversations": [{"from": "human", "value": f"x{i}"}, {"from": "gpt", "value": "a"}]}
            for i in range(50)]
    (directory / "train.json").write_text(json.dumps(rows))
    (directory / "val.json").write_text(json.dumps(rows[:2]))
    assert split_file(str(directory / "train.json"), max_mb=0) >= 2         # one row per chunk
    (directory / "train.json").unlink()
    paths = data.get_data(spec, EKLAV, str(tmp_path))
    assert json.load(open(paths["train"])) == rows
