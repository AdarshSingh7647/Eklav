import json

import pytest


@pytest.fixture
def table_raw(tmp_path):
    """A small raw table-reranking corpus (sft_examples.json format)."""
    rows = [
        {"qid": f"q{i}", "prompt": f"Question {i}\n\n### Table 1\nA\n### Table 2\nB",
         "thinking": f"Table {1 + i % 2} answers question {i}. The other one does not.",
         "answer": '\n\n{"ranked_tables": [%d, %d]}' % (1 + i % 2, 2 - i % 2)}
        for i in range(40)
    ]
    rows.append({"qid": "q0", "prompt": "dup", "thinking": "dup", "answer": '{"ranked_tables": [1]}'})   # duplicate qid
    rows.append({"qid": "bad", "prompt": "p", "thinking": "t", "answer": "no json"})                      # malformed
    path = tmp_path / "sft_examples.json"
    path.write_text(json.dumps(rows))
    return str(path)


@pytest.fixture
def passage_raw(tmp_path):
    """A small Rank1-format arrow corpus (instruction / input / output)."""
    from datasets import Dataset

    rows = []
    for i in range(60):
        trace = ("Okay, let's see. The query asks about topic %d. The passage mentions topic %d in detail. "
                 "It also lists several unrelated facts. So the passage is relevant. Therefore the answer is true." % (i, i))
        rows.append({"instruction": "Determine if the passage is relevant.", "input": f"Query: q{i}\nPassage: p{i}",
                     "output": f"<think>\n{trace}\n</think> {'true' if i % 2 else 'false'}"})
    rows.append({"instruction": "x", "input": "y", "output": "no think block"})                           # unparseable
    out = tmp_path / "rank1"
    Dataset.from_list(rows).save_to_disk(str(out))
    return str(next(out.glob("*.arrow")))


class FakeTokenizer:
    """Whitespace-free character-chunk tokenizer: decode joins pieces, like a real one."""

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)
