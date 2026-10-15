"""Each evaluator's `run` phase end to end, with the generator stubbed (no GPU needed)."""

import json
from types import SimpleNamespace

from eklav.eval import benchmarks
from eklav.eval.passage_reranking import PassageRerankingEvaluator
from eklav.eval.table_reranking import TableRerankingEvaluator


def small_benchmark():
    corpus = {f"d{i}": f"document {i}" for i in range(6)}
    queries = {"q1": "first", "q2": "second"}
    qrels = {"q1": {"d1": 1}, "q2": {"d4": 1}}
    return benchmarks.RetrievalData(queries, corpus, qrels, candidates={q: list(corpus) for q in queries},
                                    query_template="Find: {query}")


class StubScorer:
    """Scores document d<i> highest for the query whose relevant document it is."""

    def format_prompt(self, query, passage):
        return f"{query}|{passage}"

    def generate_and_score(self, prompts):
        gold = {"Find: first": "document 1", "Find: second": "document 4"}
        return [("raw reasoning", 0.99 if p.split("|")[1] == gold[p.split("|")[0]] else 0.1) for p in prompts]


def test_passage_run_writes_rows_and_metrics(tmp_path):
    evaluator = PassageRerankingEvaluator.__new__(PassageRerankingEvaluator)
    evaluator.scorer, evaluator.top_k, evaluator.first_stage = StubScorer(), 6, "bm25"
    data = small_benchmark()
    prepared = evaluator.prepare("toy", "toy", data, ["q1", "q2"])
    assert len(prepared.pairs) == 12 and prepared.prompts[0] == "Find: first|document 0"     # template applied
    summary = evaluator.run(prepared, str(tmp_path), "model-tag")
    assert summary["ndcg@10"] == 1.0 and summary["mrr"] == 1.0 and summary["n_queries"] == 2
    rows = [json.loads(line) for line in open(tmp_path / "toy.jsonl")]
    assert len(rows) == 12 and {r["model_tag"] for r in rows} == {"model-tag"}
    gold = [r for r in rows if r["gold_label"] == 1]
    assert {(r["query_id"], r["doc_id"]) for r in gold} == {("q1", "d1"), ("q2", "d4")}
    assert all(r["parsed_verdict"] == "true" for r in gold)


def test_passage_top_k_limits_the_judged_candidates():
    evaluator = PassageRerankingEvaluator.__new__(PassageRerankingEvaluator)
    evaluator.scorer, evaluator.top_k, evaluator.first_stage = StubScorer(), 3, "bm25"
    assert len(evaluator.prepare("toy", "toy", small_benchmark(), ["q1"]).pairs) == 3


def test_table_run_aggregates_windows_and_counts_parse_failures(tmp_path):
    class Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
            return "\n".join(m["content"] for m in messages)

        def encode(self, text, add_special_tokens=False):
            return text.split()

    evaluator = TableRerankingEvaluator.__new__(TableRerankingEvaluator)
    evaluator.tokenizer = Tok()
    data = small_benchmark()
    prepared = evaluator.prepare("toy", "toy", data, ["q1", "q2"])
    assert prepared.records and all("Candidate tables to rank" in r["prompt"] for r in prepared.records)

    def generate(prompts):
        outputs = []
        for p in prompts:
            if "second" in p:
                text = "no ranking given"                                   # q2: every window unparseable
            else:
                n = p.count("### Table")
                order = list(range(1, n + 1))
                if "document 1" in p:
                    order.remove(2); order.insert(0, 2)                       # promote d1 (table 2 in window)
                text = f"<think>why</think> {json.dumps({'ranked_tables': order})}"
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(text=text)]))
        return outputs

    evaluator._generate = generate
    summary = evaluator.run(prepared, str(tmp_path), "tag")
    assert summary["n_queries"] == 2 and summary["n_parse_failed_queries"] == 1
    rows = {json.loads(l)["qid"]: json.loads(l) for l in open(tmp_path / "toy.jsonl")}
    assert rows["q1"]["final_ranked_list"][0] == "d1" and rows["q1"]["metrics"]["ndcg@10"] == 1.0
    assert rows["q2"]["any_parse_failed"] and rows["q2"]["metrics"]["ndcg@10"] == 0.0
    assert summary["ndcg@10"] == 0.5
