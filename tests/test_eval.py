import json

import pytest

from eklav.eval import benchmarks, run
from eklav.eval.metrics import compute_ranking_metrics, mean_metrics, ndcg_at_k, zero_metrics
from eklav.eval.table_reranking import (aggregate_bubble_sort, build_windows, parse_ranked_tables,
                                        parse_thinking_and_answer)


@pytest.fixture
def beir_dir(tmp_path):
    (tmp_path / "qrels").mkdir()
    corpus = [{"_id": f"d{i}", "title": f"Title {i}", "text": f"table about subject {i} and more words"} for i in range(30)]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in corpus))
    queries = [{"_id": f"q{i}", "text": f"subject {i}"} for i in range(10)] + [{"_id": "qx", "text": "no judgments"}]
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in queries))
    rows = ["query-id\tcorpus-id\tscore"] + [f"q{i}\td{i}\t1" for i in range(10)] + ["q0\td5\t0"]
    (tmp_path / "qrels" / "test.tsv").write_text("\n".join(rows))
    return str(tmp_path)


def test_ndcg_matches_hand_computation():
    assert ndcg_at_k(["a", "b", "c"], {"a"}, 10) == 1.0
    assert round(ndcg_at_k(["x", "a"], {"a"}, 10), 4) == 0.6309          # 1/log2(3)


def test_metric_bundles():
    m = compute_ranking_metrics(["x", "a"], {"a"}, (1, 5))
    assert m["mrr"] == 0.5 and m["recall@1"] == 0.0 and m["recall@5"] == 1.0
    assert set(zero_metrics((1, 5))) == set(m)
    assert mean_metrics([{"ndcg@10": 1.0}, {"ndcg@10": 0.0}]) == {"ndcg@10": 0.5}


def test_registry_lists_the_paper_benchmarks():
    registered = benchmarks.names()
    for name in ("bright", "FeTaQARetrieval", "OTTQASmallRetrieval", "OpenWikiTablesRetrieval", "AITQARetrieval",
                 "NQTablesRetrieval"):
        assert name in registered
    assert len(benchmarks.resolve("bright").subset_names()) == 12


def test_unknown_benchmark_names_the_alternatives():
    with pytest.raises(ValueError, match="local:<dir>"):
        benchmarks.resolve("nope")


def test_local_beir_benchmark_loads_with_positive_judgments_only(beir_dir):
    bench = benchmarks.resolve(f"local:{beir_dir}")
    data = bench.loader(bench.name)
    assert len(data.corpus) == 30 and data.corpus["d0"] == "Title 0\ntable about subject 0 and more words"
    assert data.qrels["q0"] == {"d0": 1}                      # the score-0 judgment is not relevance
    assert benchmarks.select_queries(data) == [f"q{i}" for i in range(10)]
    sample = benchmarks.select_queries(data, 4, seed=1)
    assert sample == benchmarks.select_queries(data, 4, seed=1) and len(sample) == 4


def test_a_python_registered_benchmark_is_resolvable():
    data = benchmarks.RetrievalData({"q": "text"}, {"d": "doc"}, {"q": {"d": 1}})
    benchmarks.register(benchmarks.Benchmark("unit-test-benchmark", loader=lambda subset: data))
    assert benchmarks.resolve("unit-test-benchmark").loader("x") is data


def test_bm25_pools_find_the_matching_document(beir_dir):
    pytest.importorskip("rank_bm25")
    from eklav.eval.retrieval import build_pools

    data = benchmarks.load_beir_dir(beir_dir)
    pools = build_pools(data.corpus, {"q3": data.queries["q3"]}, {"q3": ["d3"]}, kind="bm25", top_k=5)
    assert pools["q3"][0] == "d3"


def test_gold_injection_appends_missing_relevant_documents(beir_dir):
    pytest.importorskip("rank_bm25")
    from eklav.eval.retrieval import build_pools

    data = benchmarks.load_beir_dir(beir_dir)
    pools = build_pools(data.corpus, {"q3": "Title 3 subject 3"}, {"q3": ["d17"]}, kind="bm25", top_k=1, inject_gold=True)
    assert pools["q3"] == ["d3", "d17"]                       # best hit first, then the missing relevant document


def test_table_output_parsing():
    assert parse_ranked_tables('reasoning... {"ranked_tables": [2, 1, 3]}') == [2, 1, 3]
    assert parse_ranked_tables('{"ranked_tables": [1]} then later {"ranked_tables": [3, 2]}') == [3, 2]
    assert parse_ranked_tables('"ranked_tables": ["2", "1"]') == [2, 1]
    assert parse_ranked_tables("no answer here") is None
    assert parse_thinking_and_answer("<think>why</think> {}") == ("why", "{}")


def test_bubble_sort_aggregation_and_parse_failures():
    pool = ["a", "b", "c", "d"]
    windows = [{"window_idx": 0, "window_start": 0, "window_end": 3, "window_candidate_ids": ["a", "b", "c"],
                "ranked_indices": [3, 1, 2], "parse_failed": False},
               {"window_idx": 1, "window_start": 1, "window_end": 4, "window_candidate_ids": ["a", "b", "d"],
                "ranked_indices": None, "parse_failed": True}]
    ranking, failed = aggregate_bubble_sort(pool, windows)
    assert ranking == ["c", "a", "b", "d"] and failed


def test_sliding_windows_respect_size_and_stride():
    class Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
            return "\n".join(m["content"] for m in messages)

        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))

    pool = [f"t{i}" for i in range(25)]
    corpus = {d: "word " * 5 for d in pool}
    windows = build_windows(pool, "question?", corpus, {d: 8 for d in pool}, Tok(), window_size=10, window_tokens=10_000, stride=5)
    assert [(w["window_start"], w["window_end"]) for w in windows][:3] == [(0, 10), (5, 15), (10, 20)]
    assert windows[-1]["window_end"] == 25


def test_eval_cli_lists_benchmarks_and_checks_arguments(capsys):
    assert run.main(["--list_benchmarks"]) == 0
    assert "bright" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run.main(["--benchmark", "bright"])            # missing --task / --model_path / ...


def test_eval_cli_downloads_a_local_benchmark(beir_dir, capsys):
    assert run.main(["--benchmark", f"local:{beir_dir}", "--download_only"]) == 0
    assert "11 queries, 30 documents" in capsys.readouterr().out
