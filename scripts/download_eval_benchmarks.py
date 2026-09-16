#!/usr/bin/env python3
"""
Pre-download eval-time benchmark sources (no generation, just caches the HF
datasets locally via huggingface_hub/datasets caching):
  - BRIGHT (mteb/BRIGHT, pinned revision) + jhu-clsp/rank1-run-files, for
    passage_reranking eval.
  - The 6 IBM table-retrieval benchmarks, for table_reranking eval.

Usage:
  python scripts/download_eval_benchmarks.py --passage_reranking
  python scripts/download_eval_benchmarks.py --table_reranking
  python scripts/download_eval_benchmarks.py --all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def download_passage_reranking_eval_sources():
    from eklav.data.eval_downloads import BRIGHT_DOMAINS, bright_runfile_name, load_bright_domain, load_rank1_run_file

    for domain in BRIGHT_DOMAINS:
        print(f"[BRIGHT] downloading {domain} ...")
        load_bright_domain(domain)
        print(f"[rank1-run-files] downloading {domain} candidate pool ...")
        load_rank1_run_file(bright_runfile_name(domain))
    print("[passage_reranking] all BRIGHT domains + rank1 run files cached.")


def download_table_reranking_eval_sources():
    from eklav.data.table_reranking import OOD_BENCHMARK_REPOS, download_benchmark

    for name in OOD_BENCHMARK_REPOS:
        print(f"[{name}] downloading corpus/queries/qrels ...")
        try:
            download_benchmark(name)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
    print("[table_reranking] all benchmark repos attempted.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passage_reranking", action="store_true")
    ap.add_argument("--table_reranking", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all or args.passage_reranking:
        download_passage_reranking_eval_sources()
    if args.all or args.table_reranking:
        download_table_reranking_eval_sources()
    if not (args.all or args.passage_reranking or args.table_reranking):
        ap.error("Pass at least one of --passage_reranking / --table_reranking / --all")


if __name__ == "__main__":
    main()
