#!/usr/bin/env python3
"""
Thin CLI wrapper to download/build sharegpt SFT data for one (task, method)
without launching training. Useful to pre-stage data on a machine before a
training run, or to inspect what get_data() produces.

Usage:
  python scripts/download_data.py --task passage_reranking --method eklav
  python scripts/download_data.py --task math_reasoning --method answer-only \
      --raw_dataset_path /path/to/stage2-3k.json
  python scripts/download_data.py --task table_reranking --method eklav \
      --raw_path /path/to/sft_examples.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["passage_reranking", "table_reranking", "math_reasoning"])
    ap.add_argument("--method", required=True, choices=["std-sft", "answer-only", "eklav"])
    ap.add_argument("--data_root", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--raw_arrow_path", default=None)
    ap.add_argument("--raw_dataset_path", default=None)
    ap.add_argument("--raw_path", default=None)
    args = ap.parse_args()

    if args.task == "passage_reranking":
        from eklav.data import passage_reranking as mod

        paths = mod.get_data(args.method, args.data_root, raw_arrow_path=args.raw_arrow_path)
    elif args.task == "math_reasoning":
        from eklav.data import math_reasoning as mod

        paths = mod.get_data(args.method, args.data_root, raw_dataset_path=args.raw_dataset_path)
    else:
        from eklav.data import table_reranking as mod

        paths = mod.get_data(args.method, args.data_root, raw_path=args.raw_path)

    print(f"[download_data] {args.task}/{args.method} -> {paths}")


if __name__ == "__main__":
    main()
