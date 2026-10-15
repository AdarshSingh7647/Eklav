"""The `eklav` command: train, build training data, evaluate."""

import argparse
import sys
from typing import List, Optional

COMMANDS = {
    "train": "Train a LoRA SFT model (any task, any method).",
    "data": "Build, reassemble or split training data.",
    "eval": "Evaluate a model on a benchmark.",
}


def build_data_parser() -> argparse.ArgumentParser:
    from eklav.methods import METHODS
    from eklav.tasks import TASKS

    ap = argparse.ArgumentParser(prog="eklav data", description="Training-data utilities.")
    sub = ap.add_subparsers(dest="action", required=True)

    build = sub.add_parser("build", help="Build the training data a method trains on from a raw corpus.")
    build.add_argument("--task", required=True, choices=sorted(TASKS))
    build.add_argument("--method", required=True, choices=METHODS)
    build.add_argument("--raw_path", required=True, help="Raw training corpus of the task.")
    build.add_argument("--data_root", default="data")

    join = sub.add_parser("join", help="Reassemble every *.chunks/ directory under the given roots.")
    join.add_argument("roots", nargs="*", default=["data", "results"])

    split = sub.add_parser("split", help="Split a large .json/.jsonl file into <100 MB chunks for GitHub.")
    split.add_argument("path")
    split.add_argument("--max_mb", type=int, default=90)
    return ap


def data_main(argv: List[str]) -> int:
    args = build_data_parser().parse_args(argv)
    if args.action == "build":
        from eklav.data import build_data
        from eklav.methods import SEQUENCE_LAYOUT
        from eklav.tasks import get_task

        paths = build_data(get_task(args.task).data, SEQUENCE_LAYOUT[args.method], args.data_root, args.raw_path)
        print(f"{args.task}/{args.method}: {paths}")
    elif args.action == "join":
        from eklav.chunks import join_all

        print(f"Reassembled {join_all(args.roots)} chunked files.")
    else:
        from eklav.chunks import split_file

        print(f"{args.path}: {split_file(args.path, args.max_mb)} chunks")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: eklav {" + ",".join(COMMANDS) + "} [options]\n")
        for name, help_text in COMMANDS.items():
            print(f"  {name:6s} {help_text}")
        print("\nRun `eklav <command> --help` for a command's options.")
        return 0 if argv and argv[0] in ("-h", "--help") or not argv else 2
    command, rest = argv[0], argv[1:]
    if command == "train":
        from eklav.train import main as train_main

        return train_main(rest)
    if command == "eval":
        from eklav.eval.run import main as eval_main

        return eval_main(rest)
    return data_main(rest)


if __name__ == "__main__":
    sys.exit(main())
