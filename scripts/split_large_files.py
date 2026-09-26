#!/usr/bin/env python3
"""
Split large .json (flat array) or .jsonl (line-delimited) files into
<100MB chunks so they can be pushed to GitHub, and reassemble them later.

Split layout: for a file at path/to/name.json (or .jsonl), chunks are written
to path/to/name.json.chunks/ as 000.json, 001.json, ... (or .jsonl), plus a
manifest.json recording the original filename, format, and chunk order. The
original oversized file is left in place; delete it yourself once you've
verified the chunks (this script does not delete anything).

Usage:
  python scripts/split_large_files.py split path/to/big_file.json --max_mb 90
  python scripts/split_large_files.py join path/to/big_file.json.chunks --out path/to/big_file.json
  python scripts/split_large_files.py join-all data results  # find & join every *.chunks dir under these roots
"""
import argparse
import json
import os


def _chunk_dir(path: str) -> str:
    return path + ".chunks"


def split_json_array(path: str, max_mb: int) -> None:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array at the top level, got {type(data).__name__}")

    out_dir = _chunk_dir(path)
    os.makedirs(out_dir, exist_ok=True)
    max_bytes = max_mb * 1024 * 1024

    chunks, current, current_size = [], [], 2  # 2 bytes for "[]"
    for row in data:
        row_bytes = len(json.dumps(row, ensure_ascii=False).encode("utf-8")) + 1  # +1 for comma/separator
        if current and current_size + row_bytes > max_bytes:
            chunks.append(current)
            current, current_size = [], 2
        current.append(row)
        current_size += row_bytes
    if current:
        chunks.append(current)

    names = []
    for i, chunk in enumerate(chunks):
        name = f"{i:03d}.json"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False)
        names.append(name)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"format": "json_array", "original_name": os.path.basename(path), "chunks": names}, f, indent=2)

    print(f"[split] {path}: {len(data)} records -> {len(names)} chunks in {out_dir}/")


def split_jsonl(path: str, max_mb: int) -> None:
    out_dir = _chunk_dir(path)
    os.makedirs(out_dir, exist_ok=True)
    max_bytes = max_mb * 1024 * 1024

    names = []
    chunk_idx = 0
    current_size = 0
    current_fh = None
    n_lines = 0

    def _open_next():
        nonlocal current_fh, chunk_idx, current_size
        if current_fh is not None:
            current_fh.close()
        name = f"{chunk_idx:03d}.jsonl"
        names.append(name)
        current_fh = open(os.path.join(out_dir, name), "w")
        chunk_idx += 1
        current_size = 0

    _open_next()
    with open(path) as f:
        for line in f:
            line_bytes = len(line.encode("utf-8"))
            if current_size > 0 and current_size + line_bytes > max_bytes:
                _open_next()
            current_fh.write(line if line.endswith("\n") else line + "\n")
            current_size += line_bytes
            n_lines += 1
    current_fh.close()

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"format": "jsonl", "original_name": os.path.basename(path), "chunks": names}, f, indent=2)

    print(f"[split] {path}: {n_lines} lines -> {len(names)} chunks in {out_dir}/")


def split_file(path: str, max_mb: int) -> None:
    if path.endswith(".jsonl"):
        split_jsonl(path, max_mb)
    elif path.endswith(".json"):
        split_json_array(path, max_mb)
    else:
        raise ValueError(f"{path}: unsupported extension (expected .json or .jsonl)")


def join_chunks(chunk_dir: str, out_path: str = None) -> str:
    manifest_path = os.path.join(chunk_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    if out_path is None:
        base = chunk_dir[:-len(".chunks")] if chunk_dir.endswith(".chunks") else chunk_dir
        out_path = os.path.join(os.path.dirname(base) or ".", manifest["original_name"])

    if manifest["format"] == "json_array":
        rows = []
        for name in manifest["chunks"]:
            with open(os.path.join(chunk_dir, name), encoding="utf-8") as f:
                rows.extend(json.load(f))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(f"[join] {chunk_dir} -> {out_path} ({len(rows)} records)")
    elif manifest["format"] == "jsonl":
        n_lines = 0
        with open(out_path, "w") as out_f:
            for name in manifest["chunks"]:
                with open(os.path.join(chunk_dir, name)) as f:
                    for line in f:
                        out_f.write(line)
                        n_lines += 1
        print(f"[join] {chunk_dir} -> {out_path} ({n_lines} lines)")
    else:
        raise ValueError(f"Unknown manifest format: {manifest['format']}")
    return out_path


def join_all(roots) -> None:
    for root in roots:
        for dirpath, dirnames, _ in os.walk(root):
            for d in list(dirnames):
                if d.endswith(".chunks"):
                    join_chunks(os.path.join(dirpath, d))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split")
    sp.add_argument("path")
    sp.add_argument("--max_mb", type=int, default=90)

    jp = sub.add_parser("join")
    jp.add_argument("chunk_dir")
    jp.add_argument("--out", default=None)

    ja = sub.add_parser("join-all")
    ja.add_argument("roots", nargs="+")

    args = ap.parse_args()
    if args.cmd == "split":
        split_file(args.path, args.max_mb)
    elif args.cmd == "join":
        join_chunks(args.chunk_dir, args.out)
    elif args.cmd == "join-all":
        join_all(args.roots)


if __name__ == "__main__":
    main()
