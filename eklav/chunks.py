"""
Split large JSON / JSONL files into chunks small enough for GitHub (<100 MB) and
reassemble them.

A file path/to/name.json (a flat JSON array) or name.jsonl is split into
path/to/name.json.chunks/000.json, 001.json, ... plus a manifest.json recording the
original name, format and chunk order. Splitting never deletes the original.
"""

import json
import os
from typing import Iterable, Optional

CHUNK_SUFFIX = ".chunks"


def chunk_dir(path: str) -> str:
    return path + CHUNK_SUFFIX


def _write_manifest(directory: str, fmt: str, original: str, names) -> None:
    with open(os.path.join(directory, "manifest.json"), "w") as f:
        json.dump({"format": fmt, "original_name": os.path.basename(original), "chunks": names}, f, indent=2)


def _split_json_array(path: str, max_mb: int) -> int:
    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array at the top level, got {type(rows).__name__}")
    out = chunk_dir(path)
    os.makedirs(out, exist_ok=True)
    limit = max_mb * 1024 * 1024

    chunks, current, size = [], [], 2                      # 2 bytes for "[]"
    for row in rows:
        row_bytes = len(json.dumps(row, ensure_ascii=False).encode("utf-8")) + 1
        if current and size + row_bytes > limit:
            chunks.append(current)
            current, size = [], 2
        current.append(row)
        size += row_bytes
    if current:
        chunks.append(current)

    names = []
    for i, chunk in enumerate(chunks):
        names.append(f"{i:03d}.json")
        with open(os.path.join(out, names[-1]), "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False)
    _write_manifest(out, "json_array", path, names)
    return len(names)


def _split_jsonl(path: str, max_mb: int) -> int:
    out = chunk_dir(path)
    os.makedirs(out, exist_ok=True)
    limit = max_mb * 1024 * 1024
    names, handle, size = [], None, 0
    try:
        with open(path) as f:
            for line in f:
                line_bytes = len(line.encode("utf-8"))
                if handle is None or (size > 0 and size + line_bytes > limit):
                    if handle is not None:
                        handle.close()
                    names.append(f"{len(names):03d}.jsonl")
                    handle, size = open(os.path.join(out, names[-1]), "w"), 0
                handle.write(line if line.endswith("\n") else line + "\n")
                size += line_bytes
    finally:
        if handle is not None:
            handle.close()
    _write_manifest(out, "jsonl", path, names)
    return len(names)


def split_file(path: str, max_mb: int = 90) -> int:
    """Split `path` into chunks of at most `max_mb` MB; returns the chunk count."""
    if path.endswith(".jsonl"):
        return _split_jsonl(path, max_mb)
    if path.endswith(".json"):
        return _split_json_array(path, max_mb)
    raise ValueError(f"{path}: unsupported extension (expected .json or .jsonl)")


def join_chunks(directory: str, out_path: Optional[str] = None) -> str:
    """Reassemble a chunk directory into its original file; returns the output path."""
    with open(os.path.join(directory, "manifest.json")) as f:
        manifest = json.load(f)
    if out_path is None:
        base = directory[: -len(CHUNK_SUFFIX)] if directory.endswith(CHUNK_SUFFIX) else directory
        out_path = os.path.join(os.path.dirname(base) or ".", manifest["original_name"])

    if manifest["format"] == "json_array":
        rows = []
        for name in manifest["chunks"]:
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                rows.extend(json.load(f))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
    elif manifest["format"] == "jsonl":
        with open(out_path, "w") as out:
            for name in manifest["chunks"]:
                with open(os.path.join(directory, name)) as f:
                    out.writelines(f)
    else:
        raise ValueError(f"Unknown manifest format: {manifest['format']}")
    return out_path


def join_all(roots: Iterable[str]) -> int:
    """Reassemble every chunk directory under `roots`; returns how many were joined."""
    count = 0
    for root in roots:
        for dirpath, dirnames, _ in os.walk(root):
            for d in list(dirnames):
                if d.endswith(CHUNK_SUFFIX):
                    print(f"[join] {os.path.join(dirpath, d)} -> {join_chunks(os.path.join(dirpath, d))}")
                    count += 1
    return count
