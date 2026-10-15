import json

from eklav import chunks


def test_json_array_round_trip(tmp_path):
    rows = [{"i": i, "text": "é" * 50} for i in range(100)]
    path = tmp_path / "big.json"
    path.write_text(json.dumps(rows, ensure_ascii=False))
    assert chunks.split_file(str(path), max_mb=0) == 100
    path.unlink()
    out = chunks.join_chunks(chunks.chunk_dir(str(path)))
    assert out == str(path) and json.load(open(path)) == rows


def test_jsonl_round_trip_and_join_all(tmp_path):
    lines = [json.dumps({"i": i}) for i in range(30)]
    path = tmp_path / "sub" / "rows.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join(lines) + "\n")
    chunks.split_file(str(path), max_mb=0)
    path.unlink()
    assert chunks.join_all([str(tmp_path)]) == 1
    assert path.read_text().splitlines() == lines
