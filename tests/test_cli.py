import json

from eklav import cli


def test_help_lists_the_commands(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert all(c in out for c in ("train", "data", "eval"))


def test_unknown_command_fails(capsys):
    assert cli.main(["finetune"]) == 2


def test_data_command_builds_the_training_files(table_raw, tmp_path, capsys):
    rc = cli.main(["data", "build", "--task", "table_reranking", "--method", "mask-only", "--raw_path", table_raw,
                   "--data_root", str(tmp_path)])
    assert rc == 0
    rows = json.load(open(tmp_path / "table_reranking" / "std-cot" / "train.json"))
    assert rows[0]["conversations"][1]["value"].startswith("<think>\n")
