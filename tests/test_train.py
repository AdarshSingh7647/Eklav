import json
import os

import pytest
import yaml

from eklav import train
from eklav.methods import METHODS
from eklav.tasks import TASKS


@pytest.mark.parametrize("task", sorted(TASKS))
def test_config_keeps_the_effective_batch_fixed_whatever_the_gpu_count(task):
    hp = TASKS[task].hparams
    for gpus in (1, 2, 4, 8):
        cfg = train.build_train_config(task, "Qwen/Qwen3-8B", "out", "ds", "t", "v", per_device_batch=1, world_size=gpus)
        batch = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"] * gpus
        assert batch == hp["effective_batch_size"] == 128


def test_indivisible_batch_is_rejected_with_a_helpful_message():
    with pytest.raises(ValueError, match="not divisible"):
        train.gradient_accumulation(128, per_device=64, world_size=4)


def test_paper_hyperparameters():
    for task in TASKS:
        hp = TASKS[task].hparams
        assert (hp["lora_rank"], hp["lora_alpha"], hp["learning_rate"], hp["num_train_epochs"]) == (32, 64, 1e-4, 2.0)
        assert hp["warmup_ratio"] == 0.05 and hp["lr_scheduler_type"] == "cosine"
    assert TASKS["passage_reranking"].hparams["cutoff_len"] == 2500
    assert TASKS["table_reranking"].hparams["cutoff_len"] == 32768


def test_glm_z1_gets_its_own_chat_template():
    cfg = train.build_train_config("passage_reranking", "zai-org/GLM-Z1-9B-0414", "o", "d", "t", "v", world_size=2)
    assert cfg["template"] == "glmz1"
    assert train.build_train_config("passage_reranking", "Qwen/Qwen3-4B", "o", "d", "t", "v", world_size=2)["template"] == "qwen3"


def test_set_overrides_are_typed():
    assert train.parse_overrides(["learning_rate=5e-5", "bf16=false", "report_to=none"]) == {
        "learning_rate": 5e-5, "bf16": False, "report_to": "none"}
    with pytest.raises(ValueError):
        train.parse_overrides(["oops"])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_task_and_method_prepares_a_run_end_to_end(task, method, tmp_path, table_raw, passage_raw, monkeypatch):
    monkeypatch.setenv("EKLAV_NUM_PROCESSES", "2")
    raw = passage_raw if task == "passage_reranking" else table_raw
    out = tmp_path / "run"
    rc = train.main(["--task", task, "--method", method, "--base_model", "Qwen/Qwen3-8B", "--output_dir", str(out),
                     "--raw_path", raw, "--data_root", str(tmp_path / "data"), "--per_device_train_batch_size", "2",
                     "--dry_run"])
    assert rc == 0
    cfg = yaml.safe_load(open(out / "train_config.yaml"))
    info = json.load(open(out / "dataset" / "dataset_info.json"))
    assert cfg["dataset"] in info and cfg["eval_dataset"] in info
    assert os.path.isfile(info[cfg["dataset"]]["file_name"])
    assert cfg["finetuning_type"] == "lora" and cfg["enable_thinking"] is True
