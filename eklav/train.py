"""
Train a LoRA SFT model for any task and method.

    eklav train --task passage_reranking --method eklav \
        --base_model Qwen/Qwen3-8B --output_dir runs/passage_eklav \
        --raw_path data/raw/rank1_training_data.arrow

Builds the training data if it is not already there, writes a self-contained
LLaMA-Factory config (hyperparameters from the task, loss mask from the method),
and launches `llamafactory-cli train` with the answer-only masking plugin loaded
in every process, including multi-GPU workers.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

import yaml

from eklav.data import get_data
from eklav.lf_plugin import startup_dir
from eklav.methods import METHODS, loss_mask_env
from eklav.tasks import TASKS, get_task

DEFAULT_DATA_ROOT = "data"

# Base models whose chat template differs from the tasks' default (tuned for the
# Qwen3 family). Matched case-insensitively against the tail of --base_model.
# glm-z1-9b-0414 is a reasoning model with its own chat format ("glmz1").
TEMPLATE_OVERRIDES = {
    "glm-z1-9b-0414": "glmz1",
}


def resolve_template(base_model: str, default: str) -> str:
    tail = os.path.basename(base_model.rstrip("/")).lower()
    return TEMPLATE_OVERRIDES.get(tail, default)


def num_processes() -> int:
    """Data-parallel world size the launch will use: EKLAV_NUM_PROCESSES if set,
    else the visible GPU count (llamafactory-cli starts one process per GPU)."""
    override = os.environ.get("EKLAV_NUM_PROCESSES")
    if override:
        return max(1, int(override))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        return max(1, len([d for d in visible.split(",") if d.strip()]))
    try:
        import torch

        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except ImportError:
        pass
    return 1


def gradient_accumulation(effective_batch: int, per_device: int, world_size: int) -> int:
    """Accumulation steps so that per_device * accumulation * world_size == effective_batch.
    LLaMA-Factory multiplies by the world size, so it must divide out here."""
    denom = per_device * max(1, world_size)
    if effective_batch % denom != 0:
        raise ValueError(
            f"effective_batch_size={effective_batch} is not divisible by per_device_train_batch_size="
            f"{per_device} x num_processes={world_size} (={denom}). Pick a per-device batch whose product "
            f"with the GPU count divides {effective_batch}."
        )
    return effective_batch // denom


def register_dataset(dataset_dir: str, key: str, file_path: str) -> None:
    """Add a sharegpt dataset to LLaMA-Factory's dataset_info.json."""
    os.makedirs(dataset_dir, exist_ok=True)
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    info[key] = {
        "file_name": os.path.abspath(file_path),
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def build_train_config(task_name: str, base_model: str, output_dir: str, dataset_dir: str,
                       train_key: str, val_key: str, per_device_batch: Optional[int] = None,
                       world_size: int = 1, deepspeed: Optional[str] = None,
                       overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """LLaMA-Factory LoRA SFT config for `task_name`; method-specific behaviour
    lives in the data and the masking env vars, not in this config."""
    hp = get_task(task_name).hparams
    per_device = per_device_batch if per_device_batch is not None else hp["per_device_train_batch_size"]

    config: Dict[str, Any] = {
        "model_name_or_path": base_model,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": hp["lora_rank"],
        "lora_alpha": hp["lora_alpha"],
        "lora_target": hp["lora_target"],
    }
    if deepspeed:
        config["deepspeed"] = deepspeed
    config.update({
        "dataset_dir": dataset_dir,
        "dataset": train_key,
        "template": resolve_template(base_model, hp["template"]),
        "enable_thinking": hp["enable_thinking"],
        "cutoff_len": hp["cutoff_len"],
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "dataloader_num_workers": 4,
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": hp["save_steps"],
        "save_total_limit": hp["save_total_limit"],
        "plot_loss": True,
        "overwrite_output_dir": False,
        "report_to": "tensorboard",
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": gradient_accumulation(hp["effective_batch_size"], per_device, world_size),
        "learning_rate": hp["learning_rate"],
        "num_train_epochs": hp["num_train_epochs"],
        "lr_scheduler_type": hp["lr_scheduler_type"],
        "warmup_ratio": hp["warmup_ratio"],
        "bf16": True,
    })
    if "seed" in hp:
        config["seed"] = hp["seed"]
    config.update({
        "ddp_timeout": 180000000,
        "val_size": 0.0,
        "per_device_eval_batch_size": per_device,
        "eval_strategy": "steps",
        "eval_dataset": val_key,
        "eval_steps": hp["save_steps"],
    })
    config.update(overrides or {})
    return config


def parse_value(text: str) -> Any:
    """A command-line value as a number, boolean or string. JSON is tried first because
    YAML 1.1 reads exponent forms such as 5e-5 as strings."""
    try:
        return json.loads(text)
    except ValueError:
        return yaml.safe_load(text)


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """--set key=value pairs."""
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = parse_value(value)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="eklav train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--base_model", required=True, help="Local path or HF repo id of the base model.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--raw_path", default=None,
                    help="Raw training corpus, used to build the data if it is not built yet "
                         "(passage_reranking: Rank1 arrow file; table_reranking: sft_examples.json).")
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="Where built training data lives.")
    ap.add_argument("--dataset_dir", default=None,
                    help="LLaMA-Factory dataset directory (default: <output_dir>/dataset).")
    ap.add_argument("--per_device_train_batch_size", type=int, default=None,
                    help="Memory knob; gradient accumulation is recomputed so the effective batch is unchanged.")
    ap.add_argument("--num_processes", type=int, default=None,
                    help="GPUs used for data-parallel training (default: inferred from the visible GPUs).")
    ap.add_argument("--deepspeed", default=None, help="DeepSpeed config JSON (e.g. ZeRO-2/3).")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="Override any LLaMA-Factory config key, e.g. --set learning_rate=5e-5. Repeatable.")
    ap.add_argument("--config_out", default=None, help="Where to write the config (default: <output_dir>/train_config.yaml).")
    ap.add_argument("--dry_run", action="store_true", help="Build data and write the config, but do not train.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    task = get_task(args.task)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[eklav] Resolving data for {args.task} / {args.method} ...")
    paths = get_data(task.data, args.method, args.data_root, args.raw_path)
    print(f"[eklav] Data ready: {paths}")

    dataset_dir = args.dataset_dir or os.path.join(args.output_dir, "dataset")
    prefix = f"eklav_{args.task}_{args.method}"
    train_key, val_key = f"{prefix}_train", f"{prefix}_val"
    register_dataset(dataset_dir, train_key, paths["train"])
    register_dataset(dataset_dir, val_key, paths["val"])

    world_size = args.num_processes if args.num_processes is not None else num_processes()
    config = build_train_config(
        args.task, args.base_model, args.output_dir, dataset_dir, train_key, val_key,
        per_device_batch=args.per_device_train_batch_size, world_size=world_size,
        deepspeed=args.deepspeed, overrides=parse_overrides(args.overrides),
    )
    config_path = args.config_out or os.path.join(args.output_dir, "train_config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"[eklav] Wrote {config_path}")
    print(f"[eklav] Batch: per_device={config['per_device_train_batch_size']} x "
          f"accumulation={config['gradient_accumulation_steps']} x processes={world_size} "
          f"= effective {config['per_device_train_batch_size'] * config['gradient_accumulation_steps'] * world_size}")

    env = dict(os.environ)
    env.update(loss_mask_env(args.method))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [startup_dir(), env.get("PYTHONPATH")]))
    mask = loss_mask_env(args.method)
    print(f"[eklav] Loss mask: {'answer only (' + next(iter(mask.values())) + ')' if mask else 'full response'}")

    if args.dry_run:
        print("[eklav] --dry_run set, not launching training.")
        return 0
    if shutil.which("llamafactory-cli") is None:
        print("[eklav] llamafactory-cli not found; install LLaMA-Factory (pip install llamafactory).", file=sys.stderr)
        return 1
    return subprocess.run(["llamafactory-cli", "train", config_path], env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
