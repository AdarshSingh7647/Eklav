#!/usr/bin/env python3
"""
Single training entrypoint: pick a task, method, and base model; downloads
(or builds) the right sharegpt data if not already present, assembles a
LLaMA-Factory LoRA SFT YAML config with the correct per-task hyperparameters
and Eklav masking-plugin env vars, then launches `llamafactory-cli train`.

Usage:
  python -m eklav.train --task passage_reranking --method eklav \
      --base_model Qwen/Qwen3-8B --output_dir ./runs/passage_reranking_eklav

  python -m eklav.train --task math_reasoning --method std-sft \
      --base_model /local/path/to/Qwen3-4B --output_dir ./runs/math_std_sft

Methods: std-sft | answer-only | eklav
Tasks:   passage_reranking | table_reranking | math_reasoning
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = str(REPO_ROOT / "data")
DEFAULT_LLAMAFACTORY_DATA_DIR = os.environ.get("LLAMAFACTORY_DATA_DIR")

METHODS = ["std-sft", "answer-only", "eklav"]
TASKS = ["passage_reranking", "table_reranking", "math_reasoning"]

METHOD_TO_TABLE_DATA_METHOD = {
    "std-sft": "std-sft",
    "answer-only": "answer-only",
    "eklav": "eklav",
}


def get_data_for_task(task: str, method: str, data_root: str, args: argparse.Namespace) -> Dict[str, str]:
    if task == "passage_reranking":
        from eklav.data import passage_reranking as mod

        return mod.get_data(method, data_root, raw_arrow_path=args.raw_arrow_path)
    if task == "math_reasoning":
        from eklav.data import math_reasoning as mod

        return mod.get_data(method, data_root, raw_dataset_path=args.raw_dataset_path)
    if task == "table_reranking":
        from eklav.data import table_reranking as mod

        return mod.get_data(method, data_root, raw_path=args.raw_path)
    raise ValueError(f"Unknown task: {task}")


def register_dataset_info(llamafactory_data_dir: str, dataset_key: str, file_path: str) -> None:
    info_path = os.path.join(llamafactory_data_dir, "dataset_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    info[dataset_key] = {
        "file_name": file_path,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def build_yaml_config(task: str, method: str, base_model: str, output_dir: str,
                       train_dataset_key: str, val_dataset_key: str) -> str:
    from eklav.configs import TASK_CONFIGS

    cfg = TASK_CONFIGS[task]
    lines = [
        "### model",
        f"model_name_or_path: {base_model}",
        "trust_remote_code: true",
        "",
        "### method",
        "stage: sft",
        "do_train: true",
        "finetuning_type: lora",
        f"lora_rank: {cfg['lora_rank']}",
        f"lora_alpha: {cfg['lora_alpha']}",
        f"lora_target: {cfg['lora_target']}",
    ]
    if "lora_dropout" in cfg:
        lines.append(f"lora_dropout: {cfg['lora_dropout']}")
    lines += [
        "",
        "### dataset",
        f"dataset: {train_dataset_key}",
        f"template: {cfg['template']}",
    ]
    if "enable_thinking" in cfg:
        lines.append(f"enable_thinking: {'true' if cfg['enable_thinking'] else 'false'}")
    lines += [
        f"cutoff_len: {cfg['cutoff_len']}",
        "overwrite_cache: true",
        "preprocessing_num_workers: 16",
        "dataloader_num_workers: 4",
        "",
        "### mask_history",
        f"mask_history: {'true' if _mask_history_for(task, method, cfg) else 'false'}",
        "",
        "### output",
        f"output_dir: {output_dir}",
        "logging_steps: 10",
        f"save_steps: {cfg.get('save_steps', 500)}",
        f"save_total_limit: {cfg.get('save_total_limit', 2)}",
        "plot_loss: true",
        "overwrite_output_dir: false",
        "report_to: tensorboard",
        "",
        "### train",
        f"per_device_train_batch_size: {_per_device_batch(task, method, cfg)}",
        f"gradient_accumulation_steps: {_grad_accum(task, method, cfg)}",
        f"learning_rate: {cfg['learning_rate']}",
        f"num_train_epochs: {cfg['num_train_epochs']}",
        f"lr_scheduler_type: {cfg['lr_scheduler_type']}",
        f"warmup_ratio: {cfg['warmup_ratio']}",
        "bf16: true",
    ]
    if "seed" in cfg:
        lines.append(f"seed: {cfg['seed']}")
    lines += [
        "ddp_timeout: 180000000",
        "",
        "### eval",
        "val_size: 0.0",
        f"per_device_eval_batch_size: {cfg.get('per_device_eval_batch_size', _per_device_batch(task, method, cfg))}",
        "eval_strategy: steps",
        f"eval_dataset: {val_dataset_key}",
        f"eval_steps: {cfg.get('save_steps', 500)}",
    ]
    return "\n".join(lines) + "\n"


def _mask_history_for(task: str, method: str, cfg: Dict[str, Any]) -> bool:
    if task == "table_reranking":
        from eklav.configs.table_reranking import METHOD_MASK_HISTORY

        return METHOD_MASK_HISTORY.get(method, cfg.get("mask_history", False))
    return cfg.get("mask_history", False)


def _per_device_batch(task: str, method: str, cfg: Dict[str, Any]) -> int:
    if task == "passage_reranking" and method == "eklav":
        return cfg["per_device_train_batch_size_eklav"]
    return cfg["per_device_train_batch_size"]


def _grad_accum(task: str, method: str, cfg: Dict[str, Any]) -> int:
    if "gradient_accumulation_steps" in cfg:
        return cfg["gradient_accumulation_steps"]
    effective = cfg["effective_batch_size"]
    per_device = _per_device_batch(task, method, cfg)
    return max(1, effective // per_device)


def method_env(task: str, method: str) -> Dict[str, str]:
    from eklav.configs import TASK_CONFIGS

    module_name = {"passage_reranking": "passage_reranking", "table_reranking": "table_reranking",
                    "math_reasoning": "math_reasoning"}[task]
    import importlib

    mod = importlib.import_module(f"eklav.configs.{module_name}")
    return dict(mod.METHOD_ENV.get(method, {}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--base_model", required=True, help="Local path or HF repo id of the base model.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--llamafactory_data_dir", default=DEFAULT_LLAMAFACTORY_DATA_DIR,
                     help="Path to LLaMA-Factory's data/ dir (contains dataset_info.json).")
    ap.add_argument("--config_out", default=None, help="Where to write the generated YAML (default: <output_dir>/train_config.yaml)")
    ap.add_argument("--raw_arrow_path", default=None,
                     help="[passage_reranking/answer-only only] jhu-clsp/rank1-training-data arrow file.")
    ap.add_argument("--raw_dataset_path", default=None,
                     help="[math_reasoning/answer-only only] stage2-3k.json path.")
    ap.add_argument("--raw_path", default=None,
                     help="[table_reranking, any method] sft_examples.json path.")
    ap.add_argument("--dry_run", action="store_true", help="Write the config but do not launch training.")
    args = ap.parse_args()

    if not args.llamafactory_data_dir:
        ap.error("--llamafactory_data_dir is required (or set LLAMAFACTORY_DATA_DIR) so datasets can be "
                  "registered in LLaMA-Factory's dataset_info.json.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)

    print(f"[eklav] Resolving data for task={args.task} method={args.method} ...")
    paths = get_data_for_task(args.task, args.method, args.data_root, args)
    print(f"[eklav] Data ready: {paths}")

    dataset_key_prefix = f"eklav_{args.task}_{args.method}"
    train_key, val_key = f"{dataset_key_prefix}_train", f"{dataset_key_prefix}_val"
    register_dataset_info(args.llamafactory_data_dir, train_key, paths["train"])
    register_dataset_info(args.llamafactory_data_dir, val_key, paths["val"])
    print(f"[eklav] Registered datasets '{train_key}' / '{val_key}' in {args.llamafactory_data_dir}/dataset_info.json")

    cfg_text = build_yaml_config(args.task, args.method, args.base_model, args.output_dir, train_key, val_key)
    config_out = args.config_out or os.path.join(args.output_dir, "train_config.yaml")
    with open(config_out, "w") as f:
        f.write(cfg_text)
    print(f"[eklav] Wrote LLaMA-Factory config to {config_out}")

    env_overrides = method_env(args.task, args.method)
    env = dict(os.environ)
    env.update(env_overrides)
    if env_overrides:
        print(f"[eklav] Masking plugin env vars: {env_overrides}")

    cmd = ["llamafactory-cli", "train", config_out]
    print(f"[eklav] Launch command: {' '.join(f'{k}={v}' for k, v in env_overrides.items())} {' '.join(cmd)}")

    if args.dry_run:
        print("[eklav] --dry_run set, not launching training.")
        return 0

    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
