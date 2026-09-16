#!/usr/bin/env python3
"""
Merge best-eval-loss Eklav (cotcond) and last-checkpoint std-SFT (cotgen)
math-reasoning LoRA adapters into full models, upload each to
AdarshSingh7647/Eklav-Question-Answering on HF, then delete the local
adapter + merged copies before moving to the next model.

Run one model at a time (--model) so disk usage never exceeds one merged
model's footprint at a time.
"""
import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo

CHECKPOINT_ROOT = "/mnt/data2/asing725_2/forge/downloads/model_forge_checkpoints/Math_Reasoning"
HF_REPO = "AdarshSingh7647/Eklav-Question-Answering"

MODELS = {
    "qwen3_0_6b": {
        "base_model": "/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        "eklav_dir": "cotcond_bare",
        "stdsft_dir": "cotgen",
    },
    "qwen3_4b": {
        "base_model": "/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "eklav_dir": "cotcond_bare",
        "stdsft_dir": None,
    },
    "qwen3_8b": {
        "base_model": "/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        "eklav_dir": "cotcond_bare",
        "stdsft_dir": None,
    },
    "qwen3_14b": {
        "base_model": "/mnt/shared/shared_hf_home/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18",
        "eklav_dir": "cotcond_bare",
        "stdsft_dir": None,
    },
    "glm_z1_9b": {
        "base_model": "/mnt/data2/asing725_2/forge/downloads/models/GLM-Z1-9B-0414",
        "eklav_dir": "cotcond_bare",
        "stdsft_dir": None,
    },
    "phi4_mini_reasoning": {
        "base_model": "/mnt/data2/asing725_2/forge/downloads/models/Phi-4-mini-reasoning",
        "eklav_dir": "cotcond_bare_explicit_think",
        "stdsft_dir": "cotgen",
    },
}


def best_checkpoint_by_eval_loss(adapter_root: str) -> str:
    metrics_path = os.path.join(adapter_root, "checkpoint_metrics.jsonl")
    log_path = os.path.join(adapter_root, "trainer_log.jsonl")
    best_step, best_loss = None, float("inf")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                d = json.loads(line)
                if "eval_loss" in d and d["eval_loss"] < best_loss:
                    best_loss, best_step = d["eval_loss"], d["step"]
    elif os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                d = json.loads(line)
                if "eval_loss" in d and d["eval_loss"] < best_loss:
                    best_loss, best_step = d["eval_loss"], d["current_steps"]
    if best_step is None:
        raise RuntimeError(f"no eval_loss found under {adapter_root}")
    return os.path.join(adapter_root, f"checkpoint-{best_step}"), best_loss


def last_checkpoint(adapter_root: str) -> str:
    steps = []
    for name in os.listdir(adapter_root):
        if name.startswith("checkpoint-"):
            try:
                steps.append(int(name.split("-")[1]))
            except ValueError:
                continue
    if not steps:
        raise RuntimeError(f"no checkpoints found under {adapter_root}")
    return os.path.join(adapter_root, f"checkpoint-{max(steps)}")


def merge_and_upload(base_model_path: str, adapter_path: str, merged_dir: str,
                      hf_subfolder: str) -> None:
    print(f"Loading base model {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Loading adapter {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Merging")
    model = model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del model
    torch.cuda.empty_cache()

    print(f"Uploading {merged_dir} -> {HF_REPO}/{hf_subfolder}")
    api = HfApi()
    api.upload_folder(
        repo_id=HF_REPO,
        repo_type="model",
        folder_path=merged_dir,
        path_in_repo=hf_subfolder,
        commit_message=f"Add merged {hf_subfolder}",
    )

    shutil.rmtree(merged_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--merge_scratch", default="/mnt/data2/asing725_2/tmp/eklav_merge_scratch")
    ap.add_argument("--delete_local_after_upload", action="store_true")
    ap.add_argument("--create_repo", action="store_true")
    args = ap.parse_args()

    if args.create_repo:
        create_repo(HF_REPO, repo_type="model", private=False, exist_ok=True)

    cfg = MODELS[args.model]
    base_model = cfg["base_model"]

    eklav_root = os.path.join(CHECKPOINT_ROOT, args.model, cfg["eklav_dir"])
    eklav_ckpt, eklav_loss = best_checkpoint_by_eval_loss(eklav_root)
    print(f"[{args.model}] best eklav checkpoint: {eklav_ckpt} (eval_loss={eklav_loss})")
    eklav_merged = os.path.join(args.merge_scratch, f"{args.model}_eklav_merged")
    merge_and_upload(base_model, eklav_ckpt, eklav_merged, f"{args.model}/eklav")
    if args.delete_local_after_upload:
        shutil.rmtree(eklav_root)
        print(f"Deleted {eklav_root}")

    if cfg["stdsft_dir"]:
        stdsft_root = os.path.join(CHECKPOINT_ROOT, args.model, cfg["stdsft_dir"])
        stdsft_ckpt = last_checkpoint(stdsft_root)
        print(f"[{args.model}] last std-sft checkpoint: {stdsft_ckpt}")
        stdsft_merged = os.path.join(args.merge_scratch, f"{args.model}_stdsft_merged")
        merge_and_upload(base_model, stdsft_ckpt, stdsft_merged, f"{args.model}/std-sft")
        if args.delete_local_after_upload:
            shutil.rmtree(stdsft_root)
            print(f"Deleted {stdsft_root}")
    else:
        print(f"[{args.model}] no local std-sft checkpoint found, skipping")


if __name__ == "__main__":
    main()
