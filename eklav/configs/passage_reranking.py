"""
Passage reranking LoRA SFT hyperparameters, read from
model-forge/code/passage_reranking/make_configs_qwen3_8b.py -- the
authoritative generator for the paper's Qwen3 checkpoints (make_configs.py
is the older Qwen2.5-7B-only generator and is not used here).

cutoff_len=2500 here, NOT the 32768 figure in experimental_setup.tex's shared
default section -- the qwen3_8b generator (which produced the paper's actual
checkpoints) hardcodes 2500, so that's what's used.

The reference halves per_device_train_batch_size for its cotcond run (64->32)
purely because that specific run shared a GPU with another job and hit OOMs;
gradient_accumulation_steps auto-adjusts to keep effective_batch_size at 128
either way, so training outcome is identical. Not reproduced here -- a single
per_device_train_batch_size applies to all methods.
"""

CONFIG = {
    "template": "qwen3",
    "enable_thinking": True,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "learning_rate": 1.0e-4,
    "num_train_epochs": 2.0,
    "effective_batch_size": 128,
    "cutoff_len": 2500,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "seed": 12345,
    "per_device_train_batch_size": 64,
    "save_steps": 250,
    "save_total_limit": 2,
    "mask_history": False,
}

# Env var set at launch per method (mask mechanism is sub-turn, not a YAML key).
METHOD_ENV = {
    "eklav": {"EKLAV_ANSWER_ONLY_AFTER_TOKEN": "</think>"},
    # Ablation: std-sft data/layout (full trace in the response) with only the
    # <think>...</think> span masked from the loss, instead of Eklav's
    # prompt-hint layout. Isolates masking from context placement + filtering.
    "eklav-mask-only": {"EKLAV_THINK_CONTENT_MASK": "1"},
}
