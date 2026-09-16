"""
Passage reranking LoRA SFT hyperparameters, read from
model-forge/code/passage_reranking/make_configs_qwen3_8b.py -- the
authoritative generator for the paper's Qwen3 checkpoints (make_configs.py
is the older Qwen2.5-7B-only generator and is not used here).

cutoff_len=2500 here, NOT the 32768 figure in experimental_setup.tex's shared
default section -- the qwen3_8b generator (which produced the paper's actual
checkpoints) hardcodes 2500, so that's what's used.
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
    "per_device_train_batch_size_eklav": 32,  # eklav override in reference: shares a GPU, halved headroom
    "save_steps": 750,
    "save_total_limit": 2,
    "mask_history": False,
}

# Env var set at launch per method (mask mechanism is sub-turn, not a YAML key).
METHOD_ENV = {
    "eklav": {"EKLAV_ANSWER_ONLY_AFTER_TOKEN": "</think>"},
}
