"""
Math reasoning LoRA SFT hyperparameters, read from
model-forge/code/Math_Reasoning/make_configs_qwen3_8b.py (same recipe used
across the Math_Reasoning model-size configs, per that file's own docstring).
"""

CONFIG = {
    "template": "qwen3",
    "enable_thinking": True,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_target": "all",
    "learning_rate": 1.0e-4,
    "num_train_epochs": 3.0,
    "effective_batch_size": 128,
    "cutoff_len": 32768,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "seed": 12345,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "save_total_limit": 4,
    "target_n_checkpoints": 4,
    "mask_history": False,
}

# Env var set at launch per method (mask mechanism is sub-turn, not a YAML key).
METHOD_ENV = {
    "eklav": {"EKLAV_THINK_CONTENT_MASK": "1"},
}
