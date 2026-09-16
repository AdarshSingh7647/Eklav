"""
Table reranking LoRA SFT hyperparameters, read from
model-forge/code/tabular_reranking/make_configs.py.

Deliberately different from passage_reranking/math_reasoning (smaller LoRA
rank, longer cutoff, more epochs, gentler warmup) -- preserved as-is, not
reconciled to match the other tasks.
"""

CONFIG = {
    "template": "qwen3",
    "enable_thinking": False,  # build_prompts_v3.py builds windows with enable_thinking=False at train-data-build time
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "learning_rate": 1.0e-4,
    "num_train_epochs": 3.0,
    "cutoff_len": 28672,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "save_steps": 200,
    "mask_history": False,  # per-method override in METHOD_MASK_HISTORY
}

METHOD_MASK_HISTORY = {
    "answer-only": True,  # setup2_answer_only: multi-turn, mask_history required
}

# No sub-turn masking env var needed for any table_reranking method: eklav's
# response is a bare answer JSON with no <think> block, so loss falls
# naturally on the answer turn; answer-only uses stock mask_history=true.
METHOD_ENV = {}
