"""vLLM model loading shared by the evaluators."""

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    model_path: str                       # merged model, or the base model when lora_path is given
    lora_path: Optional[str] = None
    lora_rank: int = 32                   # must be >= the adapter's rank
    num_gpus: int = 1
    max_model_len: Optional[int] = None   # None: the task's default
    max_output_tokens: int = 4096
    gpu_memory_utilization: float = 0.90


def sanitize_tokenizer_config(source: str) -> None:
    """Drop a stale list-valued 'extra_special_tokens' from tokenizer_config.json.
    Checkpoints saved by older LLaMA-Factory/transformers store it as a list, which
    current transformers rejects ("'list' object has no attribute 'keys'").

    `source` is a local directory or a Hub repo id; for a repo id the cached copy is
    patched, using the cache directory transformers itself will read."""
    if os.path.isdir(source):
        config_path = os.path.join(source, "tokenizer_config.json")
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            config_path = hf_hub_download(source, "tokenizer_config.json", cache_dir=os.environ.get("TRANSFORMERS_CACHE"))
        except EntryNotFoundError:
            return
    if not os.path.isfile(config_path):
        return
    with open(config_path) as f:
        config = json.load(f)
    if isinstance(config.get("extra_special_tokens"), list):
        del config["extra_special_tokens"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def load_tokenizer(args: ModelArgs):
    from transformers import AutoTokenizer

    source = args.lora_path or args.model_path
    sanitize_tokenizer_config(source)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_llm(args: ModelArgs, max_model_len: int):
    """(vLLM engine, LoRARequest or None)."""
    from vllm import LLM

    kwargs = dict(
        model=args.model_path, tensor_parallel_size=max(1, int(args.num_gpus)), trust_remote_code=True,
        max_model_len=max_model_len, gpu_memory_utilization=args.gpu_memory_utilization, dtype="bfloat16",
    )
    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest

        kwargs.update(enable_lora=True, max_lora_rank=args.lora_rank)
        lora_request = LoRARequest("eval_adapter", 1, args.lora_path)
    return LLM(**kwargs), lora_request
