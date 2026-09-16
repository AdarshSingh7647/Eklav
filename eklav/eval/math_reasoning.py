#!/usr/bin/env python3
"""
Math reasoning eval: vLLM generation, brace-balanced \\boxed{} extraction,
math_verify symbolic equivalence (whitespace-normalized fallback), exact
letter match for MCQ (GPQA-Diamond, MMLU). Ported from
model-forge/code/Math_Reasoning/eval_aime_checkpoints.py and its dependency
iclr/eval/QA_Reasoning/qa_common.py.

Benchmarks: AIME 2024, AIME 2025, AIME 1983-2024, GSM8K, MATH-500,
Omni-MATH, GPQA-Diamond, MMLU. Sampling: context_size=32768,
max_output_tokens=8192, temperature=0.6/top_p=0.95/top_k=20 (Qwen3
recommended thinking-mode settings) -- copied exactly from the reference,
not invented.

Saves one JSONL per (model, method, benchmark) with: prompt, raw output,
parsed answer, gold answer, row id, benchmark name, model/checkpoint tag.
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eklav.eval.metrics import extract_boxed, extract_mcq_letter, math_answers_equal, mcq_answers_equal

CONTEXT_SIZE = 32768
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

SYSTEM_PROMPT = (
    "You are a careful math assistant that reasons step-by-step and reports a "
    "final answer within \\boxed{}."
)

MATH_CUE = "\n\nReason step-by-step, then give the final answer as \\boxed{answer}."
MCQ_CUE = (
    "\n\nReason step-by-step, then give the final answer as \\boxed{X}, "
    "where X is the letter (A, B, C, or D) of the correct option."
)
_LETTERS = ["A", "B", "C", "D"]


def build_math_prompt(question: str) -> str:
    return f"{question}{MATH_CUE}"


def build_mcq_prompt(question: str, choices: Sequence[str]) -> str:
    options = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{options}{MCQ_CUE}"


@dataclass
class Row:
    row_id: str
    prompt: str
    gold: str
    kind: str  # "math" | "mcq"
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Benchmark loaders
# --------------------------------------------------------------------------- #

def load_gpqa_diamond(seed: int = 42) -> List[Row]:
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(seed)
    rows = []
    for i, ex in enumerate(ds):
        opts = [ex["Correct Answer"].strip(), ex["Incorrect Answer 1"].strip(),
                ex["Incorrect Answer 2"].strip(), ex["Incorrect Answer 3"].strip()]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[j] for j in order]
        gold_letter = _LETTERS[order.index(0)]
        rows.append(Row(f"gpqa_diamond_{i}", build_mcq_prompt(ex["Question"].strip(), shuffled), gold_letter, "mcq"))
    return rows


def load_aime2025() -> List[Row]:
    from datasets import load_dataset

    rows = []
    for part in ("AIME2025-I", "AIME2025-II"):
        ds = load_dataset("opencompass/AIME2025", part, split="test")
        for i, ex in enumerate(ds):
            rows.append(Row(f"aime2025_{part}_{i}", build_math_prompt(ex["question"].strip()),
                             str(ex["answer"]).strip(), "math"))
    return rows


def load_aime_1983_2024(limit: Optional[int] = None, seed: int = 42) -> List[Row]:
    from datasets import load_dataset

    ds = load_dataset("gneubig/aime-1983-2024", split="train")
    idx = list(range(len(ds)))
    if limit is not None and limit < len(idx):
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:limit])
    rows = []
    for i in idx:
        ex = ds[i]
        rows.append(Row(f"aime8324_{ex['ID']}", build_math_prompt(ex["Question"].strip()),
                         str(ex["Answer"]).strip(), "math", {"year": ex["Year"]}))
    return rows


def load_aime2024() -> List[Row]:
    rows = [r for r in load_aime_1983_2024() if str(r.meta.get("year")) == "2024"]
    for r in rows:
        r.row_id = r.row_id.replace("aime8324_", "aime2024_")
    return rows


def load_gsm8k(limit: Optional[int] = None, seed: int = 42) -> List[Row]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    if limit is not None and limit < len(idx):
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:limit])
    rows = []
    for i in idx:
        ex = ds[i]
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        rows.append(Row(f"gsm8k_{i}", build_math_prompt(ex["question"].strip()), gold, "math"))
    return rows


def load_math500(limit: Optional[int] = None, seed: int = 42) -> List[Row]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    idx = list(range(len(ds)))
    if limit is not None and limit < len(idx):
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:limit])
    rows = []
    for i in idx:
        ex = ds[i]
        rows.append(Row(f"math500_{i}", build_math_prompt(ex["problem"].strip()), str(ex["answer"]).strip(), "math",
                         {"subject": ex.get("subject"), "level": ex.get("level")}))
    return rows


def load_omnimath(limit: Optional[int] = None, seed: int = 42) -> List[Row]:
    from datasets import load_dataset

    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    idx = list(range(len(ds)))
    if limit is not None and limit < len(idx):
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:limit])
    rows = []
    for i in idx:
        ex = ds[i]
        rows.append(Row(f"omnimath_{i}", build_math_prompt(ex["problem"].strip()), str(ex["answer"]).strip(), "math",
                         {"domain": ex.get("domain"), "difficulty": ex.get("difficulty")}))
    return rows


def load_mmlu(limit: Optional[int] = None, seed: int = 42, subjects: Optional[Sequence[str]] = None) -> List[Row]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    if subjects is not None:
        subj_set = set(subjects)
        ds = ds.filter(lambda ex: ex["subject"] in subj_set)
    idx = list(range(len(ds)))
    if limit is not None and limit < len(idx):
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:limit])
    rows = []
    for i in idx:
        ex = ds[i]
        rows.append(Row(f"mmlu_{i}", build_mcq_prompt(ex["question"].strip(), ex["choices"]),
                         _LETTERS[ex["answer"]], "mcq", {"subject": ex["subject"]}))
    return rows


BENCHMARK_LOADERS = {
    "aime2024": load_aime2024,
    "aime2025": load_aime2025,
    "aime_1983_2024": load_aime_1983_2024,
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "omnimath": load_omnimath,
    "gpqa_diamond": load_gpqa_diamond,
    "mmlu": load_mmlu,
}


# --------------------------------------------------------------------------- #
# vLLM generation wrapper
# --------------------------------------------------------------------------- #

class QAModel:
    def __init__(self, model_path: str, num_gpus: int = 1, context_size: int = CONTEXT_SIZE,
                 max_output_tokens: int = MAX_OUTPUT_TOKENS, dtype: str = "bfloat16",
                 gpu_memory_utilization: float = 0.95, temperature: float = TEMPERATURE,
                 top_p: float = TOP_P, top_k: float = TOP_K, lora_path: Optional[str] = None,
                 lora_rank: int = 32):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.lora_path = lora_path
        self._lora_request = None
        tokenizer_source = lora_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

        llm_kwargs = dict(model=model_path, tokenizer=model_path, trust_remote_code=True, dtype=dtype,
                           max_model_len=context_size, gpu_memory_utilization=gpu_memory_utilization,
                           tensor_parallel_size=num_gpus)
        if lora_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = lora_rank
        self.llm = LLM(**llm_kwargs)

        if lora_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("eval_adapter", 1, lora_path)

        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k,
                                               max_tokens=max_output_tokens)

    def _format_prompt(self, user_content: str, system: str = SYSTEM_PROMPT) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=True)

    def generate_with_counts(self, user_contents: Sequence[str], system: str = SYSTEM_PROMPT) -> List[Tuple[str, int]]:
        prompts = [self._format_prompt(c, system=system) for c in user_contents]
        outputs = self.llm.generate(prompts, self.sampling_params, lora_request=self._lora_request)
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outputs]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def run_pass1_eval(model: QAModel, rows: Sequence[Row], out_dir: str, tag: str, benchmark_name: str,
                    model_tag: str, batch_size: int = 32) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    records: List[Dict[str, Any]] = []
    n = len(rows)
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch = rows[start:start + batch_size]
        outputs = model.generate_with_counts([r.prompt for r in batch])
        for r, (out_text, n_tok) in zip(batch, outputs):
            boxed = extract_boxed(out_text)
            if r.kind == "mcq":
                correct = mcq_answers_equal(r.gold, out_text)
                pred = extract_mcq_letter(out_text)
            else:
                correct = math_answers_equal(r.gold, boxed)
                pred = boxed
            records.append({
                "row_id": r.row_id, "benchmark": benchmark_name, "kind": r.kind,
                "prompt": r.prompt, "raw_output": out_text,
                "parsed_answer": pred, "gold_answer": r.gold, "correct": bool(correct),
                "has_boxed": boxed is not None, "num_tokens": n_tok,
                "model_tag": model_tag, "meta": r.meta,
            })
        done = min(start + batch_size, n)
        print(f"[{tag}] {done}/{n} done ({time.time() - t0:.0f}s elapsed)")

    with open(os.path.join(out_dir, f"{tag}.jsonl"), "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_correct = sum(r["correct"] for r in records)
    summary = {"tag": tag, "benchmark": benchmark_name, "n": n, "n_correct": n_correct,
               "pass_at_1": n_correct / n if n else float("nan")}
    with open(os.path.join(out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{tag}] pass@1 = {summary['pass_at_1']:.4f} ({n_correct}/{n})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARK_LOADERS),
                     help=f"Subset of {list(BENCHMARK_LOADERS)}")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows per benchmark (debug runs)")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    model = QAModel(args.model_path, num_gpus=args.num_gpus, lora_path=args.lora_path, lora_rank=args.lora_rank)

    no_limit_arg = {"aime2024", "aime2025", "gpqa_diamond"}
    all_summaries = {}
    for bench_name in args.benchmarks:
        loader = BENCHMARK_LOADERS[bench_name]
        rows = loader() if bench_name in no_limit_arg else loader(limit=args.limit)
        if args.limit and bench_name in no_limit_arg and len(rows) > args.limit:
            rows = rows[:args.limit]
        tag = f"{args.model_tag}_{bench_name}"
        out_dir = os.path.join(args.out_dir, bench_name)
        all_summaries[bench_name] = run_pass1_eval(model, rows, out_dir, tag, bench_name, args.model_tag,
                                                     batch_size=args.batch_size)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
