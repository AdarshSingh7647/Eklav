# Eklav

Eklav keeps a teacher's reasoning trace in the training input but supervises only
the final task answer. This repository trains and evaluates it for **passage
reranking** (pointwise true/false) and **table reranking** (listwise JSON ranking)
with LoRA SFT in LLaMA-Factory, against the paper's baselines **STD-CoT** and
**NAIVE-SFT** and the **MASK-ONLY** ablation.

## Install

```bash
pip install -e ".[train,eval]"      # `train`: LLaMA-Factory.  `eval`: vLLM and the retrieval stack.
```

## Quickstart

```bash
# Train any task with any method. The data is built (or reassembled) on first use.
eklav train --task passage_reranking --method eklav \
    --base_model Qwen/Qwen3-8B --output_dir runs/passage_eklav

eklav train --task table_reranking --method mask-only \
    --base_model Qwen/Qwen3-8B --output_dir runs/table_mask_only \
    --raw_path /path/to/sft_examples.json

# Evaluate on a benchmark
eklav eval --task passage_reranking --benchmark bright \
    --model_path Qwen/Qwen3-8B --lora_path runs/passage_eklav \
    --model_tag qwen3_8b_eklav --out_dir results/bright
```

`--method` is `std-cot`, `naive-sft`, `eklav` or `mask-only`; `--base_model` is any
local checkpoint or Hub id. `eklav train` writes a LLaMA-Factory config, then
launches `llamafactory-cli train` with the loss-masking plugin loaded in every
process, multi-GPU workers included. Add `--dry_run` to stop after writing the
config, and `--set key=value` to override any config entry
(`--set learning_rate=5e-5`). Run `eklav <command> --help` for all options.

## Methods

An example is `(x, r, a)`: task input, teacher reasoning trace, answer. The
student's own reasoning is never part of a training sequence; at evaluation all
teacher context is removed and the student generates its own reasoning before
answering. A method is a serialized sequence plus a loss mask (defined once in
[eklav/methods.py](eklav/methods.py)):

| Method | Teacher trace in training | Sequence | Loss on |
|---|---|---|---|
| **NAIVE-SFT** | none | `[x; a]` | answer |
| **STD-CoT** | in the response, as the imitation target | `[x; r + a]` | trace and answer |
| **EKLAV** | in the prompt, as a hint `h(r)` | `[x + h(r); a]` | answer |
| **MASK-ONLY** | in the response, loss masked (no hint) | `[x; r + a]` | answer |

"Answer" means the answer tokens plus the chat template's end-of-turn delimiters.
Reasoning is written `<think>...</think>`. A response without reasoning is the bare
answer, and the chat template then inserts an empty reasoning block; every
answer-only mask excludes it as well. MASK-ONLY trains on exactly STD-CoT's data
and differs only in the loss mask, which isolates masking from EKLAV's hint.

## Tasks

| Task | Raw training corpus | Hint `h(r)` | Cutoff |
|---|---|---|---|
| `passage_reranking` | Rank1 training set (`jhu-clsp/rank1-training-data`), 385,791 examples | first half of the trace's sentences, verdict-bearing sentences removed | 2,500 |
| `table_reranking` | 15,674 DeepSeek-R1 traces (`sft_examples.json`) | the complete trace | 32,768 |

**Passage reranking.** The hint is shorter than the trace twice over: the second half
is never shown, and verdict-bearing sentences (`VERDICT_RE` in
[eklav/tasks/passage_reranking.py](eklav/tasks/passage_reranking.py)) are dropped from
the half that is. The EKLAV sequence is therefore shorter than STD-CoT's: about 24%
fewer processed tokens, about 30% fewer with the padding a batched trainer computes.
That shorter sequence is the source of EKLAV's training-compute saving; masking loss
alone does not change the FLOPs spent on a sequence.

**Table reranking.** Truncating a multi-candidate comparison would drop candidates
from consideration, so the whole trace is the hint. The trace is only ~14% of a table
sequence, so EKLAV is not shorter than STD-CoT on this task.

All methods of a task share one hyperparameter set (paper Sec. 4): LoRA rank 32 /
alpha 64 on all linear layers, lr 1e-4, cosine schedule with 5% warmup, effective
batch 128, bf16, two epochs. `--per_device_train_batch_size` is a memory knob only;
gradient accumulation is recomputed so the effective batch stays 128.

## Training data

Data lives at `data/<task>/<layout>/{train,val}.json` in LLaMA-Factory's sharegpt
format, one layout per distinct sequence: `std-cot` (also used by MASK-ONLY),
`naive-sft` and `eklav`. Training data is built from the raw corpus the first time
a run needs it, or explicitly:

```bash
eklav data build --task passage_reranking --method eklav --raw_path data/raw/rank1_training_data.arrow
```

## Evaluation

Both reranking tasks evaluate on any retrieval benchmark. List the registered ones with
`eklav eval --list_benchmarks`: **BRIGHT** (12 domains, scored on the BM25 top-100
pools released with Rank1) and the IBM table benchmarks **AITQA, FeTaQA, OTT-QA,
OpenWikiTables and NQ-Tables**.

A new benchmark needs no code if it is in BEIR layout (`corpus.jsonl`,
`queries.jsonl`, `qrels/<split>.tsv`) or a Hub dataset with `default` (qrels),
`queries` and `corpus` configs:

```bash
eklav eval --task table_reranking --benchmark local:/data/my_benchmark --split test ...
eklav eval --task passage_reranking --benchmark hf:org/my-benchmark ...
```

or register a loader in Python (see [eklav/eval/benchmarks.py](eklav/eval/benchmarks.py)):

```python
from eklav.eval.benchmarks import Benchmark, RetrievalData, register

register(Benchmark("my-benchmark", loader=lambda subset: RetrievalData(queries, corpus, qrels)))
```

A benchmark that ships no candidate pools gets one from BM25 (passage reranking) or
the BM25 + SPLADE + mpnet reciprocal-rank-fusion hybrid (table reranking);
`--first_stage` overrides that. Each run writes `<subset>.jsonl` (full prompts and raw
generations), `<subset>_summary.json` and `summary.json`, reporting nDCG, recall,
precision and MRR at several cutoffs. Table reranking also reports how many queries had
an unparseable ranking. `--num_questions N --seed S` evaluates a reproducible sample.

## Loss masking

LLaMA-Factory's `mask_history` masks whole turns. The answer-only objective needs a mask
inside a turn: everything in the response up to and including `</think>`.
[eklav/lf_plugin](eklav/lf_plugin) patches LLaMA-Factory's encoder to do that when
`EKLAV_ANSWER_ONLY_AFTER_TOKEN="</think>"` is set; `eklav train` sets it for every
method except STD-CoT and puts the plugin's startup directory on `PYTHONPATH`. The
marker is located in decoded text, not by token ids, because BPE merges depend on
context.

## Adding a task

A task is one module in [eklav/tasks/](eklav/tasks/) defining a `TASK`: its raw-corpus
reader, hint `h(r)`, prompt wording and hyperparameters (see
[table_reranking.py](eklav/tasks/table_reranking.py)). Register it in
`eklav/tasks/__init__.py` and every method, `eklav train` and `eklav data` work for it.

## Repository layout

```
eklav/
  methods.py        the four methods and their loss masks
  data.py           sequence layouts and training-data building
  tasks/            passage and table reranking
  train.py          eklav train
  eval/             eklav eval: benchmarks, retrieval, the two evaluators
  lf_plugin/        answer-only loss mask for LLaMA-Factory
  chunks.py         splitting/reassembling large data files
data/               built training data
results/            evaluation outputs
tests/              pytest suite (no GPU needed)
```

## Results

Full results, run traces and checkpoints will be released on acceptance.
