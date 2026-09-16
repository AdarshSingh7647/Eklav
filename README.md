# Eklav

Eklav is the training/eval pipeline for an ICLR paper's method across three tasks
-- **passage reranking** (pointwise true/false), **table reranking** (listwise
JSON ranking), and **math reasoning** (worked solutions ending in `\boxed{...}`)
-- each trained via LoRA SFT (LLaMA-Factory) under three supervision methods:
**std-SFT**, **Answer-only SFT**, and **Eklav** (this paper's method). The repo
downloads/builds the right data, generates the right LLaMA-Factory config for
any base model you point it at, and runs faithful, full-metadata evals against
the paper's benchmarks.

## Quickstart

```bash
git clone <this-repo> && cd Eklav
pip install -r requirements.txt
pip install -e llamafactory_plugin/          # the masking plugin (see below)

# Train: pick a task, a method, and any base model (local path or HF repo id)
python -m eklav.train \
  --task passage_reranking --method eklav \
  --base_model Qwen/Qwen3-8B \
  --output_dir ./runs/passage_reranking_eklav_qwen3_8b \
  --llamafactory_data_dir /path/to/LLaMA-Factory/data

# Data alone, no training (optional -- train.py does this automatically)
python scripts/download_data.py --task math_reasoning --method std-sft

# Eval
python eklav/eval/passage_reranking.py \
  --model_path ./runs/passage_reranking_eklav_qwen3_8b/merged \
  --model_tag qwen3_8b_eklav --out_dir ./results/passage_reranking
```

Swap `--task` between `passage_reranking` / `table_reranking` / `math_reasoning`,
`--method` between `std-sft` / `answer-only` / `eklav`, and `--base_model` to any
local checkpoint path or HF repo id. The paper trains Qwen3-4B/8B/14B and
GLM-Z1-9B as examples, but nothing in the code is hardcoded to those.

## The three methods

- **std-SFT**: prompt -> full teacher reasoning trace + answer, with loss
  computed over the entire response (thinking included). This is the
  standard "distill the teacher's CoT" baseline.
- **Answer-only SFT**: no hint in the prompt, no reasoning anywhere in the
  loss -- the model's own thinking (if a multi-turn cue format is used) is
  masked out entirely, and loss is computed only on the final answer tokens.
- **Eklav** (this paper's method): a partial slice of the teacher's
  reasoning trace is placed in the prompt as a hint, the model reasons
  *freely* in its own `<think></think>` tags, and no loss is computed until
  the think tag closes -- loss falls only on the answer tokens. Per task:
  - *Passage reranking*: hint = first ~50% of the teacher trace (by sentence
    count, verdict-revealing sentences filtered out). Response is
    `<think>{free reasoning}</think>\n\n{true|false}`; loss is masked on
    everything through and including `</think>`.
  - *Table reranking*: hint = the teacher's **full**, untruncated trace
    (truncating a multi-candidate comparison would remove candidates from
    consideration, not just stop an argument early). Response is the bare
    answer JSON only, no `<think>` block -- loss falls naturally on the
    answer.
  - *Math reasoning*: hint = teacher trace minus its last 2 lines (which
    tend to restate the answer) plus any remaining leak phrases filtered.
    Response is `<think>\n{continuation}\n</think>\n\n{boxed answer}`; loss
    is masked on the `<think>` **content only** -- the tags and the final
    boxed answer stay supervised (this is the one place Eklav supervises the
    tag tokens; passage reranking masks through `</think>` inclusive).

## The LLaMA-Factory masking plugin

LLaMA-Factory's stock `mask_history` only masks at whole-turn granularity.
Eklav needs sub-turn masking (mask up to, or around, a text marker *within* a
single turn's response). `llamafactory_plugin/` packages this as a small,
installable patch that works against a **stock** `pip install llamafactory`
-- no fork needed.

**Install** (once, alongside your existing LLaMA-Factory environment):

```bash
pip install -e llamafactory_plugin/
```

**Enable**, either way:

```python
# Option A: import at the top of your training entrypoint
import eklav_llamafactory_plugin  # noqa: F401 -- applies the patch on import
```

```bash
# Option B: drop it on PYTHONPATH so it auto-loads at interpreter startup
# (works for `llamafactory-cli train ...` subprocess launches too)
export PYTHONPATH="/path/to/Eklav/llamafactory_plugin:$PYTHONPATH"
```

Then set the relevant env var before launching `llamafactory-cli train`:

| Env var | Effect | Used by |
|---|---|---|
| `EKLAV_ANSWER_ONLY_AFTER_TOKEN="</think>"` | Masks loss on everything up to and including the *last* occurrence of the marker in the decoded response; only what follows is supervised. | Passage reranking's Eklav method |
| `EKLAV_THINK_CONTENT_MASK=1` | Masks loss only on the text strictly between `<think>` and `</think>`; the tags and everything after stay supervised. | Math reasoning's Eklav method |

(Legacy names `FORGE_ANSWER_ONLY_AFTER_TOKEN` / `FORGE_THINK_CONTENT_MASK` are
also honored, for continuity with the original reference implementation.)

Both patches match on **decoded text**, not raw token ids -- BPE merges are
context-dependent, so a marker's token-id sequence when tokenized standalone
can differ from how the same characters tokenize once embedded in
surrounding text (confirmed empirically on GLM-Z1-9B's tokenizer). `eklav.train`
sets the right env var automatically for each task/method combination that
needs it -- table reranking's Eklav method needs neither (its response has no
`<think>` block at all, so loss falls naturally on the answer).

## Data

Four HF dataset repos already exist (`AdarshSingh7647/Eklav-Reranker-Data`,
`-CotGen-Data`, `Eklav-Math-Data`, `-CotGen-Data`) and are downloaded directly.
Answer-only (both tasks) and all of table reranking (all three methods) have
no HF repo yet, so `eklav/data/*.py` builds them locally from raw sources,
mirroring the reference construction exactly. Table reranking's raw training
source doesn't have a public link yet -- `eklav/data/table_reranking.py`'s
`TABLE_RERANKING_RAW_DATA_URL` is a placeholder for when the author publishes
one; until then, pass `--raw_path` explicitly, or rely on the local-machine
fallback path noted in that file (which will not exist on any other machine).

`eklav/eval/metrics.py` and `eklav/data/eval_downloads.py` also carry download
code for every eval benchmark: BRIGHT + `jhu-clsp/rank1-run-files` (passage
reranking), and the 6 IBM Research table-retrieval repos (table reranking;
5 OOD + NQ-Tables in-domain). Passage reranking eval is BRIGHT only, 12
domains -- NevIR is intentionally excluded. `StatCanDialogueRetrieval` and
`WatsonxDocsQARetrieval` are intentionally excluded from the table benchmarks.

## Eval

Each task's eval script (`eklav/eval/{passage_reranking,table_reranking,math_reasoning}.py`)
is a faithful port of the paper's reference eval harness (vLLM generation,
exact sampling settings, exact scoring/parsing logic -- see each file's
docstring for provenance). Every run writes one JSONL per (model, method,
benchmark), one row per example, with the full prompt, full raw generation,
parsed answer, gold answer, and id/benchmark/model metadata -- plus a
`*_summary.json` with the aggregate metric (nDCG@10 for reranking, pass@1 for
math).

## Just want the results, not the pipeline?

Results from this paper's runs have already been uploaded to
`dropbox_cot_reranker:CoT_ReRanker/iclr_paper_data/`. There's no public share
link yet for that path -- ask the author for one.
