#!/bin/bash
#SBATCH --job-name=eklav-pr-eval
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --output=./slurm/logs/%x-%j.out
#SBATCH --error=./slurm/logs/%x-%j.err
# --time is set at submission via `sbatch --time=HH:MM:SS` (see submit commands) --

set -uo pipefail

# --- knobs (override at submit: MODEL_PATH=... LORA_PATH=... MODEL_TAG=... sbatch ...) ---
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH (base model, or merged model if LORA_PATH unset)}"
LORA_PATH="${LORA_PATH:-}"                    # empty = no LoRA, MODEL_PATH is a merged model
LORA_RANK="${LORA_RANK:-32}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG, e.g. qwen3-8b_answer-only_run1_step5968}"
OUT_DIR="${OUT_DIR:?set OUT_DIR, e.g. results/passage_reranking/qwen3-8b/answer-only/run1}"
NUM_GPUS="${NUM_GPUS:-1}"
CONTEXT_SIZE="${CONTEXT_SIZE:-20000}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
TOP_K="${TOP_K:-100}"
EKLAV_REPO="${EKLAV_REPO:-$(pwd)}"
ENV="${EKLAV_ENV:?set EKLAV_ENV to your Python env prefix (e.g. a conda env path)}"
# -------------------------------------------------------------------------------

mkdir -p "$EKLAV_REPO/slurm/logs" "$OUT_DIR"

export HF_HOME="${HF_HOME:?set HF_HOME to your Hugging Face cache directory}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$HF_HOME/tmp/triton"
export PATH="$ENV/bin:$PATH"

echo "=== $(date) | job=$SLURM_JOB_ID model_tag=$MODEL_TAG model_path=$MODEL_PATH lora_path=${LORA_PATH:-<none>} out=$OUT_DIR ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

cd "$EKLAV_REPO"

LORA_ARGS=()
if [ -n "$LORA_PATH" ]; then
  LORA_ARGS=(--lora_path "$LORA_PATH" --lora_rank "$LORA_RANK")
fi

"$ENV/bin/python" -m eklav.eval.passage_reranking \
  --model_path "$MODEL_PATH" \
  "${LORA_ARGS[@]}" \
  --model_tag "$MODEL_TAG" \
  --out_dir "$OUT_DIR" \
  --num_gpus "$NUM_GPUS" \
  --context_size "$CONTEXT_SIZE" \
  --max_output_tokens "$MAX_OUTPUT_TOKENS" \
  --top_k "$TOP_K"
RC=$?
echo "=== $(date) | job=$SLURM_JOB_ID finished with exit code $RC ==="
exit $RC
