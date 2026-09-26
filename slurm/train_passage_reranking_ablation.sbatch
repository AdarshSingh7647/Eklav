#!/bin/bash
#SBATCH --job-name=eklav-pr-ablation
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH --output=./slurm/logs/%x-%j.out
#SBATCH --error=./slurm/logs/%x-%j.err
# --time is set at submission via `sbatch --time=HH:MM:SS` (see submit commands) --

# Masking-only ablation for passage reranking: isolates loss masking as the
# single factor changed relative to std-sft, holding context layout fixed.
# Eklav changes three things vs. std-sft at once -- (a) moves the teacher
# trace into the prompt as a filtered hint, (b) masks its loss, (c) lexically
# filters verdict-bearing sentences out of that hint. This run changes ONLY
# (b): it trains on std-sft's data verbatim (full trace in the assistant
# turn, no prompt hint, no filtering) with the <think>...</think> span masked
# from the loss (EKLAV_THINK_CONTENT_MASK=1) instead of supervised token-by-
# token. Comparing this run's BRIGHT scores against std-sft and eklav tells
# you how much of Eklav's gain over std-sft comes from masking alone versus
# from the prompt-placement + filtering changes.
#
# See eklav/data/passage_reranking.py and
# eklav/configs/passage_reranking.py::METHOD_ENV["eklav-mask-only"] for the
# implementation; eklav/train.py enforces this method is passage_reranking-only.

set -uo pipefail

# --- knobs (override at submit: BASE_MODEL=... PDB=... sbatch ...) ---
METHOD="eklav-mask-only"                     # fixed: this job is the ablation
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"    # match whichever std-sft/eklav checkpoint you're comparing against
PDB="${PDB:-8}"                              # per-device micro-batch (memory knob)
NGPU="${NGPU:-2}"                            # must match --gres above
EKLAV_REPO="${EKLAV_REPO:-$(pwd)}"
ENV="${EKLAV_ENV:?set EKLAV_ENV to your Python env prefix (e.g. a conda env path)}"
LF_DATA="${LF_DATA:?set LF_DATA to your LLaMA-Factory data/ directory}"
TAG="$(basename "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')"
OUT="${OUT:-$EKLAV_REPO/runs/pr_${METHOD}_${TAG}}"
# -------------------------------------------------------------------------------

mkdir -p "$EKLAV_REPO/slurm/logs" "$OUT"

export HF_HOME="${HF_HOME:?set HF_HOME to your Hugging Face cache directory}"
export HF_HUB_OFFLINE=1                      # models+data already cached locally
export PYTHONPATH="$EKLAV_REPO/llamafactory_plugin:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$HF_HOME/tmp/triton"
export OMP_NUM_THREADS=5
export PATH="$ENV/bin:$PATH"
export FORCE_TORCHRUN=1
export NNODES=1 NODE_RANK=0 NPROC_PER_NODE="$NGPU"

echo "=== $(date) | job=$SLURM_JOB_ID method=$METHOD (ablation) model=$BASE_MODEL gpus=$NGPU pdb=$PDB out=$OUT ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# --- generate/refresh the LLaMA-Factory YAML config (idempotent: safe to regenerate) ---
cd "$EKLAV_REPO"
"$ENV/bin/python" -m eklav.train \
  --task passage_reranking \
  --method "$METHOD" \
  --base_model "$BASE_MODEL" \
  --output_dir "$OUT" \
  --llamafactory_data_dir "$LF_DATA" \
  --num_processes "$NGPU" \
  --per_device_train_batch_size "$PDB" \
  --dry_run

CONFIG="$OUT/train_config.yaml"

# --- auto-resume: LLaMA-Factory itself detects the latest checkpoint under $OUT ---
# (overwrite_output_dir: false in the generated YAML + do_train + an existing
#  checkpoint-* dir under output_dir => HF Trainer resumes automatically, see
#  llamafactory/hparams/parser.py's get_last_checkpoint() call). Nothing extra
# needed here other than pointing at the SAME output_dir every time.
LAST_CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$LAST_CKPT" ]; then
  echo "[eklav] Found existing checkpoint: $LAST_CKPT -- LLaMA-Factory will auto-resume from it."
else
  echo "[eklav] No existing checkpoint under $OUT -- starting fresh."
fi

echo "[eklav] Launching: llamafactory-cli train $CONFIG"
"$ENV/bin/llamafactory-cli" train "$CONFIG"
RC=$?
echo "=== $(date) | job=$SLURM_JOB_ID finished with exit code $RC ==="
exit $RC
