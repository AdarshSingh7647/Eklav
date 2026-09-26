#!/bin/bash
#SBATCH --job-name=eklav-pr
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

set -uo pipefail

# --- knobs (override at submit: METHOD=... BASE_MODEL=... PDB=... sbatch ...) ---
METHOD="${METHOD:-answer-only}"              # std-sft | answer-only | eklav | eklav-mask-only (ablation)
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
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

echo "=== $(date) | job=$SLURM_JOB_ID method=$METHOD model=$BASE_MODEL gpus=$NGPU pdb=$PDB out=$OUT ==="
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
