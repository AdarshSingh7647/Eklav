#!/bin/bash
#SBATCH --job-name=cotgen-rerun
#SBATCH --partition=htc
#SBATCH --qos=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --output=./slurm/logs/%x-%j.out
#SBATCH --error=./slurm/logs/%x-%j.err

set -uo pipefail

ENV="${EKLAV_ENV:?set EKLAV_ENV to your Python env prefix (e.g. a conda env path)}"
SCRIPT="${SCRIPT:?set SCRIPT to the path of the rerun script to run}"

export HF_HOME="${HF_HOME:?set HF_HOME to your Hugging Face cache directory}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$HF_HOME/tmp/triton"
export PATH="$ENV/bin:$PATH"

echo "=== $(date) | job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$ENV/bin/python" "$SCRIPT"
RC=$?
echo "=== $(date) | job=$SLURM_JOB_ID finished with exit code $RC ==="
exit $RC
