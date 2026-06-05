#!/bin/bash
#SBATCH --job-name=darcy_seed_train
#SBATCH --partition=gpu-a100
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5333MB
#SBATCH --array=0-11
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# Avoid broken/stale distributed rendezvous variables
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Go to your working directory
cd /scratch/cwilczewski/physicsnemo/temp

set -euo pipefail

mkdir -p logs

SIF="/scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif"
PYTHON="apptainer exec --nv $SIF python"

SEEDS=( 0 1 2) # TODO: Use more seeds

CONFIGS=(
  "neural_operator_no_physics"
  "neural_operator_noisy_pino"
  "neural_operator_noisy"
  "neural_operator_no_scaling"
)

OUTPUT_NAMES=(
  "fno_no_physics"
  "noisy_pino"
  "noisy_fno"
  "fno_no_scaling"
)

NUM_CONFIGS=${#CONFIGS[@]}

SEED_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_CONFIGS))
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_CONFIGS))

SEED=${SEEDS[$SEED_INDEX]}
CONFIG_NAME=${CONFIGS[$CONFIG_INDEX]}
OUTPUT_NAME=${OUTPUT_NAMES[$CONFIG_INDEX]}

OUTPUT_DIR="./seeded_runs_darcy/${OUTPUT_NAME}_seed_${SEED}"

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Seed index: ${SEED_INDEX}"
echo "Config index: ${CONFIG_INDEX}"
echo "Seed: ${SEED}"
echo "Config: ${CONFIG_NAME}"
echo "Output dir: ${OUTPUT_DIR}"

# Optional: uncomment if you want fresh training every time
rm -rf "$OUTPUT_DIR"

$PYTHON inverse_darcy_fno.py \
  --config-name "$CONFIG_NAME" \
  seed="$SEED" \
  max_epochs=250 \
  hydra.run.dir="$OUTPUT_DIR"