#!/bin/bash
#SBATCH --job-name=darcy_seed_test
#SBATCH --partition=gpu-a100-small
#SBATCH --partition=gpu-a100-small
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000MB
#SBATCH --array=0-11
#SBATCH --time=04:00:00
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err

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

SEEDS=( 1 )

OUTPUT_NAMES=(
  "fno_no_physics"
  "noisy_pino"
  "noisy_fno"
  "fno_no_scaling"
)

NUM_CONFIGS=${#OUTPUT_NAMES[@]}

SEED_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_CONFIGS))
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_CONFIGS))

SEED=${SEEDS[$SEED_INDEX]}
OUTPUT_NAME=${OUTPUT_NAMES[$CONFIG_INDEX]}

MODEL_DIR="./seeded_runs_darcy/${OUTPUT_NAME}_seed_${SEED}"

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Seed: ${SEED}"
echo "Model dir: ${MODEL_DIR}"

$PYTHON test_inverse_darcy.py "$MODEL_DIR"