#!/bin/bash
#SBATCH --job-name=darcy_train
#SBATCH --partition=gpu-a100-small
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5333MB
#SBATCH --time=04:00:00
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

# Go to your working directory
cd /scratch/cwilczewski/physicsnemo/temp

# Run inside Apptainer --config-name config_fno_a100
apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python inverse_darcy_fno.py \
 --config-name pino_stronger_physics seed=2 \
 max_epochs=250 \
 hydra.run.dir="seeded_runs_darcy/pino_stronger_physics_seed_2"

apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python inverse_darcy_fno.py \
 --config-name pino_stronger_physics seed=1 \
 max_epochs=250 \
 hydra.run.dir="seeded_runs_darcy/pino_stronger_physics_seed_1"

 apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python inverse_darcy_fno.py \
 --config-name pino_stronger_physics seed=0 \
 max_epochs=250 \
 hydra.run.dir="seeded_runs_darcy/pino_stronger_physics_seed_0"