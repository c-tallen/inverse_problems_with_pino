#!/bin/bash
#SBATCH --job-name=physics_only_darcy_fno
#SBATCH --partition=gpu-a100-small
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5333MB
#SBATCH --time=04:00:00
#SBATCH --output=logs/physics_only_%j.out
#SBATCH --error=logs/physics_only_%j.err
set -euo pipefail

mkdir -p logs

# Optional: print some debugging info
echo "Running on node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Avoid broken/stale distributed rendezvous variables
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Go to your working directory
cd /scratch/cwilczewski/physicsnemo/temp

# Run inside Apptainer --config-name config_fno_a100
apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python physics_only.py --config-name physics_only_noisy seed=0
# apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python physics_only.py --config-name physics_only seed=2

# Run tests
# apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python test_inverse_darcy.py ./inverse_darcy_training/physics_only