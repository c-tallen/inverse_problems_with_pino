#!/bin/bash
#SBATCH --job-name=test_darcy_fno
#SBATCH --partition=gpu-a100-small
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000MB
#SBATCH --time=04:00:00
#SBATCH --output=tests/backup_no_physics_%j.out
#SBATCH --error=tests/backup_no_physics_%j.err

mkdir -p tests

# Optional: print some debugging info
echo "Running on node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Avoid broken/stale distributed rendezvous variables
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Go to your working directory
cd /scratch/cwilczewski/physicsnemo/temp

# Run inside Apptainer
# apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python test_inverse_darcy.py ./inverse_darcy_training/physics_only_noisy_seed_0 \
#     --output_dir "seeded_runs_tests_testset" \
#     --dataset "./datasets/Darcy_241/piececonst_r241_N1024_test.hdf5"

# apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python test_inverse_darcy.py ./inverse_darcy_training/physics_only_seed_0 \
#     --output_dir "seeded_runs_tests_testset" \
#     --dataset "./datasets/Darcy_241/piececonst_r241_N1024_test.hdf5"

# apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python test_inverse_darcy.py ./inverse_darcy_training/physics_only_noisy_seed_1 \
#     --output_dir "seeded_runs_tests_testset" \
#     --dataset "./datasets/Darcy_241/piececonst_r241_N1024_test.hdf5"

# apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python test_inverse_darcy.py ./inverse_darcy_training/physics_only_noisy_seed_2 \
#     --output_dir "seeded_runs_tests_testset" \
#     --dataset "./datasets/Darcy_241/piececonst_r241_N1024_test.hdf5"

# apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python test_inverse_darcy.py ./inverse_darcy_training/physics_only_seed_1 \
#     --output_dir "seeded_runs_tests_testset" \
#     --dataset "./datasets/Darcy_241/piececonst_r241_N1024_test.hdf5"

apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python test_inverse_darcy.py ./seeded_runs_darcy/pino_stronger_physics_seed_2 \
    --output_dir "seeded_runs_tests_testset" \
    --dataset "./datasets/Darcy_241/piececonst_r241_N1024_test.hdf5"

apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python test_inverse_darcy.py ./seeded_runs_darcy/pino_stronger_physics_seed_0 \
--output_dir "seeded_runs_tests_testset" \
--dataset "./datasets/Darcy_241/piececonst_r241_N1024_test.hdf5"

apptainer exec --nv /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif python test_inverse_darcy.py ./seeded_runs_darcy/pino_stronger_physics_seed_1 \
--output_dir "seeded_runs_tests_testset" \
--dataset "./datasets/Darcy_241/piececonst_r241_N1024_test.hdf5"