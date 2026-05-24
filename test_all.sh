#!/bin/bash
#SBATCH --job-name=test_all_darcy_fno
#SBATCH --partition=gpu-a100-small
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000MB
#SBATCH --time=01:00:00
#SBATCH --output=tests/test_all_%j.out
#SBATCH --error=tests/test_all_%j.err

mkdir -p tests

echo "Running on node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Avoid broken/stale distributed rendezvous variables
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Go to working directory
cd /scratch/cwilczewski/physicsnemo/temp || exit 1

BASE_DIR="./inverse_fno_outputs/2026-05-20"

# Iterate over all folders in BASE_DIR
for folder in "$BASE_DIR"/*; do
    # Skip if not a directory
    [ -d "$folder" ] || continue

    MODEL_PATH="$folder/checkpoints/FNO.0.49.mdlus"

    # Skip if checkpoint does not exist
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Checkpoint not found: $MODEL_PATH"
        continue
    fi

    echo "========================================="
    echo "Testing model: $MODEL_PATH"
    echo "========================================="

    # Run the test script
    apptainer exec --nv \
        /scratch/cwilczewski/physicsnemo/physicsnemo_26.03.sif \
        python test_inverse_darcy.py "$MODEL_PATH"

    # Find newest created result directory for this model
    MODEL_NAME=$(basename "$folder")

    RESULT_BASE="./test_results_all_final/$MODEL_PATH"

    if [ -d "$RESULT_BASE" ]; then
        LATEST_RESULT=$(ls -td "$RESULT_BASE"/* 2>/dev/null | head -n 1)

        if [ -n "$LATEST_RESULT" ]; then
            echo "Copying .hydra to: $LATEST_RESULT"

            cp -r "$folder/.hydra" "$LATEST_RESULT/"
        else
            echo "No result directory found for $MODEL_PATH"
        fi
    else
        echo "Result base directory does not exist: $RESULT_BASE"
    fi

done

echo "All tests completed."