# Physics-Informed Neural Operators for Inverse Problems with PINO

This repository contains implementations of Physics-Informed Neural Operators (PINO) for solving inverse problems, specifically focused on Darcy flow equations. The code combines machine learning with physics-based constraints to enable efficient learning of solution operators.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Training Models](#training-models)
- [Configuration Files](#configuration-files)
- [Results](#results)
- [Testing](#testing)
- [Citation](#citation)

## Overview

This project implements several variants of neural operators for inverse PDE problems:

- **PINO (Physics-Informed Neural Operator)**: Combines data-driven learning with physics constraints through a loss function that enforces PDE residuals
- **FNO (Fourier Neural Operator)**: Baseline data-driven neural operator without explicit physics constraints
- **Sparse PINO**: Variant trained with sparse sensor observations
- **Physics-Only Models**: Explores the effectiveness of physics constraints alone without data

The models are trained on Darcy flow problems at 241×241 and 421×421 resolutions, with support for noise and various physics weighting schemes.

## Requirements

The project requires:
- Python 3.8+
- PyTorch
- Hydra for configuration management
- PhysicsNEMO (NVIDIA's physics-informed ML framework)
- h5py for data handling
- matplotlib for visualization
- numpy and scipy

## Project Structure

```
.
├── inverse_darcy_fno.py              # Main training script for PINO models
├── inverse_darcy_sparse.py           # Training script for sparse sensor scenarios
├── physics_only.py                   # Baseline physics-only model training
├── diffusion_eq.py                   # Darcy equation (diffusion) PDE definition
├── utils.py                          # Dataset handling and utility functions
├── conf/                             # Hydra configuration files
│   ├── pino.yaml                     # PINO configuration (baseline)
│   ├── pino_stronger_physics.yaml    # PINO with higher physics weight
│   ├── fno_no_physics.yaml           # FNO baseline (no physics)
│   ├── physics_only.yaml             # Physics-only baseline
│   ├── physics_only_noisy.yaml       # Physics-only with noise
│   ├── noisy_pino.yaml               # PINO trained with noise
│   ├── pde_bench/                    # PDEBench dataset configurations
│   └── sparse/                       # Sparse sensor configurations
├── train_darcy.sh                    # SLURM script for training PINO
├── train_sparse.sh                   # SLURM script for sparse PINO
├── train_physics_only.sh             # SLURM script for physics-only model
├── train_pdebench_darcy.sh           # SLURM script for PDEBench dataset
├── test_inverse_darcy.py             # Testing and validation script
├── test_sparse_darcy.py              # Sparse model testing
├── test_with_noise.sh                # Noise robustness testing
└── datasets/                         # Data directory (created during setup)
```

## Datasets

### Darcy 241×241
- **Location**: `./datasets/Darcy_241/`
- **Files**:
  - `piececonst_r241_N1024_smooth1.hdf5` - Training set (1024 samples)
  - `piececonst_r241_N1024_validation.hdf5` - Validation set
  - `piececonst_r241_N1024_test.hdf5` - Test set
- **Resolution**: 241×241 grid points
- **Input**: Permeability coefficient field
- **Output**: Pressure solution from Darcy's law

The datasets are automatically downloaded on first run if not present. Data is stored in HDF5 format with the following keys:
- `coeff` - Input permeability coefficients
- `sol` - Output pressure solutions

### Data Format

Data is loaded as:
- **Input shape**: (N, 1, 241, 241) - permeability field
- **Output shape**: (N, 1, 241, 241) - pressure solution
- Where N is the number of samples

## Training Models

### Quick Start: Local Training

To train a PINO model locally with default configuration:

```bash
python inverse_darcy_fno.py --config-name pino seed=0
```

### Training with Custom Configuration

Specify the configuration and override parameters:

```bash
python inverse_darcy_fno.py \
  --config-name pino_stronger_physics \
  seed=0 \
  max_epochs=250 \
  batch_size=4 \
  start_lr=0.001
```

### Training Multiple Seeds

Train the same configuration with different random seeds for ensemble results:

```bash
python inverse_darcy_fno.py --config-name pino seed=0
python inverse_darcy_fno.py --config-name pino seed=1
python inverse_darcy_fno.py --config-name pino seed=2
```

### Sparse Sensor Training

Train models with incomplete observations (sparse sensors):

```bash
python inverse_darcy_sparse.py \
  --config-name sparse_pino_weight_1.0 \
  seed=0
```

### Physics-Only Training

Train with physics constraints but no data:

```bash
python physics_only.py \
  --config-name physics_only_noisy \
  seed=0
```

### Using HPC (SLURM)

For GPU-accelerated training on HPC clusters with SLURM:

```bash
sbatch train_darcy.sh              # Train PINO with multiple seeds
sbatch train_sparse.sh             # Train sparse PINO
sbatch train_physics_only.sh       # Train physics-only baseline
sbatch train_pdebench_darcy.sh     # Train on PDEBench dataset
```

**Note**: SLURM scripts are pre-configured for NVIDIA A100 GPUs with Apptainer containerization. Modify paths and partition names for your HPC environment.

## Configuration Files

### Core Parameters

All configurations use YAML format and support parameter overrides via command line.

#### `pino.yaml` - Baseline PINO Configuration

```yaml
max_epochs: 250                  # Total training epochs
batch_size: 4                    # Batch size for training
validation_batch_size: 1         # Batch size for validation
start_lr: 0.001                  # Initial learning rate
step_size: 100                   # Learning rate scheduler step size
gamma: 0.5                       # Learning rate decay factor
seed: 0                          # Random seed

physics_weight: 0.2              # Weight of physics loss (0-1)
tv_weight: 0.01                  # Total variation regularization weight
max_noise: 0.0                   # Noise level in training data

model:
  fno:
    in_channels: 1               # Input channels (permeability)
    out_channels: 1              # Output channels (pressure)
    latent_channels: 32          # Hidden channel dimension
    num_fno_layers: 4            # Number of Fourier layers
    num_fno_modes: 12            # Number of Fourier modes per layer
    padding: 9                   # Padding for FFT
    dimension: 2                 # Problem dimension
    decoder_layers: 1            # Post-processing layers
    decoder_layer_size: 128      # Size of decoder layers

data:
  train_path: ./datasets/Darcy_241/piececonst_r241_N1024_smooth1.hdf5
  validation_path: ./datasets/Darcy_241/piececonst_r241_N1024_validation.hdf5
  test_path: ./datasets/Darcy_241/piececonst_r241_N1024_test.hdf5
  resolution: 241

scaling:
  permeability: 4.49996e00       # Input normalization factor
  darcy: 3.88433e-03             # Output normalization factor
```

#### `pino_stronger_physics.yaml` - High Physics Weight

Similar to `pino.yaml` but with `physics_weight: 1.0` for stronger physics constraints.

#### `sparse/sparse_pino.yaml` - Sparse Observations

Extends PINO for incomplete sensor observations with:
```yaml
in_channels: 2                   # Permeability + sensor mask
sensor_densities: [1.0, 0.5, 0.25, 0.10, 0.05, 0.01, 0.005, 0.001]
```

### Creating Custom Configurations

1. Copy an existing config: `cp conf/pino.yaml conf/my_config.yaml`
2. Edit parameters as needed
3. Train with: `python inverse_darcy_fno.py --config-name my_config`

### Parameter Tuning Guide

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `physics_weight` | Influence of physics constraint | 0.0-1.0 |
| `tv_weight` | Smoothness regularization | 0.0-0.1 |
| `batch_size` | Gradient accumulation scale | 1-8 |
| `start_lr` | Initial optimization rate | 1e-4 to 1e-2 |
| `step_size` | Epochs between LR decay | 50-200 |
| `num_fno_modes` | Model capacity (modes²) | 8-16 |
| `latent_channels` | Hidden dimension | 16-64 |

## Training Output

Training outputs are organized in timestamped directories with the structure:

```
inverse_darcy_training/pino_seed_0/
├── .hydra/
│   ├── config.yaml              # Full resolved configuration
│   └── hydra.yaml               # Hydra metadata
├── checkpoints/
│   ├── checkpoint_epoch_10.pt   # Model checkpoints (saved every checkpoint_freq epochs)
│   └── checkpoint_latest.pt     # Latest checkpoint
├── training_log.csv             # Training metrics per epoch
└── outputs.log                  # Console output
```

### Monitoring Training

Training saves:
- **Checkpoints**: Regular model snapshots (default: every 10 epochs)
- **Metrics**: CSV log of loss, validation error, and physics residuals
- **Configuration**: Full YAML config used for reproducibility

## Testing

### Evaluate on Test Set

```bash
python test_inverse_darcy.py ./inverse_darcy_training/pino_seed_0
```

### Test Multiple Seeds

```bash
for seed in 0 1 2; do
  python test_inverse_darcy.py ./inverse_darcy_training/pino_seed_${seed}
done
```

### Sparse Model Testing

```bash
python test_sparse_darcy.py ./neural_operator_outputs/sparse/pino_0
```

### Noise Robustness Testing

```bash
bash test_with_noise.sh
```

## Results Reproducibility

To reproduce published results:

### PINO Baseline (Physics Weight 0.2)

```bash
for seed in 0 1 2; do
  python inverse_darcy_fno.py \
    --config-name pino \
    seed=${seed} \
    max_epochs=250 \
    hydra.run.dir="seeded_runs_darcy/pino_seed_${seed}"
done
```

### Sparse Sensors

```bash
python inverse_darcy_sparse.py \
  --config-name sparse_pino_weight_1.0 \
  seed=0 \
  hydra.run.dir="seeded_runs_darcy/sparse_pino_seed_0"
```

### Physics-Only Baselines

```bash
python physics_only.py \
  --config-name physics_only_noisy \
  seed=0 \
  hydra.run.dir="seeded_runs_darcy/physics_only_seed_0"
```

## Advanced Usage

### Modifying Physics Constraints

Edit `diffusion_eq.py` to change the PDE definition or physics-informed loss computation.

### Changing Network Architecture

Modify `model.fno` parameters in configuration files to experiment with:
- Number of Fourier modes (`num_fno_modes`)
- Hidden dimensions (`latent_channels`)
- Depth (`num_fno_layers`)

### Custom Loss Functions

Modify the loss computation in the training scripts to add custom regularization or constraints.

### Batch Processing Results

Use `collect_summaries.py` to aggregate results from multiple training runs and `summary_analysis.ipynb` for comparative analysis.

## Model Architecture Details

### FNO (Fourier Neural Operator)

The model uses Fourier spectral methods for efficient operator learning:

1. **Lifting**: Project input to higher dimension with MLP
2. **FNO Layers**: Apply spectral convolutions in Fourier space
3. **Decoder**: Project back to output dimension with MLP

### Physics-Informed Loss

The total loss combines:

$$\mathcal{L} = \mathcal{L}_{data} + \lambda_{physics} \cdot \mathcal{L}_{physics} + \lambda_{tv} \cdot \mathcal{L}_{tv}$$

Where:
- $\mathcal{L}_{data}$: Relative L2 error on training data
- $\mathcal{L}_{physics}$: PDE residual (Darcy equation)
- $\mathcal{L}_{tv}$: Total variation regularization

## License

This project is licensed under the Apache License 2.0. See LICENSE file for details.

## References

- Original PINO paper: [Physics-Informed Neural Operators: Deep Learning for Parametric PDEs](https://arxiv.org/abs/2111.03794)
- FNO: [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- PhysicsNEMO: NVIDIA's framework for physics-informed machine learning
