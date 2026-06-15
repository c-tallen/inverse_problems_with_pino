import argparse
from datetime import datetime
import sys

import hydra
import numpy as np
import torch
import physicsnemo
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import cast
from matplotlib.axes import Axes
from utils import CustomDataset
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from diffusion_eq import Diffusion
import pathlib
from omegaconf import DictConfig, OmegaConf
from utils import darcy_mask1, corr_indicator

def run_test(cfg: DictConfig,
             model_path: pathlib.Path,
             output_dir: pathlib.Path,
             test_dataset_path = None, 
             test_mappings=None,
             res=241,
             perm_min=0.0,
             perm_max=0.0
    ):
    
    torch.manual_seed(0)
    np.random.seed(0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Loading model and dataset on device:", device)
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    if test_mappings is not None:
        mappings_dict = test_mappings
    else:
        mappings_dict = OmegaConf.to_container(cfg.mappings, resolve=True)
    model = physicsnemo.Module.from_checkpoint(str(model_path)).to(device)
    model.eval()
    
    tp = cfg.data.test_path if hasattr(cfg.data, 'test_path') else cfg.data.validation_path
    dataset_path = (
        pathlib.Path(test_dataset_path).resolve()
        if test_dataset_path is not None
        else pathlib.Path(tp).resolve()
    )

    print(f"Using test dataset: {dataset_path}")
    resolution = res if res is not None else cfg.data.resolution
    print(f"Using resolution: {resolution}x{resolution}")
    dataset = CustomDataset(dataset_path, device=device, mappings=mappings_dict, res=resolution)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    darcy_scale = cfg.scaling.darcy
    if perm_max == 0.0 and perm_min == 0.0:
        permeability_min = cfg.data.permeability_min if cfg.data.pde_bench else 3.0
        permeability_max = cfg.data.permeability_max if cfg.data.pde_bench else 12.0
    else:
        permeability_min = perm_min
        permeability_max = perm_max
        
    # Use Diffusion equation for the Darcy PDE
    forcing_fn = cfg.physics_forcing_term
    fd_dx = 1.0 / float(resolution - 1)
    darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)
    phy_informer = PhysicsInformer(
        required_outputs=["diffusion_u"],
        equations=darcy,
        grad_method="finite_difference",
        device=str(device),
        fd_dx=fd_dx,
    )
    
    print("Starting inference on validation dataset...")
    with torch.inference_mode():
        loss_mse = [0.0 for _ in noise_levels]
        loss_rel_l2 = [0.0 for _ in noise_levels]
        physics_loss = [0.0 for _ in noise_levels]
        corr_loss = [0.0 for _ in noise_levels]
        sample_count = 0

        for sample_i, data in enumerate(dataloader):
            k = data["permeability"]
            u = data["darcy"]
            # assert u.shape == (1, 1, resolution, resolution), f"Unexpected input shape: {u.shape}"
            # assert k.shape == (1, 1, resolution, resolution), f"Unexpected target shape: {k.shape}"

            # Create one noisy version of u for each fixed noise level.
            noisy_inputs = []
            u_std = u.std(dim=(-2, -1), keepdim=True)

            for alpha in noise_levels:
                noise = torch.randn_like(u) * u_std * alpha
                u_noisy = u + noise
                noisy_inputs.append(u_noisy)
            # Add noise to unscaled input u
            u_noisy_batch = torch.cat(noisy_inputs, dim=0)

            # Match current training config: no scaling 
            u_input = u_noisy_batch / darcy_scale

            # Model outputs raw values; convert to permeability range [3, 12]
            out_raw = model(u_input)
            k_pred_batch = darcy_mask1(out_raw, permeability_min=permeability_min, permeability_max=permeability_max)

            print(f"Processing sample {sample_i}, output shape: {out_raw.shape}")

            expected_unscaled = k.detach().cpu().numpy()
            pred_unscaled_batch = k_pred_batch.detach().cpu().numpy()

            measures_per_sample = {
                "mse": [],
                "rel_l2": [],
                "physics_loss": [],
                "corr_indicator": [],
            }

            for i in range(len(noise_levels)):
                pred_i = k_pred_batch[i:i+1]

                mse = F.mse_loss(pred_i, k).item()
                corr = corr_indicator(pred_i, k).item()
                corr_loss[i] += corr

                rel_l2 = (
                    torch.linalg.norm(pred_i - k) / torch.linalg.norm(k)
                ).item()

                measures_per_sample["mse"].append(mse)
                loss_mse[i] += mse
                measures_per_sample["rel_l2"].append(rel_l2)
                loss_rel_l2[i] += rel_l2
                measures_per_sample["corr_indicator"].append(corr)

                assert pred_i.shape == k.shape, (
                    f"Output shape {pred_i.shape} does not match target shape {k.shape}"
                )

                residuals = phy_informer.forward(
                    {
                        "u": u,
                        "k": pred_i,
                    }
                )

                pde_out_arr = residuals["diffusion_u"]
                pde_core = pde_out_arr[:, :, 2:-2, 2:-2]

                physics_loss[i] += torch.mean(torch.abs(pde_core)).item()

            if sample_i < 5:
                plot_recovered(
                    noise_levels,
                    sample_i,
                    u,
                    u_noisy_batch,
                    expected_unscaled,
                    pred_unscaled_batch,
                    measures_per_sample,
                    output_dir,
                )

            sample_count += 1

        if sample_count > 0:
            # Calculate the average mse loss for each noise level
            loss_mse = [loss / sample_count for loss in loss_mse]
            loss_rel_l2 = [loss / sample_count for loss in loss_rel_l2]
            physics_loss = [loss / sample_count for loss in physics_loss]
            corr_loss = [loss / sample_count for loss in corr_loss]
        print("Total MSE for each noise level:")
        for i, error in enumerate(loss_mse):
            print(f"Noise level {noise_levels[i]}: {error}")
        print("Total relative L2 for each noise level:")
        for i, error in enumerate(loss_rel_l2):
            print(f"Noise level {noise_levels[i]}: {error}")
        print("Total physics loss for each noise level:")
        for i, error in enumerate(physics_loss):
            print(f"Noise level {noise_levels[i]}: {error}")
        print("Total correlation indicator for each noise level:")
        for i, corr in enumerate(corr_loss):
            print(f"Noise level {noise_levels[i]}: {corr}")
            
        with open(output_dir / "results_summary.txt", "w") as f:
            f.write("Noise Level\tMSE Loss\tRelative L2 Loss\tPhysics Loss\tCorrelation Indicator\n")
            for i in range(len(noise_levels)):
                f.write(f"{noise_levels[i]}\t{loss_mse[i]:.6e}\t{loss_rel_l2[i]:.6e}\t{physics_loss[i]:.6e}\t{corr_loss[i]:.6e}\n")

def plot_recovered(noise_levels, sample_i, u, u_noisy, expected_unscaled, predvar_unscaled, measures_per_sample, output_dir):
    rows = 1 + len(noise_levels)
    cols = 3
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 3.2), squeeze=False, constrained_layout=True)
    fig.suptitle(f"Inverse Darcy results for sample {sample_i}", fontsize=14)

    def plot_with_colorbar(col, row, data, title, *, cmap="viridis", vmin=None, vmax=None):
        image = np.squeeze(data)
        axis = cast(Axes, ax[row, col])
        im = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    plot_with_colorbar(0, 0, u[0, 0].cpu().numpy(), "Clean Input", cmap="viridis")
    plot_with_colorbar(1, 0, expected_unscaled[0, 0], "True", cmap="magma")
    ax[0, 2].axis("off")
    ax[0, 2].text(0.5, 0.5, "True input/output pair", ha="center", va="center", fontsize=10)

    diff_vmax = max(
        np.max(np.abs(predvar_unscaled[i, 0] - expected_unscaled[0, 0]))
        for i in range(len(noise_levels))
    )
    for i in range(len(noise_levels)):
        row = i + 1
        plot_with_colorbar(0, row, u_noisy[i, 0].cpu().numpy(), f"Noisy Input\nNoise: {noise_levels[i]}", cmap="viridis")
        plot_with_colorbar(1, row, predvar_unscaled[i, 0], f"Prediction\nNoise: {noise_levels[i]}", cmap="magma", vmin=3.0, vmax=12.0)
        diff = np.abs(predvar_unscaled[i, 0] - expected_unscaled[0, 0])
        plot_with_colorbar(2, row, diff, f"Abs Diff\nMSE: {measures_per_sample['mse'][i]:.3e}\nCorr: {measures_per_sample['corr_indicator'][i]:.3e}", cmap="inferno", vmin=0.0, vmax=diff_vmax)

    fig.savefig(output_dir / f"results_{sample_i}.png", dpi=200, bbox_inches="tight")

    plt.close(fig)

def extract_epoch_from_checkpoint(path: pathlib.Path) -> int:
    """
    Extract epoch number from filenames like:
        FNO.0.99.mdlus
        model.0.123.mdlus

    Returns 99, 123, etc.
    """
    parts = path.name.split(".")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)

    raise ValueError(f"Could not extract epoch number from checkpoint: {path.name}")


def find_newest_checkpoint(checkpoints_dir: pathlib.Path) -> pathlib.Path:
    checkpoint_paths = list(checkpoints_dir.glob("*.mdlus"))

    if not checkpoint_paths:
        raise FileNotFoundError(f"No .mdlus checkpoint files found in {checkpoints_dir}")

    return max(checkpoint_paths, key=extract_epoch_from_checkpoint)


def load_model_dir(model_dir):
    model_dir = pathlib.Path(model_dir).resolve()
    checkpoints_dir = model_dir / "checkpoints"
    config_path = model_dir / ".hydra" / "config.yaml"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory does not exist: {checkpoints_dir}")

    if not config_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {config_path}")

    model_path = find_newest_checkpoint(checkpoints_dir)
    cfg = OmegaConf.load(config_path)

    return cfg, model_path, model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir", type=pathlib.Path)

    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=None,
        help="Optional custom dataset path for testing.",
    )

    parser.add_argument(
        "--permeability_mapping",
        type=str,
        default=None,
        help="Optional custom mappings file for the test dataset.",
    )
    parser.add_argument(
        "--darcy_mapping",
        type=str,
        default=None,
        help="Optional custom mappings file for the test dataset.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Resolution of the test dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Directory to save test results.",
    )
    parser.add_argument(
        "--perm_min",
        type=float,
        default=0.0,
        help="Minimum permeability value for scaling the model output. If 0.0, uses value from config.",
    )
    parser.add_argument(
        "--perm_max",
        type=float,
        default=0.0,
        help="Maximum permeability value for scaling the model output. If 0.0, uses value from config.",
    )

    args = parser.parse_args()
    if args.permeability_mapping is not None and args.darcy_mapping is not None:
        test_mappings = {
            "permeability": args.permeability_mapping,
            "darcy": args.darcy_mapping,
        }
        print(f"Custom test mappings: {test_mappings}")
    else:
        test_mappings = None

    cfg, model_path, model_dir = load_model_dir(args.model_dir)

    model_name = model_path.stem + "_" + model_dir.name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir is not None:
        output_dir = args.output_dir / model_name / timestamp
    else:
        output_dir = (
            pathlib.Path("./test_results_original_neural_operator_method")
            / model_name
            / timestamp
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded config from: {model_dir / '.hydra' / 'config.yaml'}")
    print(f"Using newest checkpoint: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Custom test mappings: {test_mappings}")

    run_test(
        cfg,
        model_path,
        output_dir,
        test_dataset_path=args.dataset,
        test_mappings=test_mappings,
        res=args.resolution,
        perm_min=args.perm_min,
        perm_max=args.perm_max,
    )
    
    