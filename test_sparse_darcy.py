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
from utils import darcy_mask1, corr_indicator, make_random_mask, make_sparse_input

def run_test(cfg: DictConfig, model_path: pathlib.Path, output_dir: pathlib.Path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Loading model and dataset on device:", device)

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    sensor_densities = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001]

    mappings_dict = OmegaConf.to_container(cfg.mappings, resolve=True)

    model = physicsnemo.Module.from_checkpoint(str(model_path)).to(device)
    model.eval()

    dataset = CustomDataset(
        cfg.data.validation_path,
        device=device,
        mappings=mappings_dict,
        res=cfg.data.resolution,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    darcy_scale = cfg.scaling.darcy
    resolution = cfg.data.resolution

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

    print("Starting sparse inference on validation dataset...")
    print(f"Noise levels: {noise_levels}")
    print(f"Sensor densities: {sensor_densities}")

    # Store metrics using keys (sensor_density, noise_level)
    results = {}

    for sensor_density in sensor_densities:
        for noise_level in noise_levels:
            results[(sensor_density, noise_level)] = {
                "mse": 0.0,
                "rel_l2": 0.0,
                "physics_loss": 0.0,
                "corr_indicator": 0.0,
            }

    sample_count = 0

    with torch.inference_mode():
        for sample_i, data in enumerate(dataloader):
            k = data["permeability"]
            u = data["darcy"]

            assert u.shape == (1, 1, resolution, resolution), (
                f"Unexpected input shape: {u.shape}"
            )
            assert k.shape == (1, 1, resolution, resolution), (
                f"Unexpected target shape: {k.shape}"
            )

            u_std = u.std(dim=(-2, -1), keepdim=True)

            # Store plotting data for first few samples
            plot_data = {}

            for sensor_density in sensor_densities:
                noisy_inputs = []
                masks = []
                masked_inputs = []
                model_inputs = []

                for noise_level in noise_levels:
                    noise = torch.randn_like(u) * u_std * noise_level
                    u_noisy = u + noise

                    model_input, mask, u_masked = make_sparse_input(
                        u_noisy,
                        sensor_density,
                        darcy_scale,
                    )

                    noisy_inputs.append(u_noisy)
                    masks.append(mask)
                    masked_inputs.append(u_masked)
                    model_inputs.append(model_input)

                # Shape: len(noise_levels), 2, H, W
                model_input_batch = torch.cat(model_inputs, dim=0)

                out_raw = model(model_input_batch)
                k_pred_batch = darcy_mask1(out_raw)

                print(
                    f"Processing sample {sample_i}, "
                    f"p={sensor_density}, output shape: {out_raw.shape}"
                )

                expected_unscaled = k.detach().cpu().numpy()
                pred_unscaled_batch = k_pred_batch.detach().cpu().numpy()

                measures_per_sample = {
                    "mse": [],
                    "rel_l2": [],
                    "physics_loss": [],
                    "corr_indicator": [],
                }

                for i, noise_level in enumerate(noise_levels):
                    pred_i = k_pred_batch[i : i + 1]

                    assert pred_i.shape == k.shape, (
                        f"Output shape {pred_i.shape} does not match target shape {k.shape}"
                    )

                    mse = F.mse_loss(pred_i, k).item()
                    rel_l2 = (
                        torch.linalg.norm(pred_i - k) / torch.linalg.norm(k)
                    ).item()

                    corr = corr_indicator(pred_i, k).item()

                    residuals = phy_informer.forward(
                        {
                            "u": u,
                            "k": pred_i,
                        }
                    )

                    pde_out_arr = residuals["diffusion_u"]
                    pde_core = pde_out_arr[:, :, 2:-2, 2:-2]
                    pde_loss = torch.mean(torch.abs(pde_core)).item()

                    results[(sensor_density, noise_level)]["mse"] += mse
                    results[(sensor_density, noise_level)]["rel_l2"] += rel_l2
                    results[(sensor_density, noise_level)]["physics_loss"] += pde_loss
                    results[(sensor_density, noise_level)]["corr_indicator"] += corr

                    measures_per_sample["mse"].append(mse)
                    measures_per_sample["rel_l2"].append(rel_l2)
                    measures_per_sample["physics_loss"].append(pde_loss)
                    measures_per_sample["corr_indicator"].append(corr)

                if sample_i < 5:
                    plot_data[sensor_density] = {
                        "u": u,
                        "u_noisy_batch": torch.cat(noisy_inputs, dim=0),
                        "mask_batch": torch.cat(masks, dim=0),
                        "u_masked_batch": torch.cat(masked_inputs, dim=0),
                        "expected_unscaled": expected_unscaled,
                        "pred_unscaled_batch": pred_unscaled_batch,
                        "measures_per_sample": measures_per_sample,
                    }

            if sample_i < 5:
                for sensor_density, data_to_plot in plot_data.items():
                    plot_recovered_sparse(
                        noise_levels=noise_levels,
                        sensor_density=sensor_density,
                        sample_i=sample_i,
                        u=data_to_plot["u"],
                        u_noisy=data_to_plot["u_noisy_batch"],
                        mask=data_to_plot["mask_batch"],
                        u_masked=data_to_plot["u_masked_batch"],
                        expected_unscaled=data_to_plot["expected_unscaled"],
                        predvar_unscaled=data_to_plot["pred_unscaled_batch"],
                        measures_per_sample=data_to_plot["measures_per_sample"],
                        output_dir=output_dir,
                    )

            sample_count += 1

    if sample_count > 0:
        for sensor_density in sensor_densities:
            for noise_level in noise_levels:
                for metric_name in results[(sensor_density, noise_level)]:
                    results[(sensor_density, noise_level)][metric_name] /= sample_count

    print("Sparse test results:")
    for sensor_density in sensor_densities:
        print(f"\nSensor density p={sensor_density}")
        for noise_level in noise_levels:
            r = results[(sensor_density, noise_level)]
            print(
                f"Noise {noise_level}: "
                f"MSE={r['mse']:.6e}, "
                f"RelL2={r['rel_l2']:.6e}, "
                f"Physics={r['physics_loss']:.6e}, "
                f"Corr={r['corr_indicator']:.6e}"
            )

    summary_path = output_dir / "results_summary_sparse.txt"
    with open(summary_path, "w") as f:
        f.write(
            "Sensor Density\tNoise Level\tMSE Loss\tRelative L2 Loss\t"
            "Physics Loss\tCorrelation Indicator\n"
        )

        for sensor_density in sensor_densities:
            for noise_level in noise_levels:
                r = results[(sensor_density, noise_level)]
                f.write(
                    f"{sensor_density}\t"
                    f"{noise_level}\t"
                    f"{r['mse']:.6e}\t"
                    f"{r['rel_l2']:.6e}\t"
                    f"{r['physics_loss']:.6e}\t"
                    f"{r['corr_indicator']:.6e}\n"
                )

    print(f"Saved sparse summary to: {summary_path}")


def plot_recovered_sparse(
    noise_levels,
    sensor_density,
    sample_i,
    u,
    u_noisy,
    mask,
    u_masked,
    expected_unscaled,
    predvar_unscaled,
    measures_per_sample,
    output_dir,
):
    rows = 1 + len(noise_levels)
    cols = 5

    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4.6, rows * 3.2),
        squeeze=False,
        constrained_layout=True,
    )

    fig.suptitle(
        f"Sparse inverse Darcy results for sample {sample_i}, p={sensor_density}",
        fontsize=14,
    )

    def plot_with_colorbar(
        col,
        row,
        data,
        title,
        *,
        cmap="viridis",
        vmin=None,
        vmax=None,
    ):
        image = np.squeeze(data)
        axis = cast(Axes, ax[row, col])
        im = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    plot_with_colorbar(
        0,
        0,
        u[0, 0].cpu().numpy(),
        "Clean full pressure $u$",
        cmap="viridis",
    )

    plot_with_colorbar(
        1,
        0,
        expected_unscaled[0, 0],
        "True permeability $a$",
        cmap="magma",
        vmin=3.0,
        vmax=12.0,
    )

    for col in range(2, cols):
        ax[0, col].axis("off")

    ax[0, 2].text(
        0.5,
        0.5,
        "Sparse input format:\n$[M \\odot u, M] \\mapsto a$",
        ha="center",
        va="center",
        fontsize=10,
    )

    diff_vmax = max(
        np.max(np.abs(predvar_unscaled[i, 0] - expected_unscaled[0, 0]))
        for i in range(len(noise_levels))
    )

    for i, noise_level in enumerate(noise_levels):
        row = i + 1

        plot_with_colorbar(
            0,
            row,
            u_noisy[i, 0].cpu().numpy(),
            f"Noisy pressure\nnoise={noise_level}",
            cmap="viridis",
        )

        plot_with_colorbar(
            1,
            row,
            mask[i, 0].cpu().numpy(),
            f"Mask $M$\np={sensor_density}",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        plot_with_colorbar(
            2,
            row,
            u_masked[i, 0].cpu().numpy(),
            "$M \\odot u$",
            cmap="viridis",
        )

        plot_with_colorbar(
            3,
            row,
            predvar_unscaled[i, 0],
            f"Prediction\nnoise={noise_level}",
            cmap="magma",
            vmin=3.0,
            vmax=12.0,
        )

        diff = np.abs(predvar_unscaled[i, 0] - expected_unscaled[0, 0])

        plot_with_colorbar(
            4,
            row,
            diff,
            (
                f"Abs diff\n"
                f"MSE={measures_per_sample['mse'][i]:.3e}\n"
                f"RelL2={measures_per_sample['rel_l2'][i]:.3e}\n"
                f"Corr={measures_per_sample['corr_indicator'][i]:.3e}"
            ),
            cmap="inferno",
            vmin=0.0,
            vmax=diff_vmax,
        )

    density_name = str(sensor_density).replace(".", "p")

    fig.savefig(
        output_dir / f"results_sparse_p_{density_name}_sample_{sample_i}.png",
        dpi=200,
        bbox_inches="tight",
    )

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
    model_dir = sys.argv[1]

    cfg, model_path, model_dir = load_model_dir(model_dir)

    model_name = model_path.stem + "_" + model_dir.name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = (
        pathlib.Path("./test_results_sparse")
        / model_name
        / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded config from: {model_dir / '.hydra' / 'config.yaml'}")
    print(f"Using newest checkpoint: {model_path}")
    print(f"Output directory: {output_dir}")

    run_test(cfg, model_path, output_dir)
    
    