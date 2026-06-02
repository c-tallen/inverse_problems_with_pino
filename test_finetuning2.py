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
from utils import darcy_mask1, total_variance, corr_indicator


# TODO: Make hardcoded values into config options
def run_test(cfg: DictConfig, model_path: pathlib.Path, output_dir: pathlib.Path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Loading model and dataset on device:", device)
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    mappings_dict = OmegaConf.to_container(cfg.mappings, resolve=True)
    dataset = CustomDataset(cfg.data.validation_path, device=device, mappings=mappings_dict, res=cfg.data.resolution)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    permeability_scale = cfg.scaling.permeability
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
    
    model = physicsnemo.Module.from_checkpoint(str(model_path)).to(device)
    
    print("Starting inference on validation dataset...")
    metrics = {
        "mse": [0.0 for _ in noise_levels],
        "rel_l2": [0.0 for _ in noise_levels],
        "physics_loss": [0.0 for _ in noise_levels],
        "corr_inct": [0.0 for _ in noise_levels],
    }
    sample_count = 0

    for sample_i, data in enumerate(dataloader):
        k = data["permeability"]
        u = data["darcy"]
        assert u.shape == (1, 1, resolution, resolution), f"Unexpected input shape: {u.shape}"
        assert k.shape == (1, 1, resolution, resolution), f"Unexpected target shape: {k.shape}"
        
        # Create one noisy version of u for each fixed noise level.
        noisy_inputs = []
        pred_unscaled_batch = []
        expected_unscaled = k.detach().cpu().numpy()
        u_std = u.std(dim=(-2, -1), keepdim=True)
        
        measures_per_sample = {
            "mse": [],
            "rel_l2": [],
            "physics_loss": [],
            "corr_inct": [],
        }

        for i, alpha in enumerate(noise_levels):
            noise = torch.randn_like(u) * u_std * alpha
            u_noisy = u + noise
            print("Finetuning sample", sample_i, "noise level:", alpha)
            noisy_inputs.append(u_noisy)
            k_pred = refine_output_field_and_evaluate(
                i,
                u_noisy,
                k,
                phy_informer,
                model,
                darcy_scale,
                metrics,
                measures_per_sample,
                alpha
            )
            pred_unscaled_batch.append(k_pred)
        u_noisy_batch = torch.cat(noisy_inputs, dim=0).detach().cpu()
        pred_unscaled_batch = torch.cat(pred_unscaled_batch, dim=0).detach().cpu()
        if sample_i < 5:
            print("Plotting results for sample", sample_i)
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
        for key in metrics:
            metrics[key] = [loss / sample_count for loss in metrics[key]]
    print("Total MSE for each noise level:")
    for i, error in enumerate(metrics["mse"]):
        print(f"Noise level {noise_levels[i]}: {error}")
    print("Total relative L2 for each noise level:")
    for i, error in enumerate(metrics["rel_l2"]):
        print(f"Noise level {noise_levels[i]}: {error}")
    print("Total physics loss for each noise level:")
    for i, error in enumerate(metrics["physics_loss"]):
        print(f"Noise level {noise_levels[i]}: {error}")
    print("Total correlation indicator for each noise level:")
    for i, corr in enumerate(metrics["corr_inct"]):
        print(f"Noise level {noise_levels[i]}: {corr}")
        
    with open(output_dir / "results_summary.txt", "w") as f:
        f.write("Noise Level\tMSE Loss\tRelative L2 Loss\tPhysics Loss\tCorrelation Indicator\n")
        for i in range(len(noise_levels)):
            f.write(
                f"{noise_levels[i]}\t{metrics['mse'][i]:.6e}\t{metrics['rel_l2'][i]:.6e}\t{metrics['physics_loss'][i]:.6e}\t{metrics['corr_inct'][i]:.6e}\n"
            )

def refine_output_field_and_evaluate(
    i,
    u_noisy,
    k,
    phy_informer,
    model,
    darcy_scale,
    metrics,
    measures_per_sample,
    alpha,
    lambda_op=0.2,
    lambda_tv=1e-3,
):
    model.eval()

    u_scaled = u_noisy / darcy_scale
    u_for_pde = smooth_field(u_noisy, kernel_size=5)

    # 1. Get neural-operator initial prediction.
    with torch.no_grad():
        k_raw_anchor = model(u_scaled)
        k_anchor = darcy_mask1(k_raw_anchor)

    z_opt = k_raw_anchor.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([z_opt], lr=1e-3)
    
    num_steps = int(100 / (1.0 + 10.0 * alpha))
    num_steps = max(num_steps, 20)
    for step in range(num_steps):
        optimizer.zero_grad()

        k_phys = darcy_mask1(z_opt)

        residuals = phy_informer.forward(
            {
                "u": u_for_pde,
                "k": k_phys,
            }
        )

        pde_out_arr = residuals["diffusion_u"]
        pde_core = pde_out_arr[:, :, 2:-2, 2:-2]

        loss_pde = torch.mean(torch.abs(pde_core))
        loss_op = torch.mean((k_phys - k_anchor) ** 2)
        loss_tv = total_variance(k_phys)

        loss = loss_pde + lambda_op * loss_op + lambda_tv * loss_tv
        
        if step % 20 == 0 or step == num_steps - 1:
            print(
                f"Step {step+1}/{num_steps}, "
                f"Loss: {loss.item():.6e}, "
                f"PDE Loss: {loss_pde.item():.6e}, "
                f"OP Loss: {loss_op.item():.6e}, "
                f"TV Loss: {loss_tv.item():.6e}"
            )

        loss.backward()
        optimizer.step()

    print("Output-field refinement completed. Evaluating on test sample...")

    with torch.no_grad():
        k_pred = darcy_mask1(z_opt)

        mse = F.mse_loss(k_pred, k).item()
        corr = corr_indicator(k_pred, k).item()
        rel_l2 = (
            torch.linalg.norm(k_pred - k) / torch.linalg.norm(k)
        ).item()

    k_pred_eval = k_pred.detach()
    residuals = phy_informer.forward(
        {
            "u": u_for_pde,
            "k": k_pred_eval,
        }
    )

    pde_out_arr = residuals["diffusion_u"]
    pde_core = pde_out_arr[:, :, 2:-2, 2:-2]
    physics_loss = torch.mean(torch.abs(pde_core)).item()

    metrics["mse"][i] += mse
    metrics["rel_l2"][i] += rel_l2
    metrics["physics_loss"][i] += physics_loss
    metrics["corr_inct"][i] += corr

    measures_per_sample["mse"].append(mse)
    measures_per_sample["rel_l2"].append(rel_l2)
    measures_per_sample["physics_loss"].append(physics_loss)
    measures_per_sample["corr_inct"].append(corr)

    return k_pred

def smooth_field(x, kernel_size=5):
    pad = kernel_size // 2
    return F.avg_pool2d(
        F.pad(x, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
    )

def plot_recovered(noise_levels, sample_i, u, u_noisy, expected_unscaled, predvar_unscaled, measures_per_sample, output_dir):
    if isinstance(predvar_unscaled, torch.Tensor):
        predvar_unscaled = predvar_unscaled.detach().cpu().numpy()
        
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
        plot_with_colorbar(2, row, diff, f"Abs Diff\nMSE: {measures_per_sample['mse'][i]:.3e}\nCorr: {measures_per_sample['corr_inct'][i]:.3e}", cmap="inferno", vmin=0.0, vmax=diff_vmax)

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
    model_dir = sys.argv[1]

    cfg, model_path, model_dir = load_model_dir(model_dir)

    model_name = model_path.stem + "_" + model_dir.name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = (
        pathlib.Path("./test_finetuned_results")
        / model_name
        / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded config from: {model_dir / '.hydra' / 'config.yaml'}")
    print(f"Using newest checkpoint: {model_path}")
    print(f"Output directory: {output_dir}")

    run_test(cfg, model_path, output_dir)
    