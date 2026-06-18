from datetime import datetime
import sys

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
from utils import darcy_mask1, corr_indicator, make_random_mask, get_pde_loss, load_model_dir

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
        cfg.data.test_path if hasattr(cfg.data, "test_path") else "./datasets/Darcy_241/piececonst_r241_N1024_test.hdf5",
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

            # Constant baseline for this sample
            k_mean = torch.full_like(k, 7.5)

            baseline_rel_l2 = (
                torch.linalg.norm(k_mean - k) / torch.linalg.norm(k)
            ).item()

            baseline_mse = F.mse_loss(k_mean, k).item()

            print("constant baseline mse:", baseline_mse)
            print("constant baseline rel_l2:", baseline_rel_l2)

            assert u.shape == (1, 1, resolution, resolution)
            assert k.shape == (1, 1, resolution, resolution)

            u_std = u.std(dim=(-2, -1), keepdim=True)

            plot_data = {}

            for sensor_density in sensor_densities:
                noisy_inputs = []
                masked_inputs = []
                model_inputs = []

                # Same mask for all noise levels at this density/sample
                mask = make_random_mask(u, sensor_density)

                actual_density = mask.mean().item()
                observed_pixels = mask.sum().item()

                print(
                    f"sample={sample_i}, "
                    f"p target={sensor_density}, "
                    f"p actual={actual_density:.6f}, "
                    f"observed={observed_pixels:.0f}"
                )

                for noise_level in noise_levels:
                    noise = torch.randn_like(u) * u_std * noise_level
                    u_noisy = u + noise

                    u_masked = mask * u_noisy
                    model_input = torch.cat([u_masked / darcy_scale, mask], dim=1)

                    assert model_input.shape[1] == 2

                    assert torch.allclose(
                        model_input[:, 0:1],
                        u_masked / darcy_scale,
                        atol=1e-6,
                    ), "Model input channel 0 is not masked/scaled pressure."

                    assert torch.allclose(
                        model_input[:, 1:2],
                        mask,
                        atol=1e-6,
                    ), "Model input channel 1 is not the mask."

                    assert torch.allclose(
                        u_masked * (1.0 - mask),
                        torch.zeros_like(u_masked),
                        atol=1e-6,
                    ), "u_masked leaks values outside the mask."

                    noisy_inputs.append(u_noisy)
                    masked_inputs.append(u_masked)
                    model_inputs.append(model_input)

                model_input_batch = torch.cat(model_inputs, dim=0)

                out_raw = model(model_input_batch)
                k_pred_batch = darcy_mask1(out_raw)

                assert k_pred_batch.shape == (
                    len(noise_levels),
                    1,
                    resolution,
                    resolution,
                )

                # Evaluate normal sparse/noisy predictions
                measures_per_sample = {
                    "mse": [],
                    "rel_l2": [],
                    "physics_loss": [],
                    "corr_indicator": [],
                }

                for i, noise_level in enumerate(noise_levels):
                    pred_i = k_pred_batch[i : i + 1]

                    assert pred_i.shape == k.shape

                    mse = F.mse_loss(pred_i, k).item()
                    rel_l2 = (
                        torch.linalg.norm(pred_i - k) / torch.linalg.norm(k)
                    ).item()
                    corr = corr_indicator(pred_i, k).item()

                    pde_loss = get_pde_loss(phy_informer, u, pred_i)

                    results[(sensor_density, noise_level)]["mse"] += mse
                    results[(sensor_density, noise_level)]["rel_l2"] += rel_l2
                    results[(sensor_density, noise_level)]["physics_loss"] += pde_loss
                    results[(sensor_density, noise_level)]["corr_indicator"] += corr

                    measures_per_sample["mse"].append(mse)
                    measures_per_sample["rel_l2"].append(rel_l2)
                    measures_per_sample["physics_loss"].append(pde_loss)
                    measures_per_sample["corr_indicator"].append(corr)

                # Mask-only sanity test
                model_input_mask_only = torch.cat(
                    [torch.zeros_like(u), mask],
                    dim=1,
                )

                pred_mask_only = darcy_mask1(model(model_input_mask_only))

                mask_only_mse = F.mse_loss(pred_mask_only, k).item()
                mask_only_rel_l2 = (
                    torch.linalg.norm(pred_mask_only - k) / torch.linalg.norm(k)
                ).item()
                mask_only_corr = corr_indicator(pred_mask_only, k).item()

                print(
                    f"MASK ONLY | sample={sample_i}, p={sensor_density}: "
                    f"MSE={mask_only_mse:.6e}, "
                    f"RelL2={mask_only_rel_l2:.6e}, "
                    f"Corr={mask_only_corr:.6e}"
                )

                if sample_i < 5:
                    plot_data[sensor_density] = {
                        "u": u,
                        "u_noisy_batch": torch.cat(noisy_inputs, dim=0),
                        "mask": mask,
                        "u_masked_batch": torch.cat(masked_inputs, dim=0),
                        "expected_unscaled": k.detach().cpu().numpy(),
                        "pred_unscaled_batch": k_pred_batch.detach().cpu().numpy(),
                        "measures_per_sample": measures_per_sample,
                        "mask_only_result": pred_mask_only.detach().cpu().numpy(),
                    }

            if sample_i < 5:
                for sensor_density, data_to_plot in plot_data.items():
                    plot_recovered_sparse(
                        noise_levels=noise_levels,
                        sensor_density=sensor_density,
                        sample_i=sample_i,
                        u=data_to_plot["u"],
                        u_noisy=data_to_plot["u_noisy_batch"],
                        mask=data_to_plot["mask"],
                        u_masked=data_to_plot["u_masked_batch"],
                        expected_unscaled=data_to_plot["expected_unscaled"],
                        predvar_unscaled=data_to_plot["pred_unscaled_batch"],
                        measures_per_sample=data_to_plot["measures_per_sample"],
                        output_dir=output_dir,
                        mask_only_result=data_to_plot["mask_only_result"],
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
    mask_only_result=None,
):
    rows = 1 + (1 if mask_only_result is not None else 0) + len(noise_levels)
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
        sensor_mask=None,
        draw_sensor_circles=False,
    ):
        image = np.squeeze(data)
        axis = cast(Axes, ax[row, col])
        im = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)

        if draw_sensor_circles and sensor_mask is not None:
            mask_np = np.squeeze(sensor_mask)
            ys, xs = np.where(mask_np > 0.5)

            axis.scatter(
                xs,
                ys,
                s=45,
                facecolors="none",
                edgecolors="white",
                linewidths=0.8,
            )

        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
        
    u_vmin = min(
        np.min(u[0, 0].cpu().numpy()),
        np.min(u_noisy[:, 0].cpu().numpy()),
        np.min(u_masked[:, 0].cpu().numpy()),
    )
    u_vmax = max(
        np.max(u[0, 0].cpu().numpy()),
        np.max(u_noisy[:, 0].cpu().numpy()),
        np.max(u_masked[:, 0].cpu().numpy()),
    )

    plot_with_colorbar(
        0,
        0,
        u[0, 0].cpu().numpy(),
        "Clean full pressure $u$",
        cmap="viridis",
        vmin=u_vmin,
        vmax=u_vmax,
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
    
    def plot_prediction_row(row, i, noise_level):
        plot_with_colorbar(
            0,
            row,
            u_noisy[i, 0].cpu().numpy(),
            f"Noisy pressure\nnoise={noise_level}",
            cmap="viridis",
            vmin=u_vmin,
            vmax=u_vmax,
        )

        plot_with_colorbar(
            1,
            row,
            mask.cpu().numpy(),
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
            vmin=u_vmin,
            vmax=u_vmax,
            sensor_mask=mask.cpu().numpy(),
            draw_sensor_circles=sensor_density < 0.05,  # Only draw circles for very sparse cases
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

    for i, noise_level in enumerate(noise_levels):
        row = i + 1
        plot_prediction_row(row, i, noise_level)
    if mask_only_result is not None:
        row += 1
        plot_with_colorbar(
            0,
            row,
            np.zeros_like(u[0, 0].cpu().numpy()),
            "Mask-only input\n$[0, M] \\mapsto a$",
            cmap="viridis",
            vmin=u_vmin,
            vmax=u_vmax,
        )
        plot_with_colorbar(
            1,
            row,
            mask.cpu().numpy(),
            f"Mask $M$\np={sensor_density}",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plot_with_colorbar(
            3,
            row,
            mask_only_result[0, 0],
            "Mask-only prediction",
            cmap="magma",
            vmin=3.0,
            vmax=12.0,
        )
        

    density_name = str(sensor_density).replace(".", "p")

    fig.savefig(
        output_dir / f"results_sparse_p_{density_name}_sample_{sample_i}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

if __name__ == "__main__":
    model_dir = sys.argv[1]

    cfg, model_path, model_dir = load_model_dir(model_dir)

    model_name = model_path.stem + "_" + model_dir.name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = (
        pathlib.Path("./test_results_sparse_final_final")
        / model_name
        / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded config from: {model_dir / '.hydra' / 'config.yaml'}")
    print(f"Using newest checkpoint: {model_path}")
    print(f"Output directory: {output_dir}")

    run_test(cfg, model_path, output_dir)
    
    