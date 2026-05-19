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


# TODO: Make hardcoded values into config options
def get_noisy_data(u, noise_levels, std: float):
    # u shape: (1, 1, H, W) or (1, C, H, W)
    device = u.device
    dtype = u.dtype
    
    # Convert noise_levels to tensor
    noise_levels_tensor = torch.tensor(noise_levels, dtype=dtype, device=device).view(-1, 1, 1, 1)
    noise_stds = noise_levels_tensor * float(std)
    
    # Repeat u for each noise level
    # if noise_levels contains the method __len__
    if hasattr(noise_levels, "__len__"):
        u_expanded = u.repeat(len(noise_levels), 1, 1, 1)  # [3, 1, 240, 240]
    else:
        u_expanded = u

    # Create and apply noise
    noise = torch.randn_like(u_expanded) * noise_stds
    u_noisy = u_expanded + noise
    assert u_noisy.shape[0] == noise_levels_tensor.shape[0]
    return u_noisy

def run_test(noise_levels) -> tuple[list[float], list[float], list[float]]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Loading model and dataset on device:", device)
    model = physicsnemo.Module.from_checkpoint("model.mdlus").to(device)
    model.eval()
    dataset = CustomDataset("./datasets/Darcy_241/piececonst_r241_N1024_smooth2.hdf5", device=device, mappings={"permeability": "Kcoeff", "darcy": "sol"})
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    fd_dx = 1.0 / 240.0  # Assuming resolution is 240, adjust if different
    forcing_fn = 1.0
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
        sample_count = 0
        for sample_i, data in enumerate(dataloader):
            k = data["permeability"] 
            k_scaled = k / 4.49996e00
            u = data["darcy"]
            u_noisy = get_noisy_data(u, noise_levels, std=3.88433e-03)
            
            u_scaled = u_noisy / 3.88433e-03
            out = model(u_scaled)
            print(out.shape)
            print(f"Processing sample {sample_i}")
            expected_unscaled = k.detach().cpu().numpy()
            # expected = k_scaled.detach().cpu().numpy()
            pred_batch = out.detach().cpu().numpy()
            pred_unscaled_batch = pred_batch * 4.49996e00
            mse_per_sample = []
            for i in range(len(noise_levels)):
                pred_i = out[i:i+1]  # Shape: [1, 1, H, W]
                pred_i_unscaled = pred_i * 4.49996e00
                mse = F.mse_loss(k_scaled, pred_i).item()
                rel_l2 = (
                    torch.linalg.norm(pred_i_unscaled - k) / torch.linalg.norm(k)
                ).item()
                mse_per_sample.append(mse)
                loss_mse[i] += mse
                loss_rel_l2[i] += rel_l2
                # calculate physics loss
                assert pred_i.shape == k_scaled.shape, f"Output shape {pred_i.shape} does not match target shape {k_scaled.shape}"
                residuals = phy_informer.forward(
                    {
                        "u": u,
                        "k": pred_i,
                    }
                )
                pde_out_arr = residuals["diffusion_u"]

                pde_out_arr = F.pad(
                    pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                )
                physics_loss[i] += F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr)).item()

            # plotting
            if sample_i < 5:  # Limit the number of plotted samples
                plot_recovered(noise_levels, sample_i, u, u_noisy, expected_unscaled, pred_unscaled_batch, mse_per_sample)
            sample_count += 1

        if sample_count > 0:
            # Calculate the average mse loss for each noise level
            loss_mse = [loss / sample_count for loss in loss_mse]
            loss_rel_l2 = [loss / sample_count for loss in loss_rel_l2]
            physics_loss = [loss / sample_count for loss in physics_loss]
        return loss_mse, loss_rel_l2, physics_loss

def plot_recovered(noise_levels, sample_i, u, u_noisy, expected_unscaled, predvar_unscaled, mse_per_sample):
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
    ax[0, 2].text(0.5, 0.5, "No prediction\nfor clean input", ha="center", va="center", fontsize=10)

    diff_vmax = max(np.abs(predvar_unscaled[i, 0] - expected_unscaled[0, 0]).max() for i in range(len(noise_levels)))
    for i in range(len(noise_levels)):
        row = i + 1
        plot_with_colorbar(0, row, u_noisy[i, 0].cpu().numpy(), f"Noisy Input\nNoise: {noise_levels[i]}", cmap="viridis")
        plot_with_colorbar(1, row, predvar_unscaled[i, 0], f"Prediction\nNoise: {noise_levels[i]}", cmap="magma")
        plot_with_colorbar(2, row, np.abs(predvar_unscaled[i, 0] - expected_unscaled[0, 0]), f"Abs Diff\nMSE: {mse_per_sample[i]:.3e}", cmap="inferno", vmin=0.0, vmax=diff_vmax)

    fig.savefig(f"./test_results/results_{sample_i}.png", dpi=200, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    noise_levels = noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    loss_mse, loss_rel_l2, physics_loss = run_test(noise_levels)
    print("Total MSE for each noise level:")
    for i, error in enumerate(loss_mse):
        print(f"Noise level {noise_levels[i]}: {error}")
    print("Total relative L2 for each noise level:")
    for i, error in enumerate(loss_rel_l2):
        print(f"Noise level {noise_levels[i]}: {error}")
    print("Total physics loss for each noise level:")
    for i, error in enumerate(physics_loss):
        print(f"Noise level {noise_levels[i]}: {error}")