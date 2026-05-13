import numpy as np
import torch
import physicsnemo
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import CustomDataset

def get_noisy_data(u, noise_levels, std: float):
    noise_levels = torch.tensor(noise_levels, device=u.device).view(-1, 1, 1, 1)
    noise_stds = noise_levels * float(std)
    assert noise_stds.shape == (len(noise_levels), 1, 1, 1), "Noise std shape mismatch"
    u_exp = u.unsqueeze(0).expand(len(noise_levels), *u.shape)
    noise = torch.randn_like(u_exp) * noise_stds
    u_noisy = u_exp + noise
    assert u_noisy.shape[0] == len(noise_levels)
    return u_noisy

def run_test(noise_levels):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Loading model and dataset on device:", device)
    model = physicsnemo.Module.from_checkpoint("model.mdlus").to("cuda")
    model.eval()
    dataset = CustomDataset("./datasets/Darcy_241/validation.hdf5", device=device, mappings={"permeability": "Kcoeff", "darcy": "sol"})

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("Starting inference on validation dataset...")
    with torch.inference_mode():
        loss_epoch = 0
        for i, data in zip(range(20), dataloader):
            k = data["permeability"] 
            k_scaled = k / 4.49996e00
            u = data["darcy"]
            u_noisy = get_noisy_data(u, noise_levels, std=3.88433e-03)
            
            u_scaled = u_noisy / 3.88433e-03
            out = model(u_scaled)
            print(f"Processing sample {i}")
            expected_unscaled = k.detach().cpu().numpy()
            expected = k_scaled.detach().cpu().numpy()
            predvar = out.detach().cpu().numpy()
            mse = F.mse_loss(k_scaled, out).item()

            # plotting
            fig, ax = plt.subplots(2, 3, figsize=(25, 10))
            fig.suptitle(f"MSE Loss: {mse:.6e}")

            def plot_with_colorbar(data, title):
                i = plot_with_colorbar.idx % 3
                j = plot_with_colorbar.idx // 3
                d_min = np.min(data[0, 0])
                d_max = np.max(data[0, 0])
                im = ax[j, i].imshow(data[0, 0], vmin=d_min, vmax=d_max)
                plt.colorbar(im, ax=ax[j, i])
                ax[j, i].set_title(title)
                plot_with_colorbar.idx += 1
            plot_with_colorbar.idx = 0
        
            plot_with_colorbar(expected_unscaled, "True")
            plot_with_colorbar(predvar, "Pred")
            plot_with_colorbar(np.abs(predvar - expected), f"Difference (MSE: {mse:.6e})")
            plot_with_colorbar(u_noisy[0, 0].cpu().numpy(), "Noisy Input")
            plot_with_colorbar(u[0, 0].cpu().numpy(), "Clean Input")

            fig.savefig(f"./test_results/results_{i}.png")
            
            plt.close()

            loss_epoch += mse


if __name__ == "__main__":
    noise_levels = [0.01, 0.05, 0.1]
    total_error = run_test(noise_levels)