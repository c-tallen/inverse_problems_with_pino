# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Krzysztof Wilczewski.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Krzysztof Wilczewski, 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Any, cast
from hydra.utils import to_absolute_path
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.checkpoint import save_checkpoint, load_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from diffusion_eq import Diffusion
from utils import CustomDataset, darcy_mask1, relative_l2_loss, validation_step, total_variance

def validation_step_sparse(
    model,
    dataloader,
    epoch,
    permeability_scale,
    darcy_scale,
    sensor_densities,
):
    """Sparse validation step for inverse Darcy: [M * u, M] -> k."""
    model.eval()

    metrics = {}

    with torch.no_grad():
        for sensor_density in sensor_densities:
            sensor_density = float(sensor_density)

            data_loss_epoch = 0.0

            # Keep last batch for plotting, same as your original validation_step
            last_k = None
            last_k_pred = None
            last_u = None
            last_mask = None
            last_u_masked = None

            for data in dataloader:
                k = data["permeability"]
                u = data["darcy"]

                # Create sparse observation mask M
                mask = make_random_mask(u, sensor_density)

                # Apply mask to pressure field
                u_masked = mask * u

                # Scale pressure, but not the mask
                u_masked_scaled = u_masked / darcy_scale

                # Input: [M * u, M]
                model_input = torch.cat([u_masked_scaled, mask], dim=1)

                out_raw = model(model_input)
                k_pred = darcy_mask1(out_raw)

                data_loss_epoch += relative_l2_loss(k_pred, k).item()

                last_k = k
                last_k_pred = k_pred
                last_u = u
                last_mask = mask
                last_u_masked = u_masked

            avg_relative_l2 = data_loss_epoch / len(dataloader)

            metrics[f"rel_l2_p_{sensor_density}"] = avg_relative_l2

            # Convert last batch to numpy for plotting
            expected_unscaled = last_k.detach().cpu().numpy()
            predvar = last_k_pred.detach().cpu().numpy()
            u_np = last_u.detach().cpu().numpy()
            mask_np = last_mask.detach().cpu().numpy()
            u_masked_np = last_u_masked.detach().cpu().numpy()

            # Plot similar to your original validation_step, but with sparse input info
            fig, ax = plt.subplots(1, 5, figsize=(35, 5))

            def plot_with_colorbar(i, data, title):
                d_min = np.min(data[0, 0])
                d_max = np.max(data[0, 0])
                im = ax[i].imshow(data[0, 0], vmin=d_min, vmax=d_max)
                plt.colorbar(im, ax=ax[i])
                ax[i].set_title(title)
                ax[i].set_xticks([])
                ax[i].set_yticks([])

            plot_with_colorbar(0, u_np, "Full pressure $u$")
            plot_with_colorbar(1, mask_np, f"Mask $M$, p={sensor_density}")
            plot_with_colorbar(2, u_masked_np, "$M \\odot u$")
            plot_with_colorbar(3, expected_unscaled, "True permeability")
            plot_with_colorbar(4, predvar, "Predicted permeability")

            fig.savefig(f"results_sparse_p_{sensor_density}_epoch_{epoch}.png")
            plt.close()

    model.train()
    return metrics

def make_random_mask(
    u: torch.Tensor,
    sensor_density: float,
) -> torch.Tensor:
    """
    Create a random binary observation mask M with approximately
    sensor_density fraction of observed points.

    Args:
        u: Tensor of shape (B, C, H, W), normally pressure field.
        sensor_density: Fraction of observed grid points, e.g. 1.0, 0.5, 0.25.

    Returns:
        mask: Tensor of shape (B, 1, H, W), with values 0 or 1.
    """
    assert 0.0 < sensor_density <= 1.0, (
        f"sensor_density must be in (0, 1], got {sensor_density}"
    )

    batch_size, _, height, width = u.shape

    if sensor_density == 1.0:
        return torch.ones(
            batch_size,
            1,
            height,
            width,
            device=u.device,
            dtype=u.dtype,
        )

    mask = torch.rand(
        batch_size,
        1,
        height,
        width,
        device=u.device,
        dtype=u.dtype,
    ) < sensor_density

    return mask.to(dtype=u.dtype)

@hydra.main(version_base="1.3", config_path="conf", config_name="sparse_pino.yaml")
def main(cfg: DictConfig):
    """Main function for the Darcy physics-informed FNO."""
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LaunchLogger.initialize()
    DistributedManager.initialize()

    permeability_scale = cfg.scaling.permeability
    darcy_scale = cfg.scaling.darcy
    resolution = cfg.data.resolution
    mappings_dict = OmegaConf.to_container(cfg.mappings, resolve=True)
    
    # Use Diffusion equation for the Darcy PDE
    forcing_fn = cfg.physics_forcing_term
    darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)

    dataset = CustomDataset(
        to_absolute_path(cfg.data.train_path),
        mappings=mappings_dict,
        device=device,
        res=resolution,
    )
    validation_dataset = CustomDataset(
        to_absolute_path(cfg.data.validation_path),
        mappings=mappings_dict,
        device=device,
        res=resolution,
    )

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False)

    fd_dx = 1.0 / float(resolution - 1)

    model = FNO(
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    ).to(device)
    
    if cfg.physics_weight > 0.0:
        phy_informer = PhysicsInformer(
            required_outputs=["diffusion_u"],
            equations=darcy,
            grad_method="finite_difference",
            device=str(device),
            fd_dx=fd_dx,
        )
    else:
        phy_informer = None

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.start_lr,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.step_size,
        gamma=cfg.gamma,
    )

    checkpoint_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
    if checkpoint_dir.exists():
        loaded_epoch = load_checkpoint(
            str(checkpoint_dir),
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = loaded_epoch + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found, starting training from scratch.")
    
    if cfg.max_noise > 0.0:
        print(f"Adding noise to Darcy velocity with max noise level: {cfg.max_noise}")
    else:
        print("No noise will be added to Darcy velocity.")
    
    print("Dataloader length:", len(dataloader))
    for epoch in range(start_epoch, cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=max(1, len(dataloader) // 20),
        ) as log:
            for data in dataloader:
                optimizer.zero_grad()

                k = data["permeability"]
                u = data["darcy"]

                max_noise = cfg.max_noise

                if max_noise > 0.0:
                    u_std = u.std(dim=(-2, -1), keepdim=True)
                    alpha = torch.rand(u.shape[0], 1, 1, 1, device=u.device) * max_noise
                    noise = torch.randn_like(u) * u_std * alpha
                    u_observed = u + noise
                else:
                    u_observed = u

                # Randomly sample one sensor density for this batch
                sensor_densities = list(cfg.sensor_densities)
                density_idx = torch.randint(
                    low=0,
                    high=len(sensor_densities),
                    size=(1,),
                    device=u.device,
                ).item()
                sensor_density = float(sensor_densities[density_idx])

                # Create sparse sensor mask M
                mask = make_random_mask(u_observed, sensor_density)

                # Sparse observed pressure field M * u
                u_masked = mask * u_observed

                # Scale pressure values, not the mask
                u_masked_scaled = u_masked / darcy_scale

                # Model input: [M * u, M]
                u_input = torch.cat([u_masked_scaled, mask], dim=1)

                # Inverse model: [M * u, M] -> k
                out_raw = model(u_input)

                # Constrain predicted permeability to [3, 12]
                k_pred = darcy_mask1(out_raw)

                assert k_pred.shape == k.shape, (
                    f"Output shape {k_pred.shape} does not match target shape {k.shape}"
                )

                # Data loss: predicted permeability vs true permeability
                loss_data = F.mse_loss(k_pred, k)

                # PDE loss: enforce Darcy equation using full simulated u and predicted k
                if phy_informer is not None:
                    residuals = phy_informer.forward(
                        {
                            "u": u,
                            "k": k_pred,
                        }
                    )

                    pde_out_arr = residuals["diffusion_u"]

                    pde_core = pde_out_arr[:, :, 2:-2, 2:-2]
                    loss_pde = torch.mean(torch.abs(pde_core))

                    loss_tv = total_variance(k_pred)

                    weighted_pde = cfg.physics_weight * loss_pde
                    weighted_tv = cfg.tv_weight * loss_tv

                    loss = loss_data + weighted_pde + weighted_tv
                else:
                    loss_tv = torch.tensor(0.0, device=device)
                    loss_pde = torch.tensor(0.0, device=device)
                    weighted_pde = torch.tensor(0.0, device=device)
                    weighted_tv = torch.tensor(0.0, device=device)

                    loss = loss_data

                loss.backward()
                optimizer.step()
                log.log_minibatch(
                    {
                        "loss_data": loss_data.detach().item(),
                        "loss_pde": loss_pde.detach().item(),
                        "weighted_pde": weighted_pde.detach().item(),
                        "loss_tv": loss_tv.detach().item(),
                        "weighted_tv": weighted_tv.detach().item(),
                        "loss_total": loss.detach().item(),
                        "sensor_density": sensor_density,
                    }
                )
            scheduler.step()
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            
        with LaunchLogger("valid", epoch=epoch) as log:
            validation_metrics = validation_step_sparse(
                model,
                validation_dataloader,
                epoch,
                permeability_scale,
                darcy_scale,
                cfg.sensor_densities,
            )
            log.log_epoch(validation_metrics)
        if epoch % cfg.checkpoint_freq == 0 or epoch == cfg.max_epochs - 1:
            save_checkpoint(
                str(checkpoint_dir),
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()
