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
import matplotlib.pyplot as plt
import numpy as np
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
from utils import  HDF5MapStyleDataset, CustomDataset


def darcy_mask1(x: torch.Tensor) -> torch.Tensor:
    """Map raw network output to permeability range [3, 12]."""
    return torch.sigmoid(x) * 9.0 + 3.0


def darcy_mask2(x: torch.Tensor) -> torch.Tensor:
    """Binarized permeability mask used only for visualization in the original code."""
    x = torch.sigmoid(x)
    x = torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))
    return x * 9.0 + 3.0


def total_variance(x: torch.Tensor) -> torch.Tensor:
    """
    Total variation regularization for NCHW tensors.
    x shape: [batch, channels, height, width]
    """
    tv_x = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_y = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_x + tv_y


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Similar spirit to LpLoss(size_average=True).
    Computes mean relative L2 error over batch.
    """
    batch_size = pred.shape[0]
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)

    diff_norm = torch.linalg.norm(pred_flat - target_flat, dim=1)
    target_norm = torch.linalg.norm(target_flat, dim=1)

    return torch.mean(diff_norm / (target_norm + eps))

def validation_step(model, dataloader, epoch, permeability_scale, darcy_scale):
    """Validation Step"""
    model.eval()

    with torch.no_grad():
        data_loss_epoch = 0.0
        for data in dataloader:
            k = data["permeability"] 
            k_scaled = k / permeability_scale
            u = data["darcy"]
            u_scaled = u / darcy_scale
            out_raw = model(u_scaled)
            out_scaled = darcy_mask1(out_raw)

            data_loss_epoch += relative_l2_loss(out_scaled, k_scaled).item()

        # convert data to numpy
        expected_unscaled = k.detach().cpu().numpy()
        predvar = out_scaled.detach().cpu().numpy()
        predvar_unscaled = predvar * permeability_scale

        # plotting
        fig, ax = plt.subplots(1, 3, figsize=(25, 5))

        def plot_with_colorbar(i, data, title):
            d_min = np.min(data[0, 0])
            d_max = np.max(data[0, 0])
            im = ax[i].imshow(data[0, 0], vmin=d_min, vmax=d_max)
            plt.colorbar(im, ax=ax[i])
            ax[i].set_title(title)
        
        plot_with_colorbar(0, expected_unscaled, "True")
        plot_with_colorbar(1, predvar_unscaled, "Pred")
        plot_with_colorbar(2, np.abs(predvar_unscaled - expected_unscaled), "Difference")

        fig.savefig(f"results_{epoch}.png")
        plt.close()
        return data_loss_epoch / len(dataloader)


@hydra.main(version_base="1.3", config_path="conf", config_name="neural_operator.yaml")
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
    
    print("Dataloader length:", len(dataloader))
    for epoch in range(cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=len(dataloader) // 20,
        ) as log:
            for data in dataloader:
                optimizer.zero_grad()

                k = data["permeability"]
                u = data["darcy"]

                u_scaled = u / darcy_scale
                k_scaled = k / permeability_scale

                # Inverse model: u -> k
                out_raw = model(u_scaled)

                # Original code constrains predicted permeability to [3, 12]
                out_scaled = darcy_mask1(out_raw)

                # Unscale for PDE loss
                k_pred = out_scaled * permeability_scale

                assert out_scaled.shape == k_scaled.shape, (
                    f"Output shape {out_scaled.shape} does not match target shape {k_scaled.shape}"
                )

                # Data loss: predicted permeability vs true permeability
                loss_data = F.mse_loss(out_scaled, k_scaled)

                # PDE loss: enforce Darcy equation using observed u and predicted k
                if phy_informer is not None:
                    residuals = phy_informer.forward(
                        {
                            "u": u,
                            "k": k_pred,
                        }
                    )

                    pde_out_arr = residuals["diffusion_u"]

                    # Match the spirit of the original: ignore boundary region
                    pde_core = pde_out_arr[:, :, 2:-2, 2:-2]
                    loss_pde = torch.mean(torch.abs(pde_core))
                else:
                    loss_pde = torch.tensor(0.0, device=device)

                # Total variation regularization on raw model output, like original code
                loss_tv = total_variance(out_scaled)

                # Original pretraining loss:
                # pino_loss = 0.2 * loss_f + loss_data + 0.01 * loss_TV
                weighted_pde = cfg.physics_weight * loss_pde
                weighted_tv = cfg.tv_weight * loss_tv

                loss = loss_data + weighted_pde + weighted_tv

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
                    }
                )
            scheduler.step()
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            
        with LaunchLogger("valid", epoch=epoch) as log:
            validation_metrics = validation_step(
                model,
                validation_dataloader,
                epoch,
                permeability_scale,
                darcy_scale,
            )
            log.log_epoch({"Validation error": validation_metrics})

        save_checkpoint(
            "./checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
