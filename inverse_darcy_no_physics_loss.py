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
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from diffusion_eq import Diffusion
from utils import  HDF5MapStyleDataset, CustomDataset


def validation_step(model, dataloader, epoch, permeability_scale, darcy_scale, physics_weight, residual_normalizer):
    """Validation Step"""
    model.eval()

    with torch.no_grad():
        data_loss_epoch = 0.0
        physics_loss_epoch = 0.0
        for data in dataloader:
            k = data["permeability"] 
            k_scaled = k / permeability_scale
            u = data["darcy"]
            u_scaled = u / darcy_scale
            out = model(u_scaled)

            data_loss_epoch += F.mse_loss(k_scaled, out).item()

        # convert data to numpy
        expected_unscaled = k.detach().cpu().numpy()
        predvar = out.detach().cpu().numpy()
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
        data_loss_epoch /= len(dataloader)
        physics_loss_epoch /= len(dataloader)
        weighted_physics_loss = (physics_weight / residual_normalizer) * physics_loss_epoch
        return {
            "validation_data_loss": data_loss_epoch,
            "validation_physics_loss": physics_loss_epoch,
            "validation_weighted_physics_loss": weighted_physics_loss,
            "validation_total_loss": data_loss_epoch + weighted_physics_loss,
        }


@hydra.main(version_base="1.3", config_path="conf", config_name="config_fno.yaml")
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
    residual_normalizer = cfg.physics_residual_normalizer
    resolution = cfg.data.resolution
    
    # Use Diffusion equation for the Darcy PDE
    forcing_fn = cfg.physics_forcing_term
    darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)

    dataset = CustomDataset(
        to_absolute_path(cfg.data.train_path),
        mappings={
            "permeability": "Kcoeff",
            "darcy": "sol"
        },
        device=device
    )
    validation_dataset = CustomDataset(
        to_absolute_path(cfg.data.validation_path),
        mappings={
            "permeability": "Kcoeff",
            "darcy": "sol"
        },
        device=device
    )

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False)

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=cfg.start_lr,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
    
    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    for epoch in range(loaded_epoch + 1, cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            for data in dataloader:
                optimizer.zero_grad()
                k = data["permeability"]
                u = data["darcy"]
                u_scaled = u / darcy_scale
                k_scaled = k / permeability_scale
                
                if epoch == 0: print(u_scaled.shape, k_scaled.shape)

                # Compute forward pass
                out_scaled = model(u_scaled)
                k_pred = out_scaled * permeability_scale
                
                assert out_scaled.shape == k_scaled.shape, f"Output shape {out_scaled.shape} does not match target shape {k_scaled.shape}"
                
                # Compute data loss
                loss_data = F.mse_loss(k_scaled, out_scaled)

                # Compute total loss
                loss = loss_data #+ weighted_pde

                # Backward pass and optimizer and learning rate update
                loss.backward()
                optimizer.step()
                log.log_minibatch(
                    {
                        "loss_data": loss_data.detach().item(),
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
                cfg.physics_weight,
                residual_normalizer,
            )
            log.log_epoch(validation_metrics)

        save_checkpoint(
            "./checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=cast(Any, scheduler),
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
