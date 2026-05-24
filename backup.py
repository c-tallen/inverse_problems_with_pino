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
from hydra.utils import to_absolute_path
from physicsnemo.utils.logging import LaunchLogger
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from diffusion_eq import Diffusion
from utils import CustomDataset


def validation_step(model, dataloader, epoch):
    """Validation Step"""
    model.eval()

    with torch.no_grad():
        loss_epoch = 0
        for data in dataloader:
            k = data["permeability"] 
            k_scaled = k / 4.49996e00
            u = data["darcy"]
            u_scaled = u / 3.88433e-03
            out = model(u_scaled)

            loss_epoch += F.mse_loss(k_scaled, out)

        # convert data to numpy
        expected_unscaled = k.detach().cpu().numpy()
        expected = k_scaled.detach().cpu().numpy()
        predvar = out.detach().cpu().numpy()
        predvar_unscaled = predvar * 4.49996e00

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
        return loss_epoch / len(dataloader)


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
    
    # Use Diffusion equation for the Darcy PDE
    forcing_fn = 1.0
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

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

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
    
    phy_informer = PhysicsInformer(
        required_outputs=["diffusion_u"],
        equations=darcy,
        grad_method="finite_difference",
        device=device,
        fd_dx=1 / 240,  # Unit square with resoultion as 240
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=cfg.start_lr,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
    loaded_epoch = 0
    print(f"Checkpoint directory: {to_absolute_path('./checkpoints')}")
    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    for epoch in range(loaded_epoch, loaded_epoch + cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            print("Dataloader length:", len(dataloader))
            for data in dataloader:
                optimizer.zero_grad()
                k = data["permeability"]
                u = data["darcy"]
                # TODO: Deal with scaling
                u_scaled = u / 3.88433e-03
                k_scaled = k / 4.49996e00
                
                if epoch == 0: print(u_scaled.shape, k_scaled.shape)

                # Compute forward pass
                out_scaled = model(u_scaled)
                k_pred = out_scaled * 4.49996e00
                
                # Compute data loss
                loss_data = F.mse_loss(k_scaled, out_scaled)
                
                assert out_scaled.shape == k_scaled.shape, f"Output shape {out_scaled.shape} does not match target shape {k_scaled.shape}"
                if cfg.physics_weight > 0.0:
                    residuals = phy_informer.forward(
                        {
                            "u": u,
                            "k": k_pred,
                        }
                    )
                    pde_out_arr = residuals["diffusion_u"]

                    pde_out_arr = F.pad(
                        pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                    )
                    loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))
                    weighted_pde = (cfg.physics_weight / 240) * loss_pde
                    
                    # Compute total loss
                    loss = loss_data + weighted_pde
                else:
                    loss = loss_data

                # Backward pass and optimizer and learning rate update
                loss.backward()
                optimizer.step()
                if cfg.physics_weight > 0.0:
                    log.log_minibatch(
                        {"loss_data": loss_data.detach().item(), "loss_pde": loss_pde.detach().item(), "weighted_pde": weighted_pde.detach().item(), "loss_total": loss.detach().item()}
                    )
                else:
                    log.log_minibatch({"loss_data": loss.detach().item()})
            scheduler.step()
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            
        with LaunchLogger("valid", epoch=epoch) as log:
            error = validation_step(model, validation_dataloader, epoch)
            log.log_epoch({"Validation error": error})

        save_checkpoint(
            "./checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
