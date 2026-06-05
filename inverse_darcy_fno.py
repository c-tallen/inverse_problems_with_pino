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

def set_seed(seed: int):
    print(f"Setting random seed to: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional, makes things more deterministic but may slow training
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@hydra.main(version_base="1.3", config_path="conf", config_name="neural_operator_noisy.yaml")
def main(cfg: DictConfig):
    """Main function for the Darcy physics-informed FNO."""
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    set_seed(cfg.seed)

    LaunchLogger.initialize()
    DistributedManager.initialize()

    permeability_min = cfg.scaling.permeability_min if cfg.data.pde_bench else 3.0
    permeability_max = cfg.scaling.permeability_max if cfg.data.pde_bench else 12.0
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
    print(f"Training dataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    print("Using the batch size specified in the config:", cfg.batch_size)
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
            epoch_alert_freq=len(dataloader) // 20,
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
                    u_noisy = u + noise
                    u_input = u_noisy / darcy_scale
                else:
                    u_input = u / darcy_scale

                # Inverse model: u -> k
                out_raw = model(u_input)
                # Original code constrains predicted permeability to [3, 12]
                k_pred = darcy_mask1(out_raw, permeability_min=permeability_min, permeability_max=permeability_max)

                assert k_pred.shape == k.shape, (
                    f"Output shape {k_pred.shape} does not match target shape {k.shape}"
                )

                # Data loss: predicted permeability vs true permeability
                loss_data = F.mse_loss(k_pred, k) # TODO: This can be L2 or relative L2

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
                    # Total variation regularization on raw model output, like original code
                    loss_tv = total_variance(k_pred)
                    # Original pretraining loss:
                    # pino_loss = 0.2 * loss_f + loss_data + 0.01 * loss_TV
                    weighted_pde = cfg.physics_weight * loss_pde
                    weighted_tv = cfg.tv_weight * loss_tv
                    if cfg.physics_only:
                        loss = weighted_pde + weighted_tv
                    else:
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
                    }
                )
            scheduler.step()
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            
        with LaunchLogger("valid", epoch=epoch) as log:
            validation_metrics = validation_step(
                model,
                validation_dataloader,
                epoch,
                darcy_scale,
                permeability_min=permeability_min,
                permeability_max=permeability_max
            )
            log.log_epoch({"Validation error": validation_metrics})
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
