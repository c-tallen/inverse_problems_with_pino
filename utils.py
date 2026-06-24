# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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

import os
import zipfile

from matplotlib import pyplot as plt

try:
    import gdown
except:
    gdown = None

from typing import Union

import h5py
import numpy as np
import scipy.io
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset
from omegaconf import DictConfig, OmegaConf
import pathlib


# list of FNO dataset url ids on drive: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-
_FNO_datatsets_ids = {
    "Darcy_241": "1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV",
    "Darcy_421": "1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf",
}
_FNO_dataset_names = {
    "Darcy_241": (
        "piececonst_r241_N1024_smooth1.hdf5",
        "piececonst_r241_N1024_smooth2.hdf5",
    ),
    "Darcy_421": (
        "piececonst_r421_N1024_smooth1.hdf5",
        "piececonst_r421_N1024_smooth2.hdf5",
    ),
}


def load_FNO_dataset(path, input_keys, output_keys, n_examples=None):
    "Loads a FNO dataset"

    if not path.endswith(".hdf5"):
        raise Exception(
            ".hdf5 file required: please use utilities.preprocess_FNO_mat to convert .mat file"
        )

    # load data
    path = to_absolute_path(path)
    data = h5py.File(path, "r")
    _ks = [k for k in data.keys() if not k.startswith("__")]
    print(f"loaded: {path}\navaliable keys: {_ks}")

    # parse data
    invar, outvar = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar, output_keys)]:
        for k in keys:
            # get data
            x = data[k]  # N, C, H, W

            # cut examples out
            if n_examples is not None:
                x = x[:n_examples]

            # print out normalisation values
            print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")

            d[k] = x
    del data

    return (invar, outvar)


def download_FNO_dataset(name, outdir="datasets/"):
    "Tries to download FNO dataset from drive"

    if name not in _FNO_datatsets_ids:
        raise Exception(
            f"Error: FNO dataset {name} not recognised, select one from {list(_FNO_datatsets_ids.keys())}"
        )

    id = _FNO_datatsets_ids[name]
    outdir = to_absolute_path(outdir) + "/"
    namedir = f"{outdir}{name}/"

    # skip if already exists
    exists = True
    for file_name in _FNO_dataset_names[name]:
        if not os.path.isfile(namedir + file_name):
            exists = False
            break
    if exists:
        return
    print(f"FNO dataset {name} not detected, downloading dataset")

    # Make sure we have gdown installed
    if gdown is None:
        raise ModuleNotFoundError("gdown package is required to download the dataset!")

    # get output directory
    os.makedirs(namedir, exist_ok=True)

    # download dataset
    zippath = f"{outdir}{name}.zip"
    _download_file_from_google_drive(id, zippath)

    # unzip
    with zipfile.ZipFile(zippath, "r") as f:
        f.extractall(namedir)
    os.remove(zippath)

    # preprocess files
    for file in os.listdir(namedir):
        if file.endswith(".mat"):
            matpath = f"{namedir}{file}"
            preprocess_FNO_mat(matpath)
            os.remove(matpath)


def _download_file_from_google_drive(id, path):
    "Downloads a file from google drive"

    # use gdown library to download file
    gdown.download(id=id, output=path)


def preprocess_FNO_mat(path):
    "Convert a FNO .mat file to a hdf5 file, adding extra dimension to data arrays"

    assert path.endswith(".mat")
    data = scipy.io.loadmat(path)
    ks = [k for k in data.keys() if not k.startswith("__")]
    with h5py.File(path[:-4] + ".hdf5", "w") as f:
        for k in ks:
            x = np.expand_dims(data[k], axis=1)  # N, C, H, W
            f.create_dataset(
                k, data=x, dtype="float32"
            )  # note h5 files larger than .mat because no compression used


class HDF5MapStyleDataset(Dataset):
    """Simple map-style HDF5 dataset"""

    def __init__(
        self,
        file_path,
        device: Union[str, torch.device] = "cuda",
    ):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.keys = list(f.keys())

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index == None:
            device = torch.device("cuda:0")
        self.device = device

    def __len__(self):
        with h5py.File(self.file_path, "r") as f:
            return len(f[self.keys[0]])

    def __getitem__(self, idx):
        data = {}
        with h5py.File(self.file_path, "r") as f:
            for key in self.keys:
                data[key] = np.array(f[key][idx])

        invar = torch.cat(
            [
                torch.from_numpy((data["Kcoeff"][:, :240, :240]) / 4.49996e00),
                torch.from_numpy(data["Kcoeff_x"][:, :240, :240]) / 4.49996e00,
                torch.from_numpy(data["Kcoeff_y"][:, :240, :240]) / 4.49996e00,
            ]
        )
        outvar = torch.from_numpy((data["sol"][:, :240, :240]) / 3.88433e-03)

        x = np.linspace(0, 1, 240)
        y = np.linspace(0, 1, 240)

        xx, yy = np.meshgrid(x, y)
        x_invar = torch.from_numpy(xx.astype(np.float32)).view(
            1, 240, 240
        )  # add channel dimension
        y_invar = torch.from_numpy(yy.astype(np.float32)).view(
            1, 240, 240
        )  # add channel dimension

        if self.device.type == "cuda":
            # Move tensors to GPU
            invar = invar.cuda()
            outvar = outvar.cuda()
            x_invar = x_invar.cuda()
            y_invar = y_invar.cuda()

        return invar, outvar, x_invar, y_invar

class CustomDataset(Dataset):
    """Simple map-style HDF5 dataset"""

    def __init__(
            self,
            file_path,
            mappings,
            device: Union[str, torch.device] = "cuda",
            res=None,
            noise=None,
    ):
        self.mappings = mappings
        self.file_path = file_path
        self.keys = []
        with h5py.File(file_path, "r") as f:
            for k in f.keys():
                x = np.array(f[k])
                print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}, max: {x.max():.5e}, min: {x.min():.5e}")
                if k in mappings.values():
                    self.keys.append(k)

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device
        assert res is not None, "Resolution must be specified for custom dataset"
        self.res = res if res is not None else self.__len__()
        self.noise = noise
        
    def __len__(self):
        with h5py.File(self.file_path, "r") as f:
            return len(f[self.keys[0]])

    def __getitem__(self, idx):
        res = self.res
        data = {}
        with h5py.File(self.file_path, "r") as f:
            for key in self.keys:
                data[key] = np.array(f[key][idx])

        output = {}

        # Add channel dimension if needed
        for key, value in self.mappings.items():
            if data[value].ndim == 2:
                data[value] = data[value][np.newaxis, :, :]
            output[key] = torch.from_numpy((data[value][:, :res, :res]))

        if self.device.type == "cuda":
            # Move tensors to GPU
            for key, tensor in output.items():
                output[key] = tensor.cuda()

        return output


def darcy_mask1(x: torch.Tensor, permeability_min=3.0, permeability_max=12.0) -> torch.Tensor:
    """Map raw network output to permeability range [permeability_min, permeability_max]."""
    return torch.sigmoid(x) * (permeability_max - permeability_min) + permeability_min

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

def validation_step(model, dataloader, epoch, darcy_scale, permeability_min=3.0, permeability_max=12.0):
    """Validation Step"""
    model.eval()

    with torch.no_grad():
        data_loss_epoch = 0.0
        for data in dataloader:
            k = data["permeability"] 
            u = data["darcy"]
            u_scaled = u / darcy_scale
            out_raw = model(u_scaled)
            k_pred = darcy_mask1(out_raw, permeability_min=permeability_min, permeability_max=permeability_max)
            data_loss_epoch += relative_l2_loss(k_pred, k).item()

        # convert data to numpy
        expected_unscaled = k.detach().cpu().numpy()
        predvar = k_pred.detach().cpu().numpy()

        # plotting
        fig, ax = plt.subplots(1, 3, figsize=(25, 5))

        def plot_with_colorbar(i, data, title):
            d_min = np.min(data[0, 0])
            d_max = np.max(data[0, 0])
            im = ax[i].imshow(data[0, 0], vmin=d_min, vmax=d_max)
            plt.colorbar(im, ax=ax[i])
            ax[i].set_title(title)
        
        plot_with_colorbar(0, expected_unscaled, "True")
        plot_with_colorbar(1, predvar, "Pred")
        plot_with_colorbar(2, np.abs(predvar - expected_unscaled), "Difference")

        fig.savefig(f"results_{epoch}.png")
        plt.close()
        return data_loss_epoch / len(dataloader)
    
    
def corr_indicator(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)

    numerator = torch.sum(pred_centered * target_centered, dim=1)
    denominator = (
        torch.linalg.norm(pred_centered, dim=1)
        * torch.linalg.norm(target_centered, dim=1)
    )

    return torch.mean(numerator / (denominator + eps))


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

def make_sparse_input(
    u: torch.Tensor,
    sensor_density: float,
    darcy_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sparse model input [M * u, M].

    Returns:
        model_input: Tensor of shape (B, 2, H, W)
        mask: Tensor of shape (B, 1, H, W)
        u_masked: Tensor of shape (B, 1, H, W)
    """
    mask = make_random_mask(u, sensor_density)
    u_masked = mask * u
    u_masked_scaled = u_masked / darcy_scale
    model_input = torch.cat([u_masked_scaled, mask], dim=1)

    return model_input, mask, u_masked

def get_pde_loss(phy_informer, u, pred_i):
    residuals = phy_informer.forward(
                        {
                            "u": u,
                            "k": pred_i,
                        }
                    )

    pde_out_arr = residuals["diffusion_u"]
    pde_core = pde_out_arr[:, :, 2:-2, 2:-2]
    pde_loss = torch.mean(torch.abs(pde_core)).item()
    return pde_loss

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