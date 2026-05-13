# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
                print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")
                if k in mappings.values():
                    self.keys.append(k)

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device
        self.res = res if res is not None else 240 # self.__len__()
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
