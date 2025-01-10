'''
Author: DC
Date: 2024-11-06 10:36:33
LastEditTime: 2024-11-06 17:04:29
LastEditors: DC
Description: 
FilePath: \aurora_co2_2\finetune1.py
Never lose my passion
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from aurora import AuroraSmall
from aurora.normalisation import locations, scales
import xarray as xr
from aurora import  Batch, Metadata, rollout
from aurora.normalisation import locations, scales
import torch
import numpy as np
from scipy.ndimage import zoom
locations["aco2rec"] = 0.0
scales["aco2rec"] = 1.0
locations["aco2gpp"] = 0.0
scales["aco2gpp"] = 1.0
locations["aco2nee"] = 0.0
scales["aco2nee"] = 1.0
locations["fco2rec"] = 0.0
scales["fco2rec"] = 1.0
locations["fco2gpp"] = 0.0
scales["fco2gpp"] = 1.0
locations["fco2nee"] = 0.0
scales["fco2nee"] = 1.0
locations["a1_1000"] = 0.0
scales["a1_1000"] = 1.0
# Load the Aurora model
model = AuroraSmall(
    use_lora=False,
    surf_vars=(
        "aco2rec",
        "aco2gpp",
        "aco2nee",
        "fco2rec",
        "fco2gpp",
        "fco2nee",),
    static_vars=("lsm", "z", "slt"),
    atmos_vars=("t"),
).cuda()
# model.load_checkpoint("microsoft/aurora", r"C:\Users\DC\Documents\vscode4\aurora_co2_2\aurora-0.25-finetuned.ckpt", strict=False)
# File paths
co2_data_sfc = r"C:\Users\DC\Documents\vscode4\7bc2ad758dd66a384b66b64346e658f3\data_sfc.nc"
download_path = r'C:\Users\DC\Documents\vscode4\ubuntu_vs\vscode\VScode2\vscode\down'

# Load datasets
co2_data_sfc = xr.open_dataset(co2_data_sfc, engine="netcdf4")
static_vars_ds = xr.open_dataset(f"{download_path}/static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(f"{download_path}/2023-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(f"{download_path}/2023-01-01-atmospheric.nc", engine="netcdf4")
for i in range(1,11):
    None
    arr_aco2rec=zoom(co2_data_sfc['aco2rec'][:,0,:,:][[0,1]], (1,  721 / co2_data_sfc['aco2rec'].shape[2], 1440 / co2_data_sfc['aco2rec'].shape[3]), order=1)
    arr_aco2rec=arr_aco2rec[None]
    arr_aco2gpp=zoom(co2_data_sfc['aco2gpp'][:,0,:,:][[0,1]], (1,  721 / co2_data_sfc['aco2rec'].shape[2], 1440 / co2_data_sfc['aco2rec'].shape[3]), order=1)
    arr_aco2gpp=arr_aco2gpp[None]
    arr_aco2nee=zoom(co2_data_sfc['aco2nee'][:,0,:,:][[0,1]], (1,  721 / co2_data_sfc['aco2rec'].shape[2], 1440 / co2_data_sfc['aco2rec'].shape[3]), order=1)
    arr_aco2nee=arr_aco2nee[None]
    arr_fco2rec=zoom(co2_data_sfc['fco2rec'][:,0,:,:][[0,1]], (1,  721 / co2_data_sfc['aco2rec'].shape[2], 1440 / co2_data_sfc['aco2rec'].shape[3]), order=1)
    arr_fco2rec=arr_fco2rec[None]
    arr_fco2gpp=zoom(co2_data_sfc['fco2gpp'][:,0,:,:][[0,1]], (1,  721 / co2_data_sfc['aco2rec'].shape[2], 1440 / co2_data_sfc['aco2rec'].shape[3]), order=1)
    arr_fco2gpp=arr_fco2gpp[None]
    arr_fco2nee=zoom(co2_data_sfc['fco2nee'][:,0,:,:][[0,1]], (1,  721 / co2_data_sfc['aco2rec'].shape[2], 1440 / co2_data_sfc['aco2rec'].shape[3]), order=1)
    arr_fco2nee=arr_fco2nee[None]

    # arr_aco2gpp=co2_data_sfc['aco2gpp'][:,0,:,:][[i-1,i]].values
    # arr_aco2gpp=arr_aco2gpp[None]
    # arr_aco2nee=co2_data_sfc['aco2nee'][:,0,:,:][[i-1,i]].values
    # arr_aco2nee=arr_aco2nee[None]
    # arr_fco2rec=co2_data_sfc['fco2rec'][:,0,:,:][[i-1,i]].values
    # arr_fco2rec=arr_fco2rec[None]
    # arr_fco2gpp=co2_data_sfc['fco2gpp'][:,0,:,:][[i-1,i]].values
    # arr_fco2gpp=arr_fco2gpp[None]
    # arr_fco2nee=co2_data_sfc['fco2nee'][:,0,:,:][[i-1,i]].values
    # arr_fco2nee=arr_fco2nee[None]
    arr_np=np.ones((1, 2, 13, 721, 1440))
    
    batch = Batch(
    surf_vars={
        "aco2rec": torch.from_numpy(arr_aco2rec),
        "aco2gpp": torch.from_numpy(arr_aco2gpp),
        "aco2nee": torch.from_numpy(arr_aco2nee),
        "fco2rec": torch.from_numpy(arr_fco2rec),
        "fco2gpp": torch.from_numpy(arr_fco2gpp),
        "fco2nee": torch.from_numpy(arr_fco2nee),
        
    },
    static_vars={
        # "z": torch.from_numpy(zoom(static_vars_ds["z"].values[0], (1801 / static_vars_ds["z"].shape[1], 3600 / static_vars_ds["z"].shape[2]), order=1)),
        # "slt": torch.from_numpy(zoom(static_vars_ds["slt"].values[0], (1801 / static_vars_ds["z"].shape[1], 3600 / static_vars_ds["z"].shape[2]), order=1)),
        # "lsm": torch.from_numpy(zoom(static_vars_ds["lsm"].values[0], (1801 / static_vars_ds["z"].shape[1], 3600 / static_vars_ds["z"].shape[2]), order=1)),
        "z": torch.from_numpy(static_vars_ds["z"].values[0]),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
    },
    atmos_vars={
        #  "t": torch.from_numpy(zoom(atmos_vars_ds["t"].values[[0, 1]][None], (1, 1,1, 1801 / atmos_vars_ds["t"].shape[2], 3600 / atmos_vars_ds["t"].shape[3]), order=1)),
        # "u": torch.from_numpy(zoom(atmos_vars_ds["u"].values[[0, 1]][None], (1, 1,1, 1801 / atmos_vars_ds["t"].shape[2], 3600 / atmos_vars_ds["t"].shape[3]), order=1)),
        # "v": torch.from_numpy(zoom(atmos_vars_ds["v"].values[[0, 1]][None], (1, 1,1, 1801 / atmos_vars_ds["t"].shape[2], 3600 / atmos_vars_ds["t"].shape[3]), order=1)),
        # "q": torch.from_numpy(zoom(atmos_vars_ds["q"].values[[0, 1]][None], (1, 1,1, 1801 / atmos_vars_ds["t"].shape[2], 3600 / atmos_vars_ds["t"].shape[3]), order=1)),
        # "z": torch.from_numpy(zoom(atmos_vars_ds["z"].values[[0, 1]][None], (1, 1,1, 1801 / atmos_vars_ds["t"].shape[2], 3600 / atmos_vars_ds["t"].shape[3]), order=1)),
        "t": torch.from_numpy(arr_np),
        # "u": torch.from_numpy(arr_np),
        # "v": torch.from_numpy(arr_np),
        # "q": torch.from_numpy(arr_np),
        # "z": torch.from_numpy(arr_np),
    },
    metadata=Metadata(
        lat=torch.from_numpy(atmos_vars_ds.latitude.values),
        lon=torch.from_numpy(atmos_vars_ds.longitude.values),
        time=(co2_data_sfc.valid_time.values.astype("datetime64[s]").tolist()[0][i],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)
    model = model.cuda()
    model.train()
    pred = model.forward(batch)
    # print(pred)
    print()