import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from aurora import AuroraSmall
from aurora.normalisation import locations, scales
import xarray as xr
from aurora import  Batch, Metadata, rollout
from aurora.normalisation import locations, scales
import torch
import numpy as np
import torch.optim as optim
from scipy.ndimage import zoom
import logging
from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import sys
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
device_ids = [ 1,2,3]
model =AuroraSmall(
    use_lora=False,
    autocast=True,
    surf_vars=(
        "aco2rec",
        "aco2gpp",
        "aco2nee",
        "fco2rec",
        "fco2gpp",
        "fco2nee",),
    static_vars=("z","lsm"),
    atmos_vars=("t"),
).cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.000001)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)
# model.load_checkpoint("microsoft/aurora", r"C:\Users\DC\Documents\vscode4\aurora_co2_2\aurora-0.25-finetuned.ckpt", strict=False)
# File paths
co2_data_sfc = r"C:\Users\DC\Documents\vscode4\7bc2ad758dd66a384b66b64346e658f3\data_sfc.nc"
download_path = r'C:\Users\DC\Documents\vscode4\ubuntu_vs\vscode\VScode2\vscode\down'

# Load datasets
co2_data_sfc = xr.open_dataset(co2_data_sfc, engine="netcdf4")
static_vars_ds = xr.open_dataset(f"{download_path}/static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(f"{download_path}/2023-01-01-surface-level.nc", engine="netcdf4")

atmos_vars_ds = xr.open_dataset(f"{download_path}/2023-01-01-atmospheric.nc", engine="netcdf4")
loss_fn1 = torch.nn.L1Loss(reduction='none')
'''
Author: DC
Date: 2024-11-06 21:28:56
LastEditTime: 2024-11-07 11:01:40
LastEditors: DC
Description: 
FilePath: \aurora_co2_2\finetune3.ipynb
Never lose my passion
'''


def normalize_2d(data, method='manual'):
    """
    Normalize 2D data in PyTorch.

    Args:
        data (torch.Tensor): The 2D tensor to normalize.
        method (str): The normalization method to use ('manual' or 'functional').

    Returns:
        torch.Tensor: The normalized 2D tensor.
    """
    if method == 'functional':
        # Using torch.nn.functional.normalize
        normalized_data = F.normalize(data, p=2, dim=0)  # Normalize along columns
    elif method == 'manual':
        # Manual normalization
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        normalized_data = (data - mean) / std
    else:
        raise ValueError("Method must be 'manual' or 'functional'")

    return normalized_data
snapshot_path=r'C:\Users\DC\Documents\vscode4\aurora_co2_3\snapshot_path'
logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
# logging.info(str(args))
iterator = tqdm(range(1000), ncols=70)
writer = SummaryWriter(snapshot_path + '/log')
iter_num = 0
for epoch in iterator:
    torch.save(model.state_dict(), snapshot_path+'/model.pth')
    for i in range(0, 22):
        for j in range(1, 41,2):
            var_names = ['aco2rec', 'aco2gpp', 'aco2nee', 'fco2rec', 'fco2gpp', 'fco2nee']
            var_dict = {}
            for var in var_names:
                arr = zoom(co2_data_sfc[var][:, i, :, :][[j - 1, j]],
                           (1, 720 / co2_data_sfc[var].shape[2], 1440 / co2_data_sfc[var].shape[3]), order=1)
                var_dict[var] = arr[None]

            arr_aco2rec = var_dict['aco2rec']
            arr_aco2gpp = var_dict['aco2gpp']
            arr_aco2nee = var_dict['aco2nee']
            arr_fco2rec = var_dict['fco2rec']
            arr_fco2gpp = var_dict['fco2gpp']
            arr_fco2nee = var_dict['fco2nee']

            arr_np = np.ones((1, 2, 13, 720, 1440))
            arr_np2 = np.ones((720, 1440))

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
                    "z": torch.from_numpy(arr_np2),
                    "lsm": torch.from_numpy(arr_np2),
                },
                atmos_vars={
                    "t": torch.from_numpy(arr_np),
                },
                metadata=Metadata(
                    lat=torch.from_numpy(atmos_vars_ds.latitude[:720].values),
                    lon=torch.from_numpy(atmos_vars_ds.longitude[:1440].values),
                    time=(co2_data_sfc['valid_time'].values.astype("datetime64[s]").tolist()[0][i],),
                    atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
                ),
            )
            batch = batch.to('cuda')

            loss = 0
            if j != 1:
                # former_prerds=former_prerds.to('cpu')

                for key in preds[0].keys():

                    pred1 = F.normalize(former_prerds[0][key].squeeze(), p=2, dim=0).to('cuda')

                    true1 = F.normalize(torch.from_numpy(var_dict[key][:, 0, :, :]).squeeze().squeeze(), p=2, dim=0).to(
                        'cuda')
                    pred2 = F.normalize(former_prerds[1][key].squeeze(), p=2, dim=0).to('cuda')
                    true2 = F.normalize(torch.from_numpy(var_dict[key][:, 1, :, :]).squeeze().squeeze(), p=2, dim=0).to(
                        'cuda')

                    # print(pred1.shape)
                    # print(true1.shape)
                    # print(pred2.shape)
                    # print(true2.shape)
                    # print("111111111111111")


                    # print(pred1.device)
                    # print(true1.device)
                    # print(pred1.shape)
                    # print(true1.shape)

                    loss_1 = loss_fn1(pred1, true1).sum()
                    # print(loss_1)
                    loss_2= loss_fn1(pred2, true2).sum()
                    loss=loss_1+loss_2
                    loss=loss.to('cuda')
                logging.info('train loss:  ' + str(loss.sum().item()))
                # print("loss")
                # print(loss.sum().item())
                # print(loss.shape)
                # print()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for param_group in optimizer.param_groups:
                    lr_ = param_group['lr']
                if lr_ == None:
                    lr_=0.0

                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                writer.add_scalar('info/loss_1', loss_1, iter_num)
                writer.add_scalar('info/loss_2', loss_2, iter_num)
                if iter_num % 1 == 0:
                    # print(pred1.shape)
                    # Ensure the tensor has the shape (C, H, W)
                    image1 = pred1.clone().to('cpu').detach().numpy()
                    image1 = image1.reshape(1, image1.shape[0], image1.shape[1])  # Reshape to (C, H, W)
                    writer.add_image('train/time_1', image1 * 50, iter_num)

                    image2 = pred2.clone().to('cpu').detach().numpy()
                    image2 = image2.reshape(1, image2.shape[0], image2.shape[1])  # Reshape to (C, H, W)
                    writer.add_image('train/time_2', image2 * 50, iter_num)

                    outputs1 = true1.clone().to('cpu').detach().numpy()
                    outputs1 = outputs1.reshape(1, outputs1.shape[0], outputs1.shape[1])  # Reshape to (C, H, W)
                    writer.add_image('train/true1', outputs1 * 50, iter_num)

                    outputs2 = true2.clone().to('cpu').detach().numpy()
                    outputs2 = outputs2.reshape(1, outputs2.shape[0], outputs2.shape[1])  # Reshape to (C, H, W)
                    writer.add_image('train/true2', outputs2 * 50, iter_num)

                # del former_prerds,loss,pred1,true1,pred2,true2
                # torch.cuda.empty_cache()
                # batch.to('cuda')

            model = model.cuda()
            model.train()

            preds = [pred for pred in rollout(model, batch, steps=2)]
            former_prerds = preds
            iter_num +=1
writer.close()

            # print()
