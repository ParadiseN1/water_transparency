from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import xarray as xr
import torch
import random
from PIL import Image

d2r = np.pi / 180

AVG_NAN_COUNT = 33659

#  python SimVPv2/tools/non_dist_train.py -d ZsdChl --lr 1e-3 -c ./configs/zsd/SimVP.py --ex_name ZsdChl

def latlon2xyz(lat, lon):
    if type(lat) == torch.Tensor:
        x = -torch.cos(lat)*torch.cos(lon)
        y = -torch.cos(lat)*torch.sin(lon)
        z = torch.sin(lat)

    if type(lat) == np.ndarray:
        x = -np.cos(lat)*np.cos(lon)
        y = -np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
    return x, y, z


def xyz2latlon(x, y, z):
    if type(x) == torch.Tensor:
        lat = torch.arcsin(z)
        lon = torch.atan2(-y, -x)

    if type(x) == np.ndarray:
        lat = np.arcsin(z)
        lon = np.arctan2(-y, -x)
    return lat, lon

def drop_last_column_if_odd(data):
    lat_size = data.lat.size
    lon_size = data.lon.size

    lat_slice = slice(None, -1) if lat_size % 2 != 0 else slice(None)
    lon_slice = slice(None, -1) if lon_size % 2 != 0 else slice(None)

    return data.isel(lat=lat_slice, lon=lon_slice)

class SimpleZsdDataset(Dataset):
    def __init__(self, data_dir, training_time, seq_len=10, out_seq_len=1, surface_mask=False, mean=None, std=None):
        super().__init__()
        self.y_start = training_time[0]
        self.y_end = training_time[1]
        self.seq_len = 10
        self.out_seq_len = out_seq_len
        self.time = None
        self.threshold = AVG_NAN_COUNT * 1.1 # empirical value, when all datapoints with NaNs due to sun illumination are above this threshold
        self.patch_h = 32
        self.patch_w = 64
        self.mean = mean
        self.std = std

        try:
            dataset = xr.open_mfdataset(
                data_dir+'/*.nc', combine='by_coords')
        except AttributeError:
            assert False and 'Please install the latest xarray, e.g.,' \
                                'pip install  git+https://github.com/pydata/xarray/@v2022.03.0'
        dataset = drop_last_column_if_odd(dataset)
        dataset = dataset.sel(time=slice(self.y_start, self.y_end))
        
        self.num_patches_per_image = len(dataset.lat.values)//self.patch_h * len(dataset.lon.values)//self.patch_w
        
        ## Filter datapoints with NaNs
        missing_values_count = dataset['ZSD'].isnull().sum(dim=['lon', 'lat'])
        timestamps_below_threshold = (missing_values_count < self.threshold)
        print(f'Filtered {timestamps_below_threshold.values.sum()}/{len(timestamps_below_threshold.values)}')
        dataset = dataset.where(timestamps_below_threshold, drop=True)


        self.data = dataset.get('ZSD').values[:, np.newaxis, :self.patch_h, :self.patch_w]
        if surface_mask:
            self.surface_mask = np.array(Image.open(data_dir+"/surface_mask.png"))[...,0] / 255.
            self.data = self.data*self.surface_mask
            
        if self.mean is None:
            self.mean = self.data[self.data>0].mean()
            self.std = self.data[self.data>0].std()
        self.data = (self.data-self.mean)/self.std
        # if self.min == None:
        #     self.min = self.data[~np.isnan(self.data)].min()
        # self.data = self.data - self.min
        # if self.max == None:
        #     self.max = self.data[~np.isnan(self.data)].max()
        # self.data = self.data / self.max
        self.data = np.nan_to_num(self.data, nan=-777)

        self.valid_idx = np.array(
            range(0, (self.data.shape[0]- out_seq_len - self.seq_len)))
    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        idx = (self.valid_idx[index])
        # get random patch from image
        lon_i = int(random.uniform(0, self.data.shape[-2]-self.patch_h))
        lat_i = int(random.uniform(0, self.data.shape[-1]-self.patch_w))

        x_data = torch.tensor(self.data[idx:idx+self.seq_len, :, :, :])
        y_data = torch.tensor(self.data[idx+self.seq_len:idx+self.seq_len+self.out_seq_len, :, :, :])
        return x_data, y_data

def load_data(batch_size,
              val_batch_size,
              data_dir,
              num_workers=4,
              train_time=['1997', '2020'],
              val_time=['2020', '2021'],
              test_time=['2021', '2023'],
              **kwargs):
    train_set = SimpleZsdDataset(data_dir=data_dir, training_time=train_time)
    validation_set = SimpleZsdDataset(data_dir, val_time, mean=train_set.mean, std=train_set.std)
    test_set = SimpleZsdDataset(data_dir, test_time, mean=train_set.mean, std=train_set.std)

    dataloader_train = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size, shuffle=True,
                                                   pin_memory=True, drop_last=True,
                                                   num_workers=num_workers)
    dataloader_vali = torch.utils.data.DataLoader(validation_set, # validation_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(test_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers)

    return dataloader_train, dataloader_vali, dataloader_test