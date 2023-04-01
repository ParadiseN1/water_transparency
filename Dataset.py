from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import xarray as xr
import torch

d2r = np.pi / 180

AVG_NAN_COUNT = 33659

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

class ZsdDataset(Dataset):
    def __init__(self, data_dir, training_time, seq_len=10):
        super().__init__()
        self.y_start = training_time[0]
        self.y_end = training_time[1]
        self.seq_len = 10
        self.time = None
        self.threshold = AVG_NAN_COUNT * 1.1 # empirical value, when all datapoints with NaNs due to sun illumination are above this threshold

        try:
            dataset = xr.open_mfdataset(
                data_dir+'/*.nc', combine='by_coords')
        except AttributeError:
            assert False and 'Please install the latest xarray, e.g.,' \
                                'pip install  git+https://github.com/pydata/xarray/@v2022.03.0'
        dataset = dataset.sel(time=slice(self.y_start, self.y_end))
        
        ## Filter datapoints with NaNs
        missing_values_count = dataset['ZSD'].isnull().sum(dim=['lon', 'lat'])
        timestamps_below_threshold = (missing_values_count < self.threshold)
        print(f'Filtered {timestamps_below_threshold.values.sum()}/{len(timestamps_below_threshold.values)}')
        dataset = dataset.where(timestamps_below_threshold, drop=True)


        self.data = dataset.get('ZSD').values[:, np.newaxis, :, :]

        self.min = self.data[~np.isnan(self.data)].min()
        self.data = self.data - self.min
        self.max = self.data[~np.isnan(self.data)].max()
        self.data = self.data / self.max
        self.data = np.nan_to_num(self.data, nan=-1)
        # self.mean = self.data.mean(axis=(0, 2, 3)).reshape(
        #     1, self.data.shape[1], 1, 1)
        # self.std = self.data.std(axis=(0, 2, 3)).reshape(
        #     1, self.data.shape[1], 1, 1)

        # self.data = (self.data-self.mean)/self.std
        self.valid_idx = np.array(
            range(0, self.data.shape[0]//seq_len-1))
    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        idx = self.valid_idx[index] * self.seq_len
        x_data = torch.tensor(self.data[idx:idx+self.seq_len])
        y_data = torch.tensor(self.data[idx+self.seq_len:idx+self.seq_len*2])
        return x_data, y_data

def load_data(batch_size,
              val_batch_size,
              data_dir,
              num_workers=4,
              train_time=['1997', '2020'],
              val_time=['2020', '2021'],
              test_time=['2021', '2023'],
              **kwargs):
    train_set = ZsdDataset(data_dir=data_dir, training_time=train_time)
    validation_set = ZsdDataset(data_dir, val_time)
    test_set = ZsdDataset(data_dir, test_time)

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
    # return dataloader_test