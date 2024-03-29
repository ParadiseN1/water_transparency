

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, **kwargs):
    pre_seq_length = kwargs.get('pre_seq_length', 10)
    aft_seq_length = kwargs.get('aft_seq_length', 10)
    if dataname == 'kitticaltech':
        from .dataloader_kitticaltech import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length, aft_seq_length)
    elif 'kth' in dataname:  # 'kth', 'kth20', 'kth40'
        from .dataloader_kth import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length, aft_seq_length)
    elif dataname == 'mmnist':
        from .dataloader_moving_mnist import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length, aft_seq_length)
    elif dataname == 'taxibj':
        from .dataloader_taxibj import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, pre_seq_length, aft_seq_length)
    elif 'weather' in dataname:  # 'weather', 'weather_t2m', etc.
        from .dataloader_weather import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs)
    elif 'simple' in dataname.lower():
        from .dataloader_zsd_simple import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs)
    elif 'transparency' in dataname.lower():
        from .dataloader_transparency import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs)
    # elif 'zsd' in dataname.lower():
    #     from .dataloader_zsd import load_data
    #     return load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs)
    elif 'zsdchlkd' in dataname.lower():
        from .dataloader_CHL_KD import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs)
    elif 'zsdchl' in dataname.lower():
        from .dataloader_CHL import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs)
    elif 'zsdkd' in dataname.lower():
        from .dataloader_KD import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
# python tools/non_dist_train.py -d transparency --lr 1e-3 -c ./configs/zsd/simvp/SimVP.py --ex_name transparency
