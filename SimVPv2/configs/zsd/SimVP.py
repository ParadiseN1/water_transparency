method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 256
N_T = 8
N_S = 2
# training
lr = 5e-3
batch_size = 32
drop_path = 0.1
warmup_epoch = 0

# python ./SimVPv2/tools/non_dist_train.py -d ZsdChl --lr 1e-3 -c ./SimVPv2/configs/zsd/SimVP.py --ex_name ZsdChl
