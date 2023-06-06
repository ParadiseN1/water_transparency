# water_transparency
Diploma on water transparency

# Installation
- `docker build -t water_transparency .` - to build docker
- To run a container, use:
```
docker run -d -it --init \
--gpus=all \
--volume="$PWD:/app" \
--shm-size 4G \
--publish="8891:8891" \
--name watertrans \
water_transparency bash
```

```
jupyter lab --ip 0.0.0.0 --port 8891 --allow-root --no-browser
```

```
CUDA_VISIBLE_DEVICES=1 nohup python SimVPv2/tools/non_dist_train.py -d transparencyDataset  --lr 1e-3 -c SimVPv2/configs/zsd_big/SimVP.py --resume_from=results/zsd_big/checkpoints/latest.pth --ex_name zsd_big > zsd_solo_L.txt &
```