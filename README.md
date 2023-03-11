# water_transparency
Diploma on water transparency

# Installation
- `docker build -t water_transparency .` - to build docker
- To run a container, use:
```
docker run -d -it --init \
--gpus=all \
--volume="$PWD:/app" \
--shm-size 40G \
--publish="8891:8891" \
--name watertrans \
water_transparency bash
```

```
jupyter lab --ip 0.0.0.0 --port 8891 --allow-root --no-browser
```