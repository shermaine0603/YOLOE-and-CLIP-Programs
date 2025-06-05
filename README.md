# YOLOE-and-CLIP-Programs

https://docs.ultralytics.com/models/yoloe/#textvisual-prompt-models

Run this when you open the terminal.
```sh
xhost +local:root
```

```sh
docker build . -t rag
```
```sh
docker rm rag
docker run --gpus all \
--env="NVIDIA_DRIVER_CAPABILITIES=all" \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-it -v /home/raus/YOLOECLIP:/pic --name rag rag 
```
```sh
cd YOLOE-and-CLIP-Programs
```
```sh
python3 YOLOE_and_oLMO_2.py
```
