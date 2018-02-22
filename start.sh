#!/usr/bin/env bash
echo $(pwd)
sudo docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/tensorflow-notebook start.sh jupyter lab
