#!/usr/bin/env bash
echo $(pwd)
sudo docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/work jethro/jupyter start.sh jupyter lab
