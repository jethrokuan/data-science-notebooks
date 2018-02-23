#!/usr/bin/env bash
docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/work -v $(pwd)/data:/home/jovyan/work/data jethro/jupyter start.sh jupyter lab
