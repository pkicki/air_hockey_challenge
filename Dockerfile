FROM nvidia/cuda:11.6.2-base-ubuntu20.04 as base

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Europe/Berlin \
    PYTHONPATH=/air_hockey_challenge

# For nvidia GPU
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


RUN apt-get update && \
    apt-get install -y \
    python3-pip python-is-python3 git \
    ffmpeg libsm6 libxext6 vim git \
    xauth tzdata libgl1-mesa-glx libgl1-mesa-dri \
    libeigen3-dev lsb-release curl coinor-libclp-dev cmake

COPY requirements.txt .
RUN pip install -U pip && \
    pip install networkx==3.1 && \
    pip install -r requirements.txt

# experiment launcher hotfix
RUN sed -i "28 i \ \ \ \ except ValueError:\n\ \ \ \ \ \ \ \ args['git_hash'] = ''\n\ \ \ \ \ \ \ \ args['git_url'] = ''" /usr/local/lib/python3.8/dist-packages/experiment_launcher/utils.py

RUN git clone https://github.com/NVlabs/storm.git && \
    cd storm && \
    pip install -e . && \
    pip install hydra urdf_parser_py

RUN pip uninstall -y mushroom-rl && \
    git clone https://github.com/MushroomRL/mushroom-rl.git && \
    cd mushroom-rl && \
    git checkout ePPO && \
    pip install --no-use-pep517 -e .[all]

RUN pip install mp-pytorch