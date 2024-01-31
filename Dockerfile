FROM nvidia/cuda:11.6.2-base-ubuntu20.04 as base

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Europe/Berlin \
    PYTHONPATH=/air_hockey_challenge

RUN apt-get update && apt-get install -y python3-pip python-is-python3 git ffmpeg libsm6 libxext6 vim


WORKDIR /wheels

RUN apt-get update && apt-get -y install git

COPY requirements.txt .
RUN pip install -U pip && \
    pip install networkx==3.1 && \
    pip install -r requirements.txt

#COPY --from=pip-build /wheels /wheels
#WORKDIR /src

RUN apt-get update && apt-get -y install 

#RUN pip install -U pip  \
#    && pip install --no-cache-dir \
#    --no-index \
#    -r /wheels/requirements.txt \
#    -f /wheels \
#    && rm -rf /wheels

# experiment launcher hotfix
RUN sed -i "28 i \ \ \ \ except ValueError:\n\ \ \ \ \ \ \ \ args['git_hash'] = ''\n\ \ \ \ \ \ \ \ args['git_url'] = ''" /usr/local/lib/python3.8/dist-packages/experiment_launcher/utils.py

# For nvidia GPU
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# libgl1-mesa-glx libgl1-mesa-dri for non-nvidia GPU
RUN apt-get update && apt-get -y install xauth tzdata libgl1-mesa-glx libgl1-mesa-dri

# TODO move up at more serious refactor
RUN git clone https://github.com/NVlabs/storm.git && \
    cd storm && \
    pip install -e .

RUN pip uninstall -y mushroom-rl && \
    git clone https://github.com/MushroomRL/mushroom-rl.git && \
    cd mushroom-rl && \
    git checkout ePPO && \
    pip install --no-use-pep517 -e .[all]

RUN apt update && apt install -y libeigen3-dev lsb-release curl coinor-libclp-dev cmake

RUN git clone https://github.com/stevengj/nlopt.git && \
    cd nlopt && mkdir build && cd build && \
    cmake .. && make && make install

RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt update && \
    apt install -y ros-noetic-pinocchio

ENV CMAKE_PREFIX_PATH=/opt/ros/noetic
ENV LD_LIBRARY_PATH=/opt/ros/noetic/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

RUN git clone https://github.com/pkicki/hitting_point_optimization.git && \
    cd hitting_point_optimization && \
    ./build.sh