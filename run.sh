docker stop ahc
docker rm ahc
xhost + local:root

docker run -it \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" \
    --volume="$(pwd):/air_hockey_challenge" \
    --privileged \
    --network=host \
    --name=ahc \
    airhockeychallenge/challenge
