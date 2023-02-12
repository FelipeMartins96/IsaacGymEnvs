FROM isaacgym:latest

WORKDIR /workspaces/IsaacGymEnvs

USER root
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends xvfb ffmpeg

COPY --chown=gymuser . .
RUN pip install -e .

USER gymuser

RUN pip install --upgrade pip & \
    pip install black wandb
