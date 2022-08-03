FROM isaacgym:latest

WORKDIR /workspaces/IsaacGymEnvs

USER root
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends xvfb ffmpeg
USER gymuser

COPY --chown=gymuser . .

RUN pip install -e .