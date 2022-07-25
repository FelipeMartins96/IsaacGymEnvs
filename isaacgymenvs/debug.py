import isaacgym
import torch
import time
import numpy as np
from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator

cfg_task = {
    'physics_engine': 'physx',
    'name': 'VSS_V0',
}
cfg_task['env'] = {'numEnvs': 2}
cfg_task['sim'] = {
    'use_gpu_pipeline': True,
    'up_axis': 'z', 
    'dt':0.016667,
    'gravity': [0.0, 0.0, -9.81],
}

create_rlgpu_env = get_rlgames_env_creator(
    seed=0,
    task_config=cfg_task,
    task_name=cfg_task["name"],
    sim_device='cuda:0',
    rl_device='cuda:0',
    graphics_device_id=0,
    headless=False,
    multi_gpu=False,
    virtual_screen_capture=False,
    force_render=False,
)

envs = create_rlgpu_env()

print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)

obs = envs.reset()
print(obs)
for _ in range(2000):
    input()
    obs, reward, done, info = envs.step(torch.zeros((2,)+envs.action_space.shape, device="cuda:0"))
    print(obs, reward, done, info)
    envs.render()
# print(torch.cuda.current_device())
# env = VSS(
#     cfg=cfg_task,
#     rl_device="cuda:0",
#     sim_device='cuda:0',
#     graphics_device_id=0,
#     headless=False,
#     virtual_screen_capture=False,
#     force_render=False
#     )
# obs = env.reset()

# while True:
#     act = torch.zeros((1,2))
#     done = torch.zeros(1)
#     while not done.any():
#         obs, reward, done, info = env.step(act)
#         env.render()
#         time.sleep(0.05)