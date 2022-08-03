import isaacgym
import torch
import time
import numpy as np
from isaacgymenvs.utils.rlgames_utils import get_rlgames_env_creator
from isaacgym import gymapi
cfg_task = {
    'physics_engine': 'physx',
    'name': 'VSS_V0',
}
cfg_task['env'] = {'numEnvs': 1}
cfg_task['sim'] = {
    'use_gpu_pipeline': True,
    # 'up_axis': 'z', 
    # 'dt':0.016667,
    # 'gravity': [0.0, 0.0, -10.0],
    # 'substeps': 10,
    # 'physx': {
    #     'contact_offset': 0.05,
    #     'rest_offset': 0.00025,
    #     'friction_correlation_distance': 0.001,
    #     'friction_offset_threshold': 0.01,
    #     'max_depenetration_velocity': 10,
    #     'bounce_threshold_velocity': 0.04
    # }
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

action_1 = torch.ones((1,)+envs.action_space.shape, dtype=torch.float, device=envs.rl_device, requires_grad=False)
action_0 = torch.zeros((1,)+envs.action_space.shape, dtype=torch.float, device=envs.rl_device, requires_grad=False)
obs = envs.reset()
velocity = obs['obs'][0][7].item()
print(obs['obs'][0][:3])



for i in range(5000):
    # action = action_1 * 5
    factor = (i-20)/30
    action = action_0 if i < 20 else action_1 / 2
    obs, reward, done, info = envs.step(torch.tensor(envs.action_space.sample(), dtype=torch.float, device=envs.rl_device))

    velocity_ = obs['obs'][0][7].item()
    print(f'{i-20} :', velocity_,  (velocity_-velocity)/envs.sim_params.dt, action)
    # print(obs['obs'][0][:3])
    velocity = velocity_
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