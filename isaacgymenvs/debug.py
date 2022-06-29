import isaacgym
import torch
from tasks import VSS
import time
import numpy as np

cfg_task = {
    'physics_engine': 'physx'
}
cfg_task['env'] = {'numEnvs': 2}
cfg_task['sim'] = {
    'use_gpu_pipeline': False,
    'up_axis': 'z', 
    'dt':0.016667,
    'gravity': [0.0, 0.0, -9.81],
}

env = VSS(
    cfg=cfg_task,
    sim_device='cuda:0',
    graphics_device_id=0,
    headless=False
    )
obs = env.reset()

print(f"robot_x = {obs['obs'][0][0]}, robot_y = {obs['obs'][0][1]}")

r_vel = np.linalg.norm(env.robot_root_state[0,7:10].cpu())
print(r_vel)

force = -1
while True:
    force += 1
    print(f'force = {force}')
    act = torch.ones((1,2)) * torch.tensor([force, force])
    done = torch.zeros(1)
    while not done.any():
        obs, reward, done, info = env.step(act)
        # print(f"robot_x = {obs['obs'][0][0]}, robot_y = {obs['obs'][0][1]}, robot_z = {obs['obs'][0][2]}")
        nr_vel = np.linalg.norm(env.robot_root_state[0,7:10].cpu())
        print((nr_vel - r_vel)/0.016667)
        r_vel = nr_vel

        time.sleep(0.05)
