import isaacgym
import torch
from tasks import VSS
import time
cfg_task = {
    'physics_engine': 'physx'
}
cfg_task['env'] = {'numEnvs': 2}
cfg_task['sim'] = {
    'use_gpu_pipeline': True,
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

force = -1
while True:
    force += 1
    print(f'force = {force}')
    act = torch.ones((1,2)) * torch.tensor([force, 0.0])
    done = torch.zeros(1)
    while not done.any():
        obs, reward, done, info = env.step(act)
        time.sleep(0.05)
