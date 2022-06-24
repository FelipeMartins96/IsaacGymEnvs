import isaacgym
import torch
from tasks import VSS
import time
cfg_task = {
    'physics_engine': 'physx'
}
cfg_task['env'] = {'numEnvs': 1}
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

while True:
    # scale = input("Scale:")
    step = 0
    act = torch.ones((1,2)) * torch.tensor([0.0, float(step)])
    done = torch.zeros(1)
    while done == 0:
        obs, reward, done, info = env.step(act)
        print(f"done = {done}, robot_x = {obs['obs'][0][0]}, robot_y = {obs['obs'][0][1]}")
        time.sleep(1)
