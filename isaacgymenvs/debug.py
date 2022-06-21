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
while True:
    # scale = input("Scale:")
    scale = 0.0
    act = torch.ones((1,2)) * torch.tensor([0.0, float(scale)])
    done = torch.zeros(1)
    while done == 0:
        print(obs['obs'][0][1], obs['obs'][0][0])
        obs, reward, done, info = env.step(act)
        time.sleep(0.1)
