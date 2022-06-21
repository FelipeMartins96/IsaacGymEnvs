import isaacgym
import torch
from tasks import VSS

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

env.reset()
act = torch.zeros((1,2))
done = torch.zeros(1)
while done == 0:
    obs, reward, done, info = env.step(act)
