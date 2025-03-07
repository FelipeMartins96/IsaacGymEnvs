import isaacgym
import isaacgymenvs
import torch

envs = isaacgymenvs.make(
	seed=0, 
	task="FactoryTaskGears", 
	num_envs=2, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
for _ in range(2000):
	input()
	obs, reward, done, info = envs.step(
		torch.rand((2,)+envs.action_space.shape, device="cuda:0")
	)
	print(info)