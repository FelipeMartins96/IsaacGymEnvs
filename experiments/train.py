import hydra
import gym
import isaacgym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.learning.replay_buffer import ReplayBuffer

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer("action_scale", torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.0))
        self.register_buffer("action_bias", torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.0))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg : DictConfig) -> None:
    envs = isaacgym_task_map[cfg.task.name](
            cfg=omegaconf_to_dict(cfg.task),
            rl_device=cfg.rl_device,
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            virtual_screen_capture=cfg.virtual_screen_capture,
            force_render=True,
        )

    writer = SummaryWriter()


    actor = Actor(envs).to(device=cfg.rl_device)
    qf1 = QNetwork(envs).to(device=cfg.rl_device)
    qf1_target = QNetwork(envs).to(device=cfg.rl_device)
    target_actor = Actor(envs).to(device=cfg.rl_device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=cfg.agent.lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=cfg.agent.lr)
    
    rb = ReplayBuffer(2000000, cfg.rl_device)
    start_time = time.time()
    episodic_return = torch.zeros(envs.num_envs, device=cfg.rl_device)
    episodic_length = torch.zeros(envs.num_envs, device=cfg.rl_device)
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(cfg.agent.total_timesteps):
        # ALGO LOGIC: put action logic here
        if rb.get_total_count() < cfg.agent.learning_starts:
            actions = torch.tensor([envs.action_space.sample() for _ in range(envs.num_envs)], dtype=torch.float32, device=cfg.rl_device)
        else:
            with torch.no_grad():
                actions = actor(obs['obs'])
                actions += torch.normal(actor.action_bias, actor.action_scale * cfg.agent.exploration_noise)

        actions = torch.clamp(actions, -1.0, 1.0)
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        writer.add_scalar("charts/mean_action_fx", actions[:,0].mean().item(), global_step)
        writer.add_scalar("charts/mean_action_fy", actions[:,1].mean().item(), global_step)

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        episodic_return += rewards
        episodic_length += 1
        if dones.any():
            writer.add_scalar("charts/episodic_return", episodic_return[dones].mean(), global_step)
            writer.add_scalar("charts/episodic_length", episodic_length[dones].mean(), global_step)
            episodic_return[dones] = 0
            episodic_length[dones] = 0


        # TRY NOT TO MODIFY: save data to replay buffer;
        rb.store(
            {
            'observations': obs['obs'], 
            'next_observations': next_obs['obs'],
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            }
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if rb.get_total_count() > cfg.agent.learning_starts:
            data = rb.sample(cfg.agent.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data['next_observations'])
                qf1_next_target = qf1_target(data['next_observations'], next_state_actions)
                next_q_value = data['rewards'].flatten() + (1 - data['dones'].flatten()) * cfg.agent.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data['observations'], data['actions']).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            actor_loss = -qf1(data['observations'], actor(data['observations'])).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # update the target network
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(cfg.agent.tau * param.data + (1 - cfg.agent.tau) * target_param.data)
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(cfg.agent.tau * param.data + (1 - cfg.agent.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


if __name__ == "__main__":
    train()
