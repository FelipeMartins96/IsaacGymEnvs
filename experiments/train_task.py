import hydra
import gym
import isaacgym
import isaacgymenvs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from copy import deepcopy
import random

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from torch.utils.tensorboard import SummaryWriter
from isaacgymenvs.learning.replay_buffer import ReplayBuffer
from isaacgymenvs.utils.reformat import omegaconf_to_dict

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod() + n_actions,
            256,
        )
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.tanh(self.fc_mu(x))
        return x

@hydra.main(config_name="config", config_path="../isaacgymenvs/cfg")
def train(cfg) -> None:
    assert len(cfg.experiment)
    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    task = isaacgymenvs.make(
        seed=cfg.seed,
        task=cfg.task_name,
        num_envs=cfg.task.env.numEnvs,
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=cfg.headless,
        virtual_screen_capture=cfg.capture_video,
        force_render=False,
        cfg=cfg,
    )
    cfg_dict = omegaconf_to_dict(cfg)
    if cfg.wandb_activate:
        import wandb
        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            sync_tensorboard=False,
            name=cfg.experiment,
            resume="allow",
            monitor_gym=True,
            config=cfg_dict,
        )

    if cfg.capture_video:
            task.is_vector_env = True
            task = gym.wrappers.RecordVideo(
                task,
                f"videos/{cfg.experiment}",
                step_trigger=lambda step: step % 10000 == 0,
                video_length=1000,
            )

    writer = SummaryWriter(comment=cfg.experiment)
    device = "cuda:0"
    lr = 3e-4
    total_timesteps = 400000 if not False else 1100
    learning_starts = 1e7
    batch_size = 4096
    gamma = cfg.train.params.config.gamma
    tau = 0.005
    rb_size = 10000000

    n_controlled_robots = task.n_controlled_robots
    assert n_controlled_robots <= 3
    n_actions = n_controlled_robots * 2

    actor = Actor(task, n_actions).to(device=device)
    if False:
        actor.load_state_dict(torch.load(args.checkpoint))
    qf1 = QNetwork(task, n_actions).to(device=device)
    qf1_target = QNetwork(task, n_actions).to(device=device)
    target_actor = Actor(task, n_actions).to(device=device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=lr)

    rb = ReplayBuffer(rb_size, device)
    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs = deepcopy(task.reset())

    ou_theta = 0.1
    ou_sigma = 0.15
    rewards_info = torch.zeros(5, device=device, requires_grad=False)
    ep_count = 0

    def random_ou(prev):
        noise = (
            prev
            - ou_theta * prev
            + torch.normal(
                0.0,
                ou_sigma,
                size=prev.size(),
                device=device,
                requires_grad=False,
            )
        )
        return noise.clamp(-1.0, 1.0)

    exp_noise = task.zero_actions()

    for global_step in range(total_timesteps):
        logs = {}
        # ALGO LOGIC: put action logic here

        with torch.no_grad():
            exp_noise = random_ou(exp_noise)
            if False:
                actions = actor(obs['obs'])
            elif rb.get_total_count() < learning_starts:
                actions = exp_noise
            else:
                actions = actor(obs['obs']) + exp_noise

        actions = torch.clamp(actions, -1.0, 1.0)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = task.step(actions)

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        real_next_obs = next_obs['obs'].clone()
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        
        if len(env_ids):
            ep_count += len(env_ids)
            rewards_info[0] += infos['progress_buffer'][env_ids].sum()
            rewards_info[1] += infos['terminal_rewards']['goal'][env_ids].sum()
            rewards_info[2] += infos['terminal_rewards']['grad'][env_ids].sum()
            rewards_info[3] += infos['terminal_rewards']['energy'][env_ids].sum()
            rewards_info[4] += infos['terminal_rewards']['move'][env_ids].sum()
            real_next_obs[env_ids] = infos["terminal_observation"][env_ids]
            dones = dones.logical_and(infos["time_outs"].logical_not())
            exp_noise[env_ids] *= 0.0

        if not False and ep_count and global_step % task.max_episode_length == 0:
            rewards_info /= ep_count
            ep_count = 0
            logs.update({
                "episode_lengths/iter" : rewards_info[0].clone(),
                "charts/episodic_goal" : rewards_info[1].clone(),
                "charts/episodic_grad" : rewards_info[2].clone(),
                "charts/episodic_energy" : rewards_info[3].clone(),
                "charts/episodic_move" : rewards_info[4].clone(),
                "rewards/iter" : rewards_info[1:].sum(),
            })
            rewards_info *= 0

        # TRY NOT TO MODIFY: save data to replay buffer;
        rb.store(
            {
                'observations': obs['obs'],
                'next_observations': real_next_obs,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
            }
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = deepcopy(next_obs)

        # ALGO LOGIC: training.
        if not False and rb.get_total_count() > learning_starts:
            data = rb.sample(batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data['next_observations'])
                qf1_next_target = qf1_target(
                    data['next_observations'], next_state_actions
                )
                next_q_value = data['rewards'].flatten() + (
                    1 - data['dones'].flatten()
                ) * gamma * (qf1_next_target).view(-1)

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
            for param, target_param in zip(
                actor.parameters(), target_actor.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

            if global_step % task.max_episode_length == 0:
                logs.update({
                    "losses/qf1_loss" : qf1_loss.item(),
                    "losses/actor_loss" : actor_loss.item(),
                    "losses/qf1_values" : qf1_a_values.mean().item(),
                    "charts/SPS" : int(global_step / (time.time() - start_time))
                })

            if global_step % 10000 == 0:
                torch.save(
                    actor.state_dict(),
                    f"{writer.get_logdir()}/actor{cfg.experiment}.pth",
                )

        if logs:
            logs.update({'global_step': global_step})
            wandb.log(logs)
    if not False:
        torch.save(
            actor.state_dict(),
            f"{writer.get_logdir()}/actor{cfg.experiment}.pth",
        )


if __name__ == "__main__":
    train()
