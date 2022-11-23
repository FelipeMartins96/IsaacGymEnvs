import argparse
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

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from torch.utils.tensorboard import SummaryWriter
from isaacgymenvs.learning.replay_buffer import ReplayBuffer

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod() + n_actions,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc_mu(x))
        return x


def train(args) -> None:
    task = isaacgymenvs.make(
        seed=args.seed,
        task="VSS",
        num_envs=args.num_envs,
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=not args.record,
        virtual_screen_capture=args.record,
        force_render=False,
    )

    writer = SummaryWriter(comment=args.experiment)
    device = "cuda:0"
    lr = 3e-4
    total_timesteps = 2000000
    learning_starts = 1e7
    batch_size = 4096
    gamma = 0.99
    tau = 0.005
    rb_size = 8000000

    n_controlled_robots = 1
    assert n_controlled_robots <= 3
    n_actions = n_controlled_robots * 2

    actor = Actor(task, n_actions).to(device=device)
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

    frames = []
    record_flag = 1
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here

        with torch.no_grad():
            exp_noise = random_ou(exp_noise)
            if rb.get_total_count() < learning_starts:
                actions = exp_noise
            else:
                actions = actor(obs['obs']) + exp_noise

        actions = torch.clamp(actions, -1.0, 1.0)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = task.step(actions)

        if global_step % 30000 == 0:
            record_flag = 1
        if args.record and record_flag:
            frames.append(task.render(mode='rgb_array'))
            record_flag += 1
            if record_flag > 200:
                clip = ImageSequenceClip(frames, fps=20)
                clip.write_videofile(f'{writer.get_logdir()}/video-{global_step}.mp4')
                frames = []
                record_flag = 0

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        real_next_obs = next_obs['obs'].clone()
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids):
            writer.add_scalar(
                "charts/episodic_length",
                infos['progress_buffer'][env_ids].mean(),
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_goal",
                infos['terminal_rewards']['goal'][env_ids].mean(),
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_grad",
                infos['terminal_rewards']['grad'][env_ids].mean(),
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_energy",
                infos['terminal_rewards']['energy'][env_ids].mean(),
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_move",
                infos['terminal_rewards']['move'][env_ids].mean(),
                global_step,
            )
            real_next_obs[env_ids] = infos["terminal_observation"][env_ids]
            dones = dones.logical_and(infos["time_outs"].logical_not())
            exp_noise[env_ids] *= 0.0

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
        if rb.get_total_count() > learning_starts:
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

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

            if global_step % 10000 == 0:
                torch.save(
                    actor.state_dict(),
                    f"{writer.get_logdir()}/actor{args.experiment}.pth",
                )
    torch.save(
        actor.state_dict(),
        f"{writer.get_logdir()}/actor{args.experiment}.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default=False, action="store_true")
    parser.add_argument("--experiment", default='', type=str)
    parser.add_argument("--num-envs", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    train(args)
