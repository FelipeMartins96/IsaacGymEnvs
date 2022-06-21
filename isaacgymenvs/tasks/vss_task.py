import os
from tokenize import group
from isaacgym import gymtorch
from isaacgym import gymapi
import numpy as np
import torch

from tasks.base.vec_task import VecTask

class VSS(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = 100

        self.cfg["env"]["numObservations"] = 8
        self.cfg["env"]["numActions"] = 2

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.0, -2.0, 2.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(_root_tensor)
        self.actor_pos = self.root_state.view(self.num_envs, -1, 13)[..., 0:3]
        self.robot_quat = self.root_state.view(self.num_envs, -1, 13)[:, 2, 3:7]

    def create_sim(self):
    #    - set up-axis
        # set the up axis to be z-up given that assets are y-up by default
        # self.up_axis = "z"
    #    - call super().create_sim with device args (see docstring)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
    #    - create ground plane
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0.0
        plane_params.dynamic_friction = 0.0
        self.gym.add_ground(self.sim, plane_params)
    #    - set up environments
        lower = gymapi.Vec3(-1, -1, 0.0)
        upper = gymapi.Vec3(1, 1, 1)
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/urdf")
        
        # Load field urdf
        asset_field_file = "vss_3v3_field.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        field_asset = self.gym.load_asset(self.sim, asset_root, asset_field_file, asset_options)

        # Load robot urdf
        asset_robot_file = "vss_robot.urdf"
        asset_options = gymapi.AssetOptions()
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_robot_file, asset_options)

        # Load ball urdf
        asset_ball_file = "vss_ball.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        ball_asset = self.gym.load_asset(self.sim, asset_root, asset_ball_file, asset_options)

        self.robot_handles = []
        self.ball_handles = []
        self.field_handles = []
        self.envs = []
        ball_radius = 21.34 * 1/1000.0
        for i in range(self.num_envs):
            # Create environment
            env_ptr = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            # add field
            field_handle = self.gym.create_actor(env_ptr, field_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0)), 'field', group=i, filter=0)

            # add ball
            ball_handle = self.gym.create_actor(env_ptr, ball_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, ball_radius)), 'ball', group=i, filter=0)
            
            # add robot
            robot_handle = self.gym.create_actor(env_ptr, robot_asset, gymapi.Transform(p=gymapi.Vec3(-0.2, 0.0, 0.0375)), 'robot', group=i, filter=0)

            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)
            self.ball_handles.append(ball_handle)
            self.field_handles.append(field_handle)
        self.n_env_rigid_bodies = self.gym.get_env_rigid_body_count(env_ptr)


    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        MAX_FORCE = 5.5
        
        actions_tensor = torch.zeros((self.num_envs, self.n_env_rigid_bodies, 3), device=self.device, dtype=torch.float)
        # import pdb; pdb.set_trace()
        actions_tensor[:, -2:, 1] = actions.to(self.device).squeeze() * MAX_FORCE
        forces = gymtorch.unwrap_tensor(actions_tensor)
        # apply only forces
        self.gym.apply_rigid_body_force_tensors(self.sim, forces, None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

    def reset_idx(self, env_ids):
        x_positions = 1.4 * (torch.rand((len(env_ids),2), device=self.device) - 0.5)
        y_positions = 1.2 * (torch.rand((len(env_ids), 2), device=self.device) - 0.5)

        self.actor_pos[env_ids, 1:, 0] = x_positions[:]
        self.actor_pos[env_ids, 1:, 1] = y_positions[:]
        self.actor_pos[env_ids, 1, 2] = 21.34 * 1/1000.0
        self.actor_pos[env_ids, 2, 2] = 0.0375


        # import pdb; pdb.set_trace()
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state))

        self.reset_buf[env_ids] = 0 
        self.progress_buf[env_ids] = 0

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_actor_root_state_tensor(self.sim)

        # Actors ids 0: field, 1: ball, 2: robot
        self.obs_buf[env_ids, 0] = self.actor_pos[env_ids, 2, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.actor_pos[env_ids, 2, 1].squeeze()
        self.obs_buf[env_ids, 2] = self.actor_pos[env_ids, 1, 0].squeeze()
        self.obs_buf[env_ids, 3] = self.actor_pos[env_ids, 1, 1].squeeze()
        self.obs_buf[env_ids, 4:8] = self.robot_quat[env_ids, :].squeeze()

    
    def compute_reward(self):
        # retrieve environment observations from buffer
        robot_pos = self.obs_buf[:, 0:2]
        ball_pos = self.obs_buf[:, 2:4]

        self.rew_buf[:], self.reset_buf[:] = compute_vss_reward(robot_pos, ball_pos, self.reset_buf, self.progress_buf, self.max_episode_length
        )

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_vss_reward(robot_pos, ball_pos, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    
    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = -torch.linalg.norm(robot_pos-ball_pos, dim=1)

    # adjust reward for reset agents
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset