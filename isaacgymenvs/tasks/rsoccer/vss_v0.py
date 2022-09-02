import os
from tokenize import group
from isaacgym import gymtorch
from isaacgym import gymapi
import numpy as np
import torch
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis

from isaacgymenvs.tasks.base.vec_task import VecTask

class VSS_V0(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.render_mode = 'rgb_array'

        self.max_episode_length = 300

        self.cfg["env"]["numObservations"] = 9
        self.cfg["env"]["numActions"] = 2
        self.cfg['sim']['up_axis'] = 'z'
        self.cfg['sim']['dt'] =0.016667
        self.cfg['sim']['gravity'] = [0.0, 0.0, -10.0]
        self.cfg['sim']['substeps'] = 10
        self.cfg['sim']['physx'] = {}
        self.cfg['sim']['physx']['contact_offset'] = 0.05
        self.cfg['sim']['physx']['rest_offset'] = 0.00025
        self.cfg['sim']['physx']['friction_correlation_distance'] = 0.001
        self.cfg['sim']['physx']['friction_offset_threshold'] = 0.01
        self.cfg['sim']['physx']['max_depenetration_velocity'] = 10
        self.cfg['sim']['physx']['bounce_threshold_velocity'] = 0.04

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.0, -2.0, 2.4)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_state = gymtorch.wrap_tensor(_root_tensor).view(self.num_envs, 3, 13)

        self.ball_root_state = self.root_state[:, 1]
        self.robot_root_state = self.root_state[:, 2]

        # [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w, vel_x, vel_y, vel_z, w_x, w_y, w_z]
        self.robot_initial = torch.tensor([0.0, 0.0, 0.0052, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device, requires_grad=False)
        self.ball_initial = torch.tensor([0.0, 0.0, 0.0052, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device, requires_grad=False)
        self.z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float, device=self.device, requires_grad=False)
        self.field_scale = torch.tensor([1.4, 1.2], dtype=torch.float, device=self.device, requires_grad=False)
        
        self.reset_idx(np.arange(self.num_envs))
        self.compute_observations()

    def reset_idx(self, env_ids):

        self.ball_root_state[env_ids] = self.ball_initial
        self.robot_root_state[env_ids] = self.robot_initial

        #randomize positions
        rand_pos = (torch.rand((len(env_ids), 2, 2), dtype=torch.float, device=self.device) - 0.5) * self.field_scale
        self.ball_root_state[env_ids, :2] = rand_pos[:, 0]
        self.robot_root_state[env_ids, :2] = rand_pos[:, 1]

        #randomize rotations
        rand_angles = torch_rand_float(-np.pi, np.pi, (len(env_ids), 2), device=self.device)
        self.ball_root_state[env_ids, 3:7] = quat_from_angle_axis(rand_angles[:, 0], self.z_axis)
        self.robot_root_state[env_ids, 3:7] = quat_from_angle_axis(rand_angles[:, 1], self.z_axis)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state))

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
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/rsoccer")
        
        # Load field urdf
        asset_field_file = "vss_divB_field.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        field_asset = self.gym.load_asset(self.sim, asset_root, asset_field_file, asset_options)

        # Load robot urdf
        asset_robot_file = "vss_robot.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.max_linear_velocity = 1.2
        asset_options.max_angular_velocity = 30
        asset_options.linear_damping = 5.0
        asset_options.angular_damping = 5.0
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_robot_file, asset_options)
        robot_asset_rigid_shape_properties = self.gym.get_asset_rigid_shape_properties(robot_asset)
        robot_asset_rigid_shape_properties[0].friction = 0.1
        
        self.gym.set_asset_rigid_shape_properties(robot_asset, robot_asset_rigid_shape_properties)

        # Load ball urdf
        asset_ball_file = "golf_ball.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        ball_asset = self.gym.load_asset(self.sim, asset_root, asset_ball_file, asset_options)

        self.robot_handles = []
        self.ball_handles = []
        self.field_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # Create environment
            env_ptr = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            # add field
            field_handle = self.gym.create_actor(env_ptr, field_asset, gymapi.Transform(), 'field', group=i, filter=0)

            # add ball
            ball_handle = self.gym.create_actor(env_ptr, ball_asset, gymapi.Transform(), 'ball', group=i, filter=0)
            
            # add robot
            robot_handle = self.gym.create_actor(env_ptr, robot_asset, gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.1), r=None), 'robot', group=i, filter=0)

            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)
            self.ball_handles.append(ball_handle)
            self.field_handles.append(field_handle)

        self.n_env_rigid_bodies = self.gym.get_env_rigid_body_count(env_ptr)

    def pre_physics_step(self, actions):
        # reset progress_buf for envs reseted on previous step
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.progress_buf[env_ids] = 0

        # implement pre-physics simulation code here
        #    - e.g. apply actions
        # Applying a force on X axis at the center of the robot base
        actions.to(self.device)

        forces_tensor = torch.zeros((self.num_envs, self.n_env_rigid_bodies, 3), device=self.device, dtype=torch.float)
        torques_tensor = torch.zeros((self.num_envs, self.n_env_rigid_bodies, 3), device=self.device, dtype=torch.float)
        
        # wheel_forces = forces_tensor[:, -2:, :1].view((self.num_envs, 2))
        # wheel_forces[:] = actions[:] / 2

        forces_tensor[:, -3, :2] = actions[:]

        forces = gymtorch.unwrap_tensor(forces_tensor)
        torques = gymtorch.unwrap_tensor(torques_tensor)
        self.gym.apply_rigid_body_force_tensors(self.sim, forces, torques, gymapi.GLOBAL_SPACE)

    def compute_reward(self):
        # Calculate previous robot distance to ball
        self.rew_buf[:] = compute_vss_reward(self.robot_root_state[:, :2], self.ball_root_state[:, :2])
        # Refresh state tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # Diference between last robot distance to ball and current
        self.rew_buf[:] -= compute_vss_reward(self.robot_root_state[:, :2], self.ball_root_state[:, :2])

    def compute_observations(self, env_ids=None):
        # Actors ids 0: field, 1: ball, 2: robot
        env_ids = np.arange(self.num_envs) if env_ids is None else env_ids
        self.obs_buf[env_ids, :3] = self.robot_root_state[env_ids, :3]
        self.obs_buf[env_ids, 3:6] = self.robot_root_state[env_ids, 7:10]
        self.obs_buf[env_ids, -3:] = self.ball_root_state[env_ids, :3] # ball x, y

    def post_physics_step(self):
        self.progress_buf += 1

        # Calculate rewards (Refreshes state tensors)
        self.compute_reward()

        # Save observations previously to resets
        self.compute_observations()
        self.extras["terminal_observation"] = self.obs_buf.clone().to(self.rl_device)

        # Reset dones (Reseting only on timeouts)
        self.reset_buf = (self.progress_buf >= self.max_episode_length)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.compute_observations(env_ids)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_vss_reward(robot_pos, ball_pos):
    # type: (Tensor, Tensor) -> Tensor
    
    return torch.linalg.norm(robot_pos-ball_pos, dim=1)