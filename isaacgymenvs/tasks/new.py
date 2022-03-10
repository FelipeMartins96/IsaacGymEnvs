from isaacgym import gymtorch
from isaacgym import gymapi

from tasks.base.vec_task import VecTask

class MyNewTask(VecTask):
    
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = 500

        super().__init__(self.cfg, sim_device, graphics_device_id, headless)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    
    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        #    - call super().create_sim with device args (see docstring)
        #    - create ground plane
        #    - set up environments
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, 0, int(self.num_envs**0.5))

    def _create_envs(self, num_envs, env_spacing, num_per_row):
        spacing = 1.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.density = 1.0
        sphere_asset = self.gym.create_sphere(self.sim, 0.1, asset_options)
        pose = gymapi.Transform()
        pose.p.z = 2.0
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.balls = []
        self.envs = []
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            ball_handle = self.gym.create_actor(
                env_ptr, sphere_asset, pose, 'ball', i, 1, 0
            )

            self.envs.append(env_ptr)
            self.balls.append(ball_handle)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        pass

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        pass
