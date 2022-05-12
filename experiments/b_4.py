"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Body physics properties example
-------------------------------
An example that demonstrates how to load rigid body, update its properties
and apply various actions. Specifically, there are three scenarios that
presents the following:
- Load rigid body asset with varying properties
- Modify body shape properties
- Modify body visual properties
- Apply body force
- Apply body linear velocity
"""

# TODO: 4 Add point bodies to robot with fixed joints to Cube, and apply forces/velocity to it
# TODO: 5 Add wheels to robot and apply forces/velocity to it

from isaacgym import gymutil
from isaacgym import gymapi

# initialize gym
gym = gymapi.acquire_gym()
# parse arguments
args = gymutil.parse_arguments(description="Body Physics Properties Example")

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.relaxation = 0.9
    sim_params.flex.dynamic_friction = 0.0
    sim_params.flex.static_friction = 0.0
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim_params.gravity = gymapi.Vec3(0., 0., -9.8)
sim_params.up_axis = gymapi.UP_AXIS_Z
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
cp = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cp)
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 0.0
plane_params.dynamic_friction = 0.0

gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 1
spacing = 2
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# create list to mantain environment and asset handles
envs = []
box_handles = []
actor_handles = []

MM_TO_M = 1/1000.0

# Create box asset
box_size = 80 * MM_TO_M
box_options = gymapi.AssetOptions()
asset_box = gym.create_box(sim, box_size, box_size, box_size, box_options)

sp = gym.get_asset_rigid_shape_properties(asset_box)
sp[0].friction = 0.0
gym.set_asset_rigid_shape_properties(asset_box, sp)

# Create ball
ball_radius = 21.34 * MM_TO_M
asset_ball = gym.create_sphere(sim, ball_radius)


print('Creating %d environments' % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # add box
    actor_handle = gym.create_actor(env, asset_box, gymapi.Transform(p=gymapi.Vec3(-0.3,0.0,box_size/2)), 'robot', group=i, filter=0)
    ball_handle = gym.create_actor(env, asset_ball, gymapi.Transform(p=gymapi.Vec3(0.0,0.0,ball_radius)), 'ball', group=i, filter=0)
    gym.set_rigid_body_color(env, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.8,0.3,0.))
    
# look at the first env
cam_pos = gymapi.Vec3(0., -2., 2)
cam_target = gymapi.Vec3(0., 0., 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# subscribe to input events. This allows input to be used to interact
# with the simulation
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "w")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "s")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_K, "k")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_O, "o")

lw_force = gymapi.Vec3(0., 0., 0.)
rw_force = gymapi.Vec3(0., 0., 0.)
wheel_radius = 12.5 * MM_TO_M
lw_joint_pos = gymapi.Vec3(-box_size/2, 0., wheel_radius - box_size/2)
rw_joint_pos = gymapi.Vec3(box_size/2, 0., wheel_radius - box_size/2)
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)


    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.value == 0:
            lw_force = gymapi.Vec3(0., 0., 0.)
            rw_force = gymapi.Vec3(0., 0., 0.)
        elif evt.action == "w" and evt.value > 0:
            lw_force = gymapi.Vec3(0., .05, 0.)
        elif evt.action == "s" and evt.value > 0:
            lw_force = gymapi.Vec3(0., -.05, 0.)
        elif evt.action == "k" and evt.value > 0:
            rw_force = gymapi.Vec3(0., -.05, 0.)
        elif evt.action == "o" and evt.value > 0:
            rw_force = gymapi.Vec3(0., .05, 0.)

    gym.apply_body_force_at_pos(env=env, rigidHandle=actor_handle, force=lw_force, pos=lw_joint_pos, space=gymapi.CoordinateSpace.LOCAL_SPACE) 
    gym.apply_body_force_at_pos(env=env, rigidHandle=actor_handle, force=rw_force, pos=rw_joint_pos, space=gymapi.CoordinateSpace.LOCAL_SPACE) 

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
