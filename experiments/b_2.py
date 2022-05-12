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

# TODO: 2 Test using Local and global coordinates file:///home/fbm2/Documents/isaacgym/docs/api/python/enum_py.html#isaacgym.gymapi.CoordinateSpace flag only available for forces
# TODO: 3 Add Ball
# TODO: 4 Add point bodies to robot with fixed joints to Cube, and apply forces/velocity to it
# TODO: 5 Add wheels to robot and apply forces/velocity to it

def print_asset_info(asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" %
          (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # Iterate through joints
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # iterate through degrees of freedom (DOFs)
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))


def print_actor_info(gym, env, actor_handle):

    name = gym.get_actor_name(env, actor_handle)

    body_names = gym.get_actor_rigid_body_names(env, actor_handle)
    body_dict = gym.get_actor_rigid_body_dict(env, actor_handle)

    joint_names = gym.get_actor_joint_names(env, actor_handle)
    joint_dict = gym.get_actor_joint_dict(env, actor_handle)

    dof_names = gym.get_actor_dof_names(env, actor_handle)
    dof_dict = gym.get_actor_dof_dict(env, actor_handle)

    print()
    print("===== Actor: %s =======================================" % name)

    print("\nBodies")
    print(body_names)
    print(body_dict)

    print("\nJoints")
    print(joint_names)
    print(joint_dict)

    print("\n Degrees Of Freedom (DOFs)")
    print(dof_names)
    print(dof_dict)
    print()

    # Get body state information
    body_states = gym.get_actor_rigid_body_states(
        env, actor_handle, gymapi.STATE_ALL)

    # Print some state slices
    print("Poses from Body State:")
    print(body_states['pose'])          # print just the poses

    print("\nVelocities from Body State:")
    print(body_states['vel'])          # print just the velocities
    print()

    # iterate through bodies and print name and position
    body_positions = body_states['pose']['p']
    for i in range(len(body_names)):
        print("Body '%s' has position" % body_names[i], body_positions[i])

    print("\nDOF states:")

    # get DOF states
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

    # print some state slices
    # Print all states for each degree of freedom
    print(dof_states)
    print()

    # iterate through DOFs and print name and position
    dof_positions = dof_states['pos']
    for i in range(len(dof_names)):
        print("DOF '%s' has position" % dof_names[i], dof_positions[i])


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

# Create box asset
box_size = 0.08
box_options = gymapi.AssetOptions()
asset_box = gym.create_box(sim, box_size, box_size, box_size, box_options)

sp = gym.get_asset_rigid_shape_properties(asset_box)
sp[0].friction = 0.0
gym.set_asset_rigid_shape_properties(asset_box, sp)

print_asset_info(asset_box, "Box")

print('Creating %d environments' % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # add box
    name = 'robot'
    actor_handle = gym.create_actor(env, asset_box, gymapi.Transform(p=gymapi.Vec3(0.0,0.0,0.1)), name, i, 0)

# look at the first env
cam_pos = gymapi.Vec3(0., -2., 2)
cam_target = gymapi.Vec3(0., 0., 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# subscribe to input events. This allows input to be used to interact
# with the simulation
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "w")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "a")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "s")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "d")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "q")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "e")

Force = gymapi.Vec3(0., 0., 0.)
Torque = gymapi.Vec3(0., 0., 0.)

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)


    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.value == 0:
            Force = gymapi.Vec3(0., 0., 0.)
            Torque = gymapi.Vec3(0., 0., 0.)
        elif evt.action == "w":
            Force = gymapi.Vec3(0., .5, 0.)
        elif evt.action == "a":
            Force = gymapi.Vec3(-.5, 0., 0.)
        elif evt.action == "s":
            Force = gymapi.Vec3(0., -.5, 0.)
        elif evt.action == "d":
            Force = gymapi.Vec3(.5, 0., 0.)
        elif evt.action == "q":
            Torque = gymapi.Vec3(0., 0., 0.01)
        elif evt.action == "e":
            Torque = gymapi.Vec3(0., 0., -0.01)

    gym.apply_body_forces(env, gym.get_rigid_handle(env,'robot','box'), Force, Torque, gymapi.CoordinateSpace.LOCAL_SPACE)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
