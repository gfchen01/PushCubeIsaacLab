import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Push cube standalone test")
parser.add_argument("--draw", action="store_true", default=False, help="Draw pointcloud from depth")
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils
import os
import omni.replicator.core as rep

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg, Articulation, ArticulationCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

# sensors
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from omni.isaac.lab_assets import VELODYNE_VLP_16_RAYCASTER_CFG

from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.utils import convert_dict_to_backend

##
# Pre-defined configs
##
from omni.isaac.lab_assets import DIFFEREENTIAL_CFG

import open3d as o3d
import numpy as np

class CarActionTerm(ActionTerm):
    """Simple action term that implements a PD controller to track a target position.

    The action term is applied to the cube asset. It involves two steps:

    1. **Process the raw actions**: Typically, this includes any transformations of the raw actions
       that are required to map them to the desired space. This is called once per environment step.
    2. **Apply the processed actions**: This step applies the processed actions to the asset.
       It is called once per simulation step.

    In this case, the action term simply applies the raw actions to the cube asset. The raw actions
    are the desired target positions of the cube in the environment frame. The pre-processing step
    simply copies the raw actions to the processed actions as no additional processing is required.
    The processed actions are then applied to the cube asset by implementing a PD controller to
    track the target position.
    """

    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        # call super constructor
        super().__init__(cfg, env)
        # create buffers
        self._raw_actions = torch.zeros(env.num_envs, 2, device=self.device) # target velocity: [vx, w]
        self._processed_actions = torch.zeros(env.num_envs, 2, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 2, device=self.device) # target wheel velocity: [w_l, w_r]
        
        self._last_action = torch.zeros(env.num_envs, 2, device=self.device)
        
        self.wheel_radius = cfg.wheel_radius
        self.wheel_base = cfg.wheel_base

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def last_action(self) -> torch.Tensor:
        return self._last_action

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        self._last_action = self._processed_actions
        # store the raw actions
        self._raw_actions[:] = actions
        # no-processing of actions
        self._processed_actions[:] = self._raw_actions[:]

    def apply_actions(self):
        # # implement a PD controller to track the target position
        # pos_error = self._processed_actions - (self._asset.data.root_pos_w - self._env.scene.env_origins)
        # vel_error = -self._asset.data.root_lin_vel_w
        # # set velocity targets
        # self._vel_command[:, :3] = self.p_gain * pos_error + self.d_gain * vel_error
        # self._asset.write_root_velocity_to_sim(self._vel_command)
        
        self._vel_command[:, 0] = ((2 * self._processed_actions[:, 0]) - (self._processed_actions[:, 1] * self.wheel_base)) / (2 * self.wheel_radius)
        self._vel_command[:, 1] = ((2 * self._processed_actions[:, 0]) + (self._processed_actions[:, 1] * self.wheel_base)) / (2 * self.wheel_radius)
        
        self._asset.set_joint_velocity_target(self._vel_command)

@configclass
class CarActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CarActionTerm
    """The class corresponding to the action term."""
    
    wheel_radius: float = 0.0337
    """Radius of the wheel."""
    
    wheel_base: float = 0.12

    # p_gain: float = 5.0
    # """Proportional gain of the PD controller."""
    # d_gain: float = 0.5
    # """Derivative gain of the PD controller."""
    
SURROUNDING_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/clp/Downloads/T_shape.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            kinematic_enabled=True,
        ),
        # mass_props=sim_utils.MassPropertiesCfg(
        #     mass=1000.0,  
        # ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        scale=(0.2, 0.2, 0.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.02)),
)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration.

    The scene comprises of a ground plane, light source and floating cubes (gravity disabled).
    """

    # add terrain
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # add car
    differential_car: ArticulationCfg = DIFFEREENTIAL_CFG.copy()
    differential_car.prim_path = "{ENV_REGEX_NS}/differential_car"
    
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.02)),
    )
    
    surrounding: RigidObjectCfg = SURROUNDING_CFG.copy()
    surrounding.prim_path = "{ENV_REGEX_NS}/surrounding"
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/differential_car/chassis/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e3)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.3), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    
    # ray_caster = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/differential_car/chassis",
    #     mesh_prim_paths=["/World/ground"],
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=16, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=0.2
    #     ), # Velodyne VLP-16
    #     attach_yaw_only=True,
    #     debug_vis=True,
    #     max_distance=100,
    # )

# Utility functions

def rgb_image(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """RGB image from the camera."""
    # extract the used quantities (to enable type-hinting)
    camera: Camera = env.scene["camera"]
    # print(f'image shape: {camera.data.output["rgb"].shape}')
    dim_num = camera.data.output["rgb"].shape[0]
    return camera.data.output["rgb"].view(dim_num, -1)

def depth_image(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Depth image from the camera."""
    # extract the used quantities (to enable type-hinting)
    camera: Camera = env.scene["camera"]
    dim_num = camera.data.output["distance_to_image_plane"].shape[0]
    return camera.data.output["distance_to_image_plane"].view(dim_num, -1)

def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins

def base_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_w

def base_orientation_2d(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.heading_w

def base_linear_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w

def base_angular_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w

def obstacle_position(env: ManagerBasedEnv, obstacle_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    obstacle: RigidObject = env.scene[obstacle_cfg.name]
    return obstacle.data.root_pos_w - env.scene.env_origins

# Utility functions

def dis2obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance to the obstacle."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    obs: RigidObject = env.scene[obs_cfg.name]
    return torch.norm(asset.data.root_pos_w - obs.data.root_pos_w, dim=-1)

def obs2origin(env: ManagerBasedEnv, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance to the goal."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[obs_cfg.name]
    return torch.norm(asset.data.root_pos_w - env.scene.env_origins, dim=-1)

def dis2obs_reward(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance to the obstacle."""
    return torch.exp(-dis2obs(env, asset_cfg, obs_cfg))

def obs2goal_reward(env: ManagerBasedEnv, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance to the goal."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[obs_cfg.name]
    return -torch.log(torch.norm(asset.data.root_pos_w - env.scene.env_origins, dim=-1) + 0.05)

def first_order_smooth_reward(env: ManagerBasedEnv) -> torch.Tensor:
    # First order smoothness reward
    actionTerm = env.action_manager.get_term("joint_vel")
    return torch.exp(-torch.norm(actionTerm.processed_actions - actionTerm.last_action, dim=-1))

def is_car_too_far(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if the agent is too far from the obstacle."""
    return dis2obs(env, asset_cfg, obs_cfg) > 2.0

def is_obs_too_far(env: ManagerBasedEnv, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if the agent is too far from the obstacle."""
    return obs2origin(env, obs_cfg) > 2.0

def is_obs_at_goal(env: ManagerBasedEnv, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if the agent is too far from the obstacle."""
    return obs2origin(env, obs_cfg) < 0.05

# config classes

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_vel = CarActionTermCfg(asset_name="differential_car")
    
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # cube velocity
        position = ObsTerm(func=base_position, params={"asset_cfg": SceneEntityCfg("differential_car")})
        orientation_2d = ObsTerm(func=base_orientation_2d, params={"asset_cfg": SceneEntityCfg("differential_car")})
        linear_velocity = ObsTerm(func=base_linear_velocity, params={"asset_cfg": SceneEntityCfg("differential_car")})
        angular_velocity = ObsTerm(func=base_angular_velocity, params={"asset_cfg": SceneEntityCfg("differential_car")})
        obstacle_position = ObsTerm(func=obstacle_position, params={"obstacle_cfg": SceneEntityCfg("cube")})
        
        # rgb = ObsTerm(func=rgb_image, params={"asset_cfg": SceneEntityCfg("differential_car")})
        depth = ObsTerm(func=depth_image, params={"asset_cfg": SceneEntityCfg("differential_car")})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("differential_car"),
        },
    )
    
    reset_obs = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-0.01, 0.01)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )
    
    # reset_all = EventTerm(
    #     func=mdp.reset_scene_to_default,
    #     mode="reset",
    #     params={},
    # )
    
@configclass
class RewardsCfg:
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    
    minus_dis2obs = RewTerm(func=dis2obs_reward, weight=1.0, params={"asset_cfg": SceneEntityCfg("differential_car"), 
                                                                     "obs_cfg": SceneEntityCfg("cube")})
    
    minus_obs2origin = RewTerm(func=obs2goal_reward, weight=1.0, params={"obs_cfg": SceneEntityCfg("cube")})
    
    smoothness = RewTerm(func=first_order_smooth_reward, weight=0.1)
    
    minus_is_alive = RewTerm(func=mdp.is_alive, weight=-0.01)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # # (2) Too far from the obstacle
    # cube_too_far = DoneTerm(func=is_car_too_far, params={"asset_cfg": SceneEntityCfg("differential_car"), "obs_cfg": SceneEntityCfg("cube")}, time_out=True)

    # # (3) obstacle too far from origin
    # obs_too_far = DoneTerm(func=is_obs_too_far, params={"obs_cfg": SceneEntityCfg("cube")}, time_out=True)
    
    # (4) obstacle at goal
    obs_reach_goal = DoneTerm(func=is_obs_at_goal, params={"obs_cfg": SceneEntityCfg("cube")}, time_out=True)

@configclass
class PushCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4, env_spacing=10)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        
        # simulation settings
        self.sim.dt = 0.1
        self.sim.physics_material = self.scene.terrain.physics_material
        self.episode_length_s = 25


def main():
    env = ManagerBasedRLEnv(cfg=PushCubeEnvCfg())
    
    base_target_vel = torch.zeros(env.num_envs, 2, device=env.device)
    base_target_vel[:, 0] = 1.0
    base_target_vel[:, 1] = 2.0
    
    obs, _ = env.reset()
    count = 0
    
    non_image_obs_dim = 12
    
    if env.sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)
        
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
    )

    cloud_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "pointcloud")
    if not os.path.exists(cloud_output_dir):
        os.makedirs(cloud_output_dir)

    camera = env.scene["camera"]
    
    is_saved = False
    while simulation_app.is_running():
        if count % 50 == 0:
            count = 0
            is_saved = False
            obs, _ = env.reset()
            print("-" * 80)
            print("[INFO] Reset the environment.")
        obs, rew, terminated, truncated, info = env.step(base_target_vel)
        
        # extract depth image
        depth_images = obs["policy"][:, non_image_obs_dim:]
        depth_images = depth_images.view(env.num_envs, 480, 640)
        # print(f"Depth images pos: {env.scene['camera'].data.pos_w}")
        
        
        if args_cli.draw:
            if count < 40 or is_saved:
                continue
            is_saved = True
            
            pointclouds = torch.tensor([], device=env.device)
            for i in range(env.num_envs):
                pointcloud = create_pointcloud_from_depth(
                    intrinsic_matrix=camera.data.intrinsic_matrices[i],
                    depth=camera.data.output["distance_to_image_plane"][i],
                    position=camera.data.pos_w[i],
                    orientation=camera.data.quat_w_ros[i],
                    device=env.device,
                )
                pointclouds = torch.stack([pointclouds, pointcloud], dim=0)
                # print(f"Pointcloud shape: {pointcloud.shape}")
            
            # pointclouds = torch.tensor(pointclouds, device=env.device)
            
            # vehicle_pos = obs["policy"][:, :3]
            # vehicle_pos = vehicle_pos[:, :2]
            
            # vehicle_ori = obs["policy"][:, 4]
            
            # vehicle_pose = torch.cat([vehicle_pos, vehicle_ori.unsqueeze(1)], dim=1)
            
            # torch.save(vehicle_pose, os.path.join(cloud_output_dir, f"vehicle_pose_{count}.pt"))
            # torch.save(pointclouds, os.path.join(cloud_output_dir, f"pointcloud_{count}.pt"))
            
            # pointclouds = torch.cat(pointclouds, dim=0)
            # # pointclouds = pointclouds.view(-1, 3)
            # # print(f"Pointcloud shape: {pointclouds.shape}")

            # # pointcloud_o3d = o3d.geometry.PointCloud()
            # # pointcloud_o3d.points = o3d.utility.Vector3dVector(pointclouds.cpu().numpy())
            # # pointcloud_o3d = pointcloud_o3d.voxel_down_sample(voxel_size=0.05)
            
            # # Save the pointcloud to the 'output' directory in the same folder as the script
            # # o3d.io.write_point_cloud(os.path.join(cloud_output_dir, f"pointcloud_{count}.ply"), pointcloud_o3d, write_ascii=True)
            # # print('Pointcloud saved.')

            # # points_downsampled = np.array(pointcloud_o3d.points)
            # # pointclouds = torch.tensor(points_downsampled, device=env.device)

            if len(pointclouds) > 0 and sim.has_gui():
                pc_markers.visualize(translations=pointclouds)
        
        # print(f"camera data output: {camera.data.output}")
        
        if args_cli.save:
            # Save images from camera at camera_index
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            # tensordict allows easy indexing of tensors in the dictionary
            cam_data = convert_dict_to_backend(camera.data.output, backend="numpy")
            # print(f"cam_data type: {type(cam_data['rgb'])}")
            single_cam_data = {"rgb": cam_data["rgb"][0]}
            
            # Extract the other information
            single_cam_info = camera.data.info[0]

            # Pack data back into replicator format to save them using its writer
            if env.sim.get_version()[0] == 4:
                rep_output = {"annotators": {}}
                for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                    if info is not None:
                        rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                    else:
                        rep_output["annotators"][key] = {"render_product": {"data": data}}
            else:
                rep_output = dict()
                for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                    if info is not None:
                        rep_output[key] = {"data": data, "info": info}
                    else:
                        rep_output[key] = data
            # Save images
            # Note: We need to provide On-time data for Replicator to save the images.
            rep_output["trigger_outputs"] = {"on_time": camera.frame[0]}
            rep_writer.write(rep_output)
        
        # base_actual_vel = obs["policy"][0, 3:9]
        # print(f"Base actual velocity: {base_actual_vel}")
        count += 1
    
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
