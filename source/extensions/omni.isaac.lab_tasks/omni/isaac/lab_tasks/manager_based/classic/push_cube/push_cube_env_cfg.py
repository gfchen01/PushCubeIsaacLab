import torch

# from omni.isaac.wheeled_robots.robots import WheeledRobot
# from omni.isaac.core.utils.types import ArticulationAction

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
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

import numpy as np

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration.

    The scene comprises of a ground plane, light source and floating cubes (gravity disabled).
    """

    # add terrain
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # add cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 2.0, 0.1)),
    )
    
    # jetbot = WheeledRobot(prim_path="{ENV_REGEX_NS}/jetbot",
    #                       name="Joan",
    #                       wheel_dof_names=["left_wheel_joint", "right_wheel_joint"]
    # )
    
    # action = ArticulationAction(joint_velocities = np.array([1.14, 1.42]))

    # jetbot.apply_wheel_actions(action)
    
    cube_obs: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_obs",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, 0.1)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )



##
# Custom action term
##


class CubeActionTerm(ActionTerm):
    _asset: RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        # call super constructor
        super().__init__(cfg, env)
        # create buffers
        self._raw_actions = torch.zeros(env.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 6, device=self.device)

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

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # no-processing of actions
        self._processed_actions[:, 0:2] = self._raw_actions[:]
        self._processed_actions[:, 2:6] = 0.0

    def apply_actions(self):
        self._vel_command = self._processed_actions
        self._asset.write_root_velocity_to_sim(self._vel_command)


@configclass
class CubeActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CubeActionTerm
    """The class corresponding to the action term."""


def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins

def obstacle_position(env: ManagerBasedEnv, obstacle_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    obstacle: RigidObject = env.scene[obstacle_cfg.name]
    return obstacle.data.root_pos_w - env.scene.env_origins


##
# Environment settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_vel = CubeActionTermCfg(asset_name="cube")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # cube velocity
        position = ObsTerm(func=base_position, params={"asset_cfg": SceneEntityCfg("cube")})
        obstacle_position = ObsTerm(func=obstacle_position, params={"obstacle_cfg": SceneEntityCfg("cube_obs")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-0.0, 0.0),
    #             "y": (-0.0, 0.0),
    #             "z": (-0.0, 0.0),
    #         },
    #         "asset_cfg": SceneEntityCfg("cube"),
    #     },
    # )
    
    # reset_obs = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "z": (-0.5, 0.5),
    #         },
    #         "asset_cfg": SceneEntityCfg("cube_obs"),
    #     },
    # )
    
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={},
    )

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
    return -dis2obs(env, asset_cfg, obs_cfg)

def obs2goal_reward(env: ManagerBasedEnv, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance to the goal."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[obs_cfg.name]
    return -torch.log(torch.norm(asset.data.root_pos_w - env.scene.env_origins, dim=-1))

def is_cube_too_far(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if the agent is too far from the obstacle."""
    return dis2obs(env, asset_cfg, obs_cfg) > 5.0

def is_obs_too_far(env: ManagerBasedEnv, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if the agent is too far from the obstacle."""
    return obs2origin(env, obs_cfg) > 5.0

def is_obs_at_goal(env: ManagerBasedEnv, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if the agent is too far from the obstacle."""
    return obs2origin(env, obs_cfg) < 0.2

@configclass
class RewardsCfg:
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    
    minus_dis2obs = RewTerm(func=dis2obs_reward, weight=1.0, params={"asset_cfg": SceneEntityCfg("cube"), "obs_cfg": SceneEntityCfg("cube_obs")})
    
    minus_obs2origin = RewTerm(func=obs2goal_reward, weight=1.0, params={"obs_cfg": SceneEntityCfg("cube_obs")})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # (2) Too far from the obstacle
    cube_too_far = DoneTerm(func=is_cube_too_far, params={"asset_cfg": SceneEntityCfg("cube"), "obs_cfg": SceneEntityCfg("cube_obs")}, time_out=True)

    # (3) obstacle too far from origin
    obs_too_far = DoneTerm(func=is_obs_too_far, params={"obs_cfg": SceneEntityCfg("cube_obs")}, time_out=True)
    
    # (4) obstacle at goal
    obs_reach_goal = DoneTerm(func=is_obs_at_goal, params={"obs_cfg": SceneEntityCfg("cube_obs")}, time_out=True)

@configclass
class PushCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4, env_spacing=2.5)
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
        self.sim.dt = 0.01
        self.sim.physics_material = self.scene.terrain.physics_material
        self.episode_length_s = 100
        