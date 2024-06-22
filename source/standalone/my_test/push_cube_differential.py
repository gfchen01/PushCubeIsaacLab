# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

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

##
# Pre-defined configs
##
from omni.isaac.lab_assets import DIFFEREENTIAL_CFG



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

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
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
        self._vel_command[:, 1] = -((2 * self._processed_actions[:, 0]) + (self._processed_actions[:, 1] * self.wheel_base)) / (2 * self.wheel_radius)
        
        self._vel_command[:, 0] = 10.0
        self._vel_command[:, 1] = 10.0
        
        print(f"Velocity command: {self._vel_command[0, :]}")
        
        self._asset.set_joint_velocity_target(self._vel_command)

@configclass
class CarActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CarActionTerm
    """The class corresponding to the action term."""
    
    wheel_radius: float = 0.03
    """Radius of the wheel."""
    
    wheel_base: float = 0.1125

    # p_gain: float = 5.0
    # """Proportional gain of the PD controller."""
    # d_gain: float = 0.5
    # """Derivative gain of the PD controller."""

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

# Utility functions

def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins

def base_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w

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
    return -dis2obs(env, asset_cfg, obs_cfg)

def obs2goal_reward(env: ManagerBasedEnv, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance to the goal."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[obs_cfg.name]
    return -torch.log(torch.norm(asset.data.root_pos_w - env.scene.env_origins, dim=-1))

def is_car_too_far(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if the agent is too far from the obstacle."""
    return dis2obs(env, asset_cfg, obs_cfg) > 5.0

def is_obs_too_far(env: ManagerBasedEnv, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if the agent is too far from the obstacle."""
    return obs2origin(env, obs_cfg) > 5.0

def is_obs_at_goal(env: ManagerBasedEnv, obs_cfg: SceneEntityCfg) -> torch.Tensor:
    """Check if the agent is too far from the obstacle."""
    return obs2origin(env, obs_cfg) < 0.2

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
        velocity = ObsTerm(func=base_velocity, params={"asset_cfg": SceneEntityCfg("differential_car")})
        obstacle_position = ObsTerm(func=obstacle_position, params={"obstacle_cfg": SceneEntityCfg("cube")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
@configclass
class EventCfg:
    """Configuration for events."""

    # reset_robot = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-0.0, 0.0),
    #             "y": (-0.0, 0.0),
    #             "z": (-0.0, 0.0),
    #         },
    #         "asset_cfg": SceneEntityCfg("differential_car"),
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
    #         "asset_cfg": SceneEntityCfg("cube"),
    #     },
    # )
    
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={},
    )
    
@configclass
class RewardsCfg:
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    
    minus_dis2obs = RewTerm(func=dis2obs_reward, weight=1.0, params={"asset_cfg": SceneEntityCfg("differential_car"), 
                                                                     "obs_cfg": SceneEntityCfg("cube")})
    
    minus_obs2origin = RewTerm(func=obs2goal_reward, weight=1.0, params={"obs_cfg": SceneEntityCfg("cube")})

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # # (2) Too far from the obstacle
    # cube_too_far = DoneTerm(func=is_car_too_far, params={"asset_cfg": SceneEntityCfg("differential_car"), "obs_cfg": SceneEntityCfg("cube")}, time_out=True)

    # # (3) obstacle too far from origin
    # obs_too_far = DoneTerm(func=is_obs_too_far, params={"obs_cfg": SceneEntityCfg("cube")}, time_out=True)
    
    # # (4) obstacle at goal
    # obs_reach_goal = DoneTerm(func=is_obs_at_goal, params={"obs_cfg": SceneEntityCfg("cube")}, time_out=True)

@configclass
class PushCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=2.5)
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
        self.episode_length_s = 50


def main():
    env = ManagerBasedRLEnv(cfg=PushCubeEnvCfg())
    
    base_target_vel = torch.zeros(env.num_envs, 2, device=env.device)
    base_target_vel[:, 0] = 0.2
    obs, _ = env.reset()
    count = 0
    while simulation_app.is_running():
        if count % 1000 == 0:
            count = 0
            obs, _ = env.reset()
            print("-" * 80)
            print("[INFO] Reset the environment.")
        obs, _, _, _, _ = env.step(base_target_vel)
        
        base_actual_vel = obs["policy"][0, 3:6]
        print(f"Base actual velocity: {base_actual_vel}")
        count += 1
    
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
