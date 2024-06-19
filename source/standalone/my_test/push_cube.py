# Copyright (c) 2022-2024, Guofei Chen, guofei@cmu.edu.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script creates a simple environment with a floating cube. The cube is controlled by a PD
controller to track an arbitrary target position.

While going through this tutorial, we recommend you to pay attention to how a custom action term
is defined. The action term is responsible for processing the raw actions and applying them to the
scene entities. The rest of the environment is similar to the previous tutorials.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p source/standalone/tutorials/04_envs/floating_cube.py --num_envs 32
"""

# from __future__ import annotations

import torch

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

##
# Custom action term
##


class CubeActionTerm(ActionTerm):
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

    p_gain: float = 5.0
    """Proportional gain of the PD controller."""
    d_gain: float = 0.5
    """Derivative gain of the PD controller."""


##
# Custom observation term
##


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
# Scene definition
##


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
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "z": (-0.5, 0.5),
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

# Reward function body

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

@configclass
class RewardsCfg:
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    
    minus_dis2obs = RewTerm(func=dis2obs, weight=-1.0, params={"asset_cfg": SceneEntityCfg("cube"), "obs_cfg": SceneEntityCfg("cube_obs")})
    
    minus_obs2origin = RewTerm(func=obs2origin, weight=-1.0, params={"obs_cfg": SceneEntityCfg("cube_obs")})

##
# Environment configuration
##


@configclass
class CubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=64, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        
        # simulation settings
        self.sim.dt = 0.01
        self.sim.physics_material = self.scene.terrain.physics_material
        self.episode_length_s = 50


# def main():
#     """Main function."""

#     # setup base environment
#     env = ManagerBasedRLEnv(cfg=CubeEnvCfg())

#     # # setup target position commands
#     # # target_position = torch.rand(env.num_envs, 3, device=env.device) * 2
#     # target_position = torch.zeros(env.num_envs, 3, device=env.device)
#     # target_position[:, 2] += 0.0
#     # # offset all targets so that they move to the world origin
#     # target_position -= env.scene.env_origins

#     # simulate physics
#     count = 0
#     obs, _ = env.reset()
#     while simulation_app.is_running():
#         with torch.inference_mode():
#             # step env
#             cube_vel_cmd = torch.randn_like(env.action_manager.action)
#             obs, rew, terminated, truncated, info = env.step(cube_vel_cmd)
#             # update counter
#             print(f'Reward: {rew}')

#     env.close()


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()
