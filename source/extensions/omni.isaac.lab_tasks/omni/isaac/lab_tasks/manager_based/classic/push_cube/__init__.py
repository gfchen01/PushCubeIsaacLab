# Copyright (c) 2022-2024, Guofei Chen, guofei@cmu.edu.
# All rights reserved.
#

"""This sub-module contains the functions that are specific to the push manipulation environments."""

import gymnasium as gym

from . import agents, push_cube_env_cfg

gym.register(
    id="PushCube",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": push_cube_env_cfg.PushCubeEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.PushCubePPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)