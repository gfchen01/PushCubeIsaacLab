import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

DIFFEREENTIAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/luke/Downloads/jetbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15), joint_pos={"left_wheel_joint": 0.0, "right_wheel_joint": 0.0}
    ),
    actuators={
        "left_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=1e4,
        ),
        "right_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint"], 
            effort_limit=400.0, 
            velocity_limit=100.0, 
            stiffness=0.0, 
            damping=1e4,
        ),
    },
)
"""Configuration for a simple Cartpole robot."""
