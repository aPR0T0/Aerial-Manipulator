# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .. import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

_AERIAL_MANIP_URDF_PATH = os.path.join(
    ISAACLAB_ASSETS_DATA_DIR,
    "robots",
    "aerial_manipulator",
    "simple_mesh",
    "simple_mesh_aerial_manipulator.urdf",
)


AERIAL_MANIP_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_AERIAL_MANIP_URDF_PATH,
        root_link_name="base_link",
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="velocity",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=120.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "manipulator_joint_1": 0.0,
            "manipulator_joint_2": 0.0,
        },
        joint_vel={
            "manipulator_joint_1": 0.0,
            "manipulator_joint_2": 0.0,
        },
    ),
    actuators={
        "manipulator": ImplicitActuatorCfg(
            joint_names_expr=["manipulator_joint_.*"],
            effort_limit_sim={"manipulator_joint_1": 0.25, "manipulator_joint_2": 0.15},
            velocity_limit_sim={"manipulator_joint_1": 14.0, "manipulator_joint_2": 18.0},
            stiffness=0.0,
            damping=8.0,
        ),
    },
)
"""Configuration for a minimal aerial manipulator (2-DOF arm on drone body)."""
