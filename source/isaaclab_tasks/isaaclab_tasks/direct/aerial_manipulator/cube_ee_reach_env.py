# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

##
# Pre-defined configs
##
from isaaclab_assets import AERIAL_MANIP_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class CubeEeReachEnvWindow(BaseEnvWindow):
    """Window manager for the cube end-effector reach environment."""

    def __init__(self, env: CubeEeReachEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class CubeEeReachEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 6
    observation_space = 25
    state_space = 0
    debug_vis = True

    ui_window_class_type = CubeEeReachEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=2.5, replicate_physics=True, clone_in_fabric=False
    )

    # robot
    robot: ArticulationCfg = AERIAL_MANIP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01
    wrench_body_name = "base_link"
    end_effector_body_name = "end_effector"

    manipulator_joint_names_expr = ["manipulator_joint_1", "manipulator_joint_2"]
    manip_boundary_margin = 0.05
    manip_reset_velocity_ratio = 0.4
    manip_velocity_limit_ratio = 1.0

    # cube
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.85, 0.35)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
                max_depenetration_velocity=10.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    cube_xy_position_range = 0.5
    cube_height = 0.015
    ee_goal_height_offset = 0.1

    # reward scales
    ee_distance_reward_scale = 20.0
    ee_progress_reward_scale = 20.0
    # base_to_goal_reward_scale = 6.0
    base_xy_align_reward_scale = 22.0
    base_above_ee_reward_scale = 8.0
    base_above_ee_margin = 0.05
    arm_vertical_straight_reward_scale = 8.0
    arm_vertical_xy_tolerance = 0.05
    time_penalty_reward_scale = -1.0
    lin_vel_reward_scale = -0.10
    ang_vel_reward_scale = -0.05
    tilt_reward_scale = -0.5
    manip_joint_vel_reward_scale = -0.01
    manip_action_rate_reward_scale = -0.03
    success_bonus_reward = 12.0

    # success criteria
    success_ee_distance = 0.04
    success_lin_vel_threshold = 0.35
    success_ang_vel_threshold = 0.65
    success_hold_time_s = 2.0


class CubeEeReachEnv(DirectRLEnv):
    cfg: CubeEeReachEnvCfg

    def __init__(self, cfg: CubeEeReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._body_id = self._robot.find_bodies(self.cfg.wrench_body_name)[0]
        if len(self._body_id) != 1:
            raise RuntimeError(f"Expected one body named '{self.cfg.wrench_body_name}', found {len(self._body_id)}")

        self._ee_body_ids, self._ee_body_names = self._robot.find_bodies(self.cfg.end_effector_body_name)
        if len(self._ee_body_ids) != 1:
            raise RuntimeError(
                f"Expected one body named '{self.cfg.end_effector_body_name}', found {len(self._ee_body_ids)}"
            )
        self._ee_body_idx = self._ee_body_ids[0]

        self._manip_joint_ids, self._manip_joint_names = self._robot.find_joints(
            self.cfg.manipulator_joint_names_expr, preserve_order=True
        )
        if len(self._manip_joint_ids) == 0:
            raise RuntimeError(
                f"Failed to resolve manipulator joints with patterns: {self.cfg.manipulator_joint_names_expr}"
            )
        self._manip_joint_ids_tensor = torch.tensor(self._manip_joint_ids, dtype=torch.long, device=self.device)
        self._num_manip_joints = len(self._manip_joint_ids)

        self._manip_actions = torch.zeros(self.num_envs, self._num_manip_joints, device=self.device)
        self._previous_manip_actions = torch.zeros(self.num_envs, self._num_manip_joints, device=self.device)

        self._manip_prev_joint_vel = torch.zeros(self.num_envs, self._num_manip_joints, device=self.device)
        self._episode_max_manip_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_max_manip_acc = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self._previous_ee_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._distance_init_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_hold_steps_required = max(1, math.ceil(self.cfg.success_hold_time_s / self.step_dt))
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "ee_distance",
                "ee_progress",
                # "base_to_goal",
                "base_xy_align",
                "base_above_ee",
                "arm_vertical_straight",
                "time_penalty",
                "lin_vel",
                "ang_vel",
                "tilt",
                "manip_joint_vel",
                "manip_action_rate",
                "success_bonus",
            ]
        }

        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._cube = RigidObject(self.cfg.object)
        self.scene.rigid_objects["cube"] = self._cube

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._manip_actions[:] = self._actions[:, 4:]

        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4]

    def _apply_action(self):
        self._apply_manipulator_actions()
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _compute_ee_goal_pos_w(self) -> torch.Tensor:
        ee_goal_pos_w = self._cube.data.root_pos_w.clone()
        ee_goal_pos_w[:, 2] += self.cfg.ee_goal_height_offset
        return ee_goal_pos_w

    def _compute_goal_stable_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_goal_pos_w = self._compute_ee_goal_pos_w()
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_idx]
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b

        ee_distance = torch.linalg.norm(ee_goal_pos_w - ee_pos_w, dim=1)
        lin_speed = torch.linalg.norm(root_lin_vel_b, dim=1)
        ang_speed = torch.linalg.norm(root_ang_vel_b, dim=1)

        stable_mask = torch.logical_and(
            ee_distance < self.cfg.success_ee_distance,
            torch.logical_and(
                lin_speed < self.cfg.success_lin_vel_threshold,
                ang_speed < self.cfg.success_ang_vel_threshold,
            ),
        )
        return stable_mask, ee_distance

    def _apply_manipulator_actions(self):
        joint_pos = self._robot.data.joint_pos[:, self._manip_joint_ids_tensor]
        joint_vel = self._robot.data.joint_vel[:, self._manip_joint_ids_tensor]
        joint_pos_limits = self._robot.data.soft_joint_pos_limits[:, self._manip_joint_ids_tensor]
        joint_vel_limits = torch.clamp(
            self._robot.data.soft_joint_vel_limits[:, self._manip_joint_ids_tensor], min=1.0e-3, max=30.0
        )

        joint_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]
        boundary_margin = torch.minimum(
            torch.full_like(joint_range, self.cfg.manip_boundary_margin),
            0.45 * joint_range,
        )
        lower_turn = joint_pos_limits[..., 0] + boundary_margin
        upper_turn = joint_pos_limits[..., 1] - boundary_margin

        blocked_upper = torch.logical_and(joint_pos >= upper_turn, self._manip_actions > 0.0)
        blocked_lower = torch.logical_and(joint_pos <= lower_turn, self._manip_actions < 0.0)
        blocked_motion = torch.logical_or(blocked_upper, blocked_lower)
        safe_manip_actions = torch.where(blocked_motion, torch.zeros_like(self._manip_actions), self._manip_actions)

        manip_vel_target = safe_manip_actions * joint_vel_limits * self.cfg.manip_velocity_limit_ratio
        self._robot.set_joint_velocity_target(manip_vel_target, joint_ids=self._manip_joint_ids)

        manip_joint_vel = joint_vel.abs().amax(dim=1)
        manip_joint_acc = ((joint_vel - self._manip_prev_joint_vel) / self.physics_dt).abs().amax(dim=1)
        self._episode_max_manip_vel = torch.maximum(self._episode_max_manip_vel, manip_joint_vel)
        self._episode_max_manip_acc = torch.maximum(self._episode_max_manip_acc, manip_joint_acc)
        self._manip_prev_joint_vel[:] = joint_vel

    def _get_observations(self) -> dict:
        cube_pos_l = self._cube.data.root_pos_w - self.scene.env_origins
        ee_goal_pos_l = cube_pos_l.clone()
        ee_goal_pos_l[:, 2] += self.cfg.ee_goal_height_offset

        ee_pos_l = self._robot.data.body_pos_w[:, self._ee_body_idx] - self.scene.env_origins
        ee_to_goal_l = ee_goal_pos_l - ee_pos_l

        root_pos_l = self._robot.data.root_pos_w - self.scene.env_origins
        # base_to_goal_l = ee_goal_pos_l - root_pos_l

        joint_pos = self._robot.data.joint_pos[:, self._manip_joint_ids_tensor]
        joint_vel = self._robot.data.joint_vel[:, self._manip_joint_ids_tensor]
        joint_pos_limits = self._robot.data.soft_joint_pos_limits[:, self._manip_joint_ids_tensor]
        joint_vel_limits = torch.clamp(
            self._robot.data.soft_joint_vel_limits[:, self._manip_joint_ids_tensor], min=1.0e-3
        )

        joint_pos_center = 0.5 * (joint_pos_limits[..., 0] + joint_pos_limits[..., 1])
        joint_pos_half_range = torch.clamp(0.5 * (joint_pos_limits[..., 1] - joint_pos_limits[..., 0]), min=1.0e-3)
        joint_pos_norm = (joint_pos - joint_pos_center) / joint_pos_half_range
        joint_vel_norm = joint_vel / joint_vel_limits

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                cube_pos_l,
                ee_pos_l,
                ee_to_goal_l,
                # base_to_goal_l,
                joint_pos_norm,
                joint_vel_norm,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b

        ee_goal_pos_w = self._compute_ee_goal_pos_w()
        cube_pos_w = self._cube.data.root_pos_w
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_idx]
        root_pos_w = self._robot.data.root_pos_w

        stable_mask, ee_distance = self._compute_goal_stable_mask()
        # base_distance = torch.linalg.norm(ee_goal_pos_w - root_pos_w, dim=1)
        base_xy_distance = torch.linalg.norm(cube_pos_w[:, :2] - root_pos_w[:, :2], dim=1)
        base_minus_ee_z = root_pos_w[:, 2] - ee_pos_w[:, 2]
        base_to_ee_xy_distance = torch.linalg.norm(root_pos_w[:, :2] - ee_pos_w[:, :2], dim=1)

        ee_distance_mapped = 1 - torch.tanh(ee_distance / 0.12)
        # base_distance_mapped = 1 - torch.tanh(base_distance / 0.8)
        base_xy_align_mapped = 1 - torch.tanh(base_xy_distance / 0.3)
        base_above_ee_mapped = torch.tanh((base_minus_ee_z - self.cfg.base_above_ee_margin) / 0.05)
        arm_vertical_straight_mapped = 1 - torch.tanh(base_to_ee_xy_distance / self.cfg.arm_vertical_xy_tolerance)
        ee_progress = torch.where(
            self._distance_init_pending,
            torch.zeros_like(ee_distance),
            self._previous_ee_distance - ee_distance,
        )

        lin_vel_sq = torch.sum(torch.square(root_lin_vel_b), dim=1)
        ang_vel_sq = torch.sum(torch.square(root_ang_vel_b), dim=1)
        tilt_error = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        manip_joint_vel = torch.sum(torch.square(self._robot.data.joint_vel[:, self._manip_joint_ids_tensor]), dim=1)
        manip_action_rate = torch.sum(torch.square(self._manip_actions - self._previous_manip_actions), dim=1)

        success_hold_steps = torch.where(
            stable_mask, self._success_hold_steps + 1, torch.zeros_like(self._success_hold_steps)
        )
        success_after_hold = success_hold_steps >= self._success_hold_steps_required

        rewards = {
            "ee_distance": ee_distance_mapped * self.cfg.ee_distance_reward_scale * self.step_dt,
            "ee_progress": ee_progress * self.cfg.ee_progress_reward_scale,
            # "base_to_goal": base_distance_mapped * self.cfg.base_to_goal_reward_scale * self.step_dt,
            "base_xy_align": base_xy_align_mapped * self.cfg.base_xy_align_reward_scale * self.step_dt,
            "base_above_ee": base_above_ee_mapped * self.cfg.base_above_ee_reward_scale * self.step_dt,
            "arm_vertical_straight": arm_vertical_straight_mapped
            * self.cfg.arm_vertical_straight_reward_scale
            * self.step_dt,
            "time_penalty": torch.full_like(ee_distance, self.cfg.time_penalty_reward_scale * self.step_dt),
            "lin_vel": lin_vel_sq * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_sq * self.cfg.ang_vel_reward_scale * self.step_dt,
            "tilt": tilt_error * self.cfg.tilt_reward_scale * self.step_dt,
            "manip_joint_vel": manip_joint_vel * self.cfg.manip_joint_vel_reward_scale * self.step_dt,
            "manip_action_rate": manip_action_rate * self.cfg.manip_action_rate_reward_scale * self.step_dt,
            "success_bonus": success_after_hold.float() * self.cfg.success_bonus_reward,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value

        self._previous_ee_distance[:] = ee_distance
        self._distance_init_pending[:] = False
        self._previous_manip_actions[:] = self._manip_actions

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        stable_mask, _ = self._compute_goal_stable_mask()

        self._success_hold_steps = torch.where(
            stable_mask,
            self._success_hold_steps + 1,
            torch.zeros_like(self._success_hold_steps),
        )
        self._success = self._success_hold_steps >= self._success_hold_steps_required

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        terminated = torch.logical_or(died, self._success)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        assert env_ids is not None

        successful_envs = self._success[env_ids]
        ee_goal_pos_w = self._compute_ee_goal_pos_w()

        final_ee_goal_distance = torch.linalg.norm(
            ee_goal_pos_w[env_ids] - self._robot.data.body_pos_w[env_ids, self._ee_body_idx],
            dim=1,
        ).mean()
        final_ee_to_cube_distance = torch.linalg.norm(
            self._cube.data.root_pos_w[env_ids] - self._robot.data.body_pos_w[env_ids, self._ee_body_idx],
            dim=1,
        ).mean()
        # final_base_distance = torch.linalg.norm(
        #     ee_goal_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids],
        #     dim=1,
        # ).mean()
        final_base_xy_to_cube_distance = torch.linalg.norm(
            self._cube.data.root_pos_w[env_ids, :2] - self._robot.data.root_pos_w[env_ids, :2],
            dim=1,
        ).mean()
        final_base_minus_ee_z = torch.mean(
            self._robot.data.root_pos_w[env_ids, 2] - self._robot.data.body_pos_w[env_ids, self._ee_body_idx, 2]
        )
        final_base_to_ee_xy_distance = torch.linalg.norm(
            self._robot.data.root_pos_w[env_ids, :2] - self._robot.data.body_pos_w[env_ids, self._ee_body_idx, :2],
            dim=1,
        ).mean()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        success_count = torch.count_nonzero(successful_envs).item()
        total_count = len(env_ids)
        died_count = torch.count_nonzero(
            torch.logical_and(self.reset_terminated[env_ids], torch.logical_not(successful_envs))
        ).item()
        time_out_count = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        if success_count > 0:
            success_time = torch.mean(self.episode_length_buf[env_ids][successful_envs].float() * self.step_dt).item()
        else:
            success_time = 0.0

        extras["Episode_Termination/success"] = success_count
        extras["Episode_Termination/died"] = died_count
        extras["Episode_Termination/time_out"] = time_out_count
        extras["Metrics/success_rate"] = success_count / total_count
        extras["Metrics/time_to_success"] = success_time
        extras["Metrics/final_ee_goal_distance"] = final_ee_goal_distance.item()
        extras["Metrics/final_ee_to_cube_distance"] = final_ee_to_cube_distance.item()
        # extras["Metrics/final_base_to_goal_distance"] = final_base_distance.item()
        extras["Metrics/final_base_xy_to_cube_distance"] = final_base_xy_to_cube_distance.item()
        extras["Metrics/final_base_minus_ee_z"] = final_base_minus_ee_z.item()
        extras["Metrics/final_base_to_ee_xy_distance"] = final_base_to_ee_xy_distance.item()
        extras["Metrics/manip_peak_joint_vel"] = torch.mean(self._episode_max_manip_vel[env_ids]).item()
        extras["Metrics/manip_peak_joint_acc"] = torch.mean(self._episode_max_manip_acc[env_ids]).item()

        self._episode_max_manip_vel[env_ids] = 0.0
        self._episode_max_manip_acc[env_ids] = 0.0
        self.extras["log"] = extras

        self._robot.reset(env_ids)
        self._cube.reset(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._manip_actions[env_ids] = 0.0
        self._previous_manip_actions[env_ids] = 0.0
        self._success_hold_steps[env_ids] = 0
        self._success[env_ids] = False
        self._distance_init_pending[env_ids] = True

        cube_xy = torch.empty((len(env_ids), 2), device=self.device).uniform_(
            -self.cfg.cube_xy_position_range, self.cfg.cube_xy_position_range
        )
        cube_pos_w = torch.zeros((len(env_ids), 3), device=self.device)
        cube_pos_w[:, :2] = cube_xy + self._terrain.env_origins[env_ids, :2]
        cube_pos_w[:, 2] = self.cfg.cube_height

        cube_yaw = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
        zeros = torch.zeros_like(cube_yaw)
        cube_quat_w = quat_from_euler_xyz(zeros, zeros, cube_yaw)

        cube_state = self._cube.data.default_root_state[env_ids].clone()
        cube_state[:, 0:3] = cube_pos_w
        cube_state[:, 3:7] = cube_quat_w
        cube_state[:, 7:] = 0.0
        self._cube.write_root_pose_to_sim(cube_state[:, :7], env_ids)
        self._cube.write_root_velocity_to_sim(cube_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        joint_pos_limits = self._robot.data.soft_joint_pos_limits[env_ids][:, self._manip_joint_ids_tensor]
        joint_vel_limits = self._robot.data.soft_joint_vel_limits[env_ids][:, self._manip_joint_ids_tensor]
        joint_vel_limits = torch.clamp(joint_vel_limits, min=1.0e-3, max=50.0)
        rand_joint_pos = joint_pos_limits[..., 0] + torch.rand_like(joint_pos_limits[..., 0]) * (
            joint_pos_limits[..., 1] - joint_pos_limits[..., 0]
        )
        rand_joint_vel = (
            torch.empty_like(joint_vel_limits).uniform_(-1.0, 1.0)
            * joint_vel_limits
            * self.cfg.manip_reset_velocity_ratio
        )
        joint_pos[:, self._manip_joint_ids_tensor] = rand_joint_pos
        joint_vel[:, self._manip_joint_ids_tensor] = rand_joint_vel
        self._manip_prev_joint_vel[env_ids] = rand_joint_vel

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "cube_pos_visualizer"):
                cube_marker_cfg = CUBOID_MARKER_CFG.copy()
                cube_marker_cfg.markers["cuboid"].size = (0.03, 0.03, 0.03)
                cube_marker_cfg.prim_path = "/Visuals/Command/cube_position"
                self.cube_pos_visualizer = VisualizationMarkers(cube_marker_cfg)

            if not hasattr(self, "ee_pos_visualizer"):
                ee_marker_cfg = CUBOID_MARKER_CFG.copy()
                ee_marker_cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
                ee_marker_cfg.prim_path = "/Visuals/Command/ee_position"
                self.ee_pos_visualizer = VisualizationMarkers(ee_marker_cfg)

            if not hasattr(self, "ee_goal_pos_visualizer"):
                ee_goal_marker_cfg = CUBOID_MARKER_CFG.copy()
                ee_goal_marker_cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)
                ee_goal_marker_cfg.prim_path = "/Visuals/Command/ee_goal_position"
                self.ee_goal_pos_visualizer = VisualizationMarkers(ee_goal_marker_cfg)

            self.cube_pos_visualizer.set_visibility(True)
            self.ee_pos_visualizer.set_visibility(True)
            self.ee_goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "cube_pos_visualizer"):
                self.cube_pos_visualizer.set_visibility(False)
            if hasattr(self, "ee_pos_visualizer"):
                self.ee_pos_visualizer.set_visibility(False)
            if hasattr(self, "ee_goal_pos_visualizer"):
                self.ee_goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        del event
        cube_pos_w = self._cube.data.root_pos_w
        ee_goal_pos_w = self._compute_ee_goal_pos_w()
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_idx]
        self.cube_pos_visualizer.visualize(cube_pos_w)
        self.ee_pos_visualizer.visualize(ee_pos_w)
        self.ee_goal_pos_visualizer.visualize(ee_goal_pos_w)
