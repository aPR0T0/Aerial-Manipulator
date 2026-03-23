# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from .pose_reach_env import PoseReachEnv, PoseReachEnvCfg


@configclass
class OptimalPoseReachEnvCfg(PoseReachEnvCfg):
    action_space = 6

    manip_reset_velocity_ratio = 0.5
    manip_velocity_limit_ratio = 1.0

    lin_vel_reward_scale = -0.2
    ang_vel_reward_scale = -0.05
    distance_to_goal_reward_scale = 35.0
    distance_progress_reward_scale = 120.0
    time_penalty_reward_scale = -2.0
    tilt_reward_scale = -0.3
    manip_joint_vel_reward_scale = -0.01
    manip_action_rate_reward_scale = -0.03
    success_bonus_reward = 15.0

    success_tolerance = 0.2
    success_lin_vel_threshold = 0.35
    success_ang_vel_threshold = 0.65


class OptimalPoseReachEnv(PoseReachEnv):
    cfg: OptimalPoseReachEnvCfg

    def __init__(self, cfg: OptimalPoseReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._manip_actions = torch.zeros(self.num_envs, self._num_manip_joints, device=self.device)
        self._previous_manip_actions = torch.zeros(self.num_envs, self._num_manip_joints, device=self.device)
        self._previous_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "distance_progress",
                "distance_to_goal",
                "time_penalty",
                "lin_vel",
                "ang_vel",
                "tilt",
                "manip_joint_vel",
                "manip_action_rate",
                "success_bonus",
            ]
        }

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

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        distance_progress = self._previous_distance_to_goal - distance_to_goal

        lin_vel_sq = torch.sum(torch.square(root_lin_vel_b), dim=1)
        ang_vel_sq = torch.sum(torch.square(root_ang_vel_b), dim=1)
        tilt_error = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        manip_joint_vel = torch.sum(
            torch.square(self._robot.data.joint_vel[:, self._manip_joint_ids_tensor]),
            dim=1,
        )
        manip_action_rate = torch.sum(torch.square(self._manip_actions - self._previous_manip_actions), dim=1)

        lin_speed = torch.linalg.norm(root_lin_vel_b, dim=1)
        ang_speed = torch.linalg.norm(root_ang_vel_b, dim=1)
        self._success = torch.logical_and(
            distance_to_goal < self.cfg.success_tolerance,
            torch.logical_and(
                lin_speed < self.cfg.success_lin_vel_threshold,
                ang_speed < self.cfg.success_ang_vel_threshold,
            ),
        )

        rewards = {
            "distance_progress": distance_progress * self.cfg.distance_progress_reward_scale,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "time_penalty": torch.full_like(distance_to_goal, self.cfg.time_penalty_reward_scale * self.step_dt),
            "lin_vel": lin_vel_sq * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_sq * self.cfg.ang_vel_reward_scale * self.step_dt,
            "tilt": tilt_error * self.cfg.tilt_reward_scale * self.step_dt,
            "manip_joint_vel": manip_joint_vel * self.cfg.manip_joint_vel_reward_scale * self.step_dt,
            "manip_action_rate": manip_action_rate * self.cfg.manip_action_rate_reward_scale * self.step_dt,
            "success_bonus": self._success.float() * self.cfg.success_bonus_reward,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value

        self._previous_distance_to_goal[:] = distance_to_goal
        self._previous_manip_actions[:] = self._manip_actions

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)

        lin_speed = torch.linalg.norm(root_lin_vel_b, dim=1)
        ang_speed = torch.linalg.norm(root_ang_vel_b, dim=1)
        self._success = torch.logical_and(
            distance_to_goal < self.cfg.success_tolerance,
            torch.logical_and(
                lin_speed < self.cfg.success_lin_vel_threshold,
                ang_speed < self.cfg.success_ang_vel_threshold,
            ),
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        terminated = torch.logical_or(died, self._success)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        assert env_ids is not None

        successful_envs = self._success[env_ids]

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
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
        extras["Metrics/time_to_goal"] = success_time
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/manip_peak_joint_vel"] = torch.mean(self._episode_max_manip_vel[env_ids]).item()
        extras["Metrics/manip_peak_joint_acc"] = torch.mean(self._episode_max_manip_acc[env_ids]).item()

        self._episode_max_manip_vel[env_ids] = 0.0
        self._episode_max_manip_acc[env_ids] = 0.0
        self.extras["log"] = extras

        self._robot.reset(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._manip_actions[env_ids] = 0.0
        self._previous_manip_actions[env_ids] = 0.0
        self._success[env_ids] = False

        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

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

        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._previous_distance_to_goal[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - default_root_state[:, :3],
            dim=1,
        )

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
