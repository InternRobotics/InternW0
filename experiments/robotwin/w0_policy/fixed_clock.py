"""Execute absolute interval-end joint commands on the training clock.

This replaces TOPP time parameterization. A model action is one fixed period;
drive targets are interpolated from the previous target to its endpoint. This
is a different controller protocol from official RoboTwin TOPP evaluation.
"""
import math

import numpy as np


def simulation_steps(target_fps, timestep):
    if not math.isfinite(target_fps) or target_fps <= 0 or not math.isfinite(timestep) or timestep <= 0:
        raise ValueError('target_fps and simulation timestep must be finite and positive.')
    count = 1.0 / (target_fps * timestep)
    steps = round(count)
    if steps < 1 or not math.isclose(count, steps, rel_tol=0, abs_tol=1e-5):
        raise ValueError('Training control period must be an integer number of simulation steps.')
    return steps


def take_fixed_action(env, action, target_fps):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError('Expected 14 finite absolute joint/gripper commands.')
    if env.take_action_cnt >= env.step_lim or env.eval_success:
        return
    dt = float(env.scene.get_timestep())
    steps = simulation_steps(float(target_fps), dt)
    start = np.asarray(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(), dtype=np.float64)
    if start.shape != (14,) or not np.isfinite(start).all():
        raise ValueError('RoboTwin must expose six joints and one gripper per arm.')
    # Match the environment's normalized gripper command bounds.
    action = action.copy()
    action[[6, 13]] = action[[6, 13]].clip(0, 1)
    if env.eval_video_path is not None:
        env.eval_video_ffmpeg.stdin.write(env._get_eval_video_frame().tobytes())
    env.take_action_cnt += 1
    velocity = (action - start) / (steps * dt)
    for i in range(1, steps + 1):
        point = start + (action - start) * (i / steps)
        env.robot.set_arm_joints(point[:6], velocity[:6], 'left')
        env.robot.set_arm_joints(point[7:13], velocity[7:13], 'right')
        env.robot.set_gripper(float(point[6]), 'left')
        env.robot.set_gripper(float(point[13]), 'right')
        env.scene.step()
        env._update_render()
        if env.check_success():
            env.eval_success = True
            env.get_obs()
            if env.eval_video_path is not None:
                env.eval_video_ffmpeg.stdin.write(env._get_eval_video_frame().tobytes())
            break
    if env.render_freq:
        env.viewer.render()
