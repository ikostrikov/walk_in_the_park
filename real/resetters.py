"""For resetting the robot to a ready pose."""

import inspect
import os

import numpy as np

import real.sac_dev.learning.sac_agent as sac_agent
import tensorflow as tf
from real.envs.env_wrappers import reset_task
from real.robots import robot_config

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

GOAL_ANGLES = np.asarray([
    0.00842845, 0.67221212, -1.4888885, 0.06030566, 0.72687078, -1.47780502,
    -0.01285338, 0.77798939, -1.44194186, 0.07631981, 0.68932199, -1.4569447
])


def orientation_terminal_condition(env):
    """Returns True if the robot is close to upright."""
    cutoff = np.pi / 4
    roll, pitch, _ = env.robot.GetBaseRollPitchYaw()
    return abs(roll) < cutoff and abs(pitch) < cutoff


class RewardPlateau(object):
    def __init__(self, n=5, delta=.001, lowbar=0.8):
        self.n = n
        self.delta = delta
        self.last_reward = float('inf')
        self.count_same = 0
        self.lowbar = lowbar

    def __call__(self, env):
        reward = env._task.reward(env)
        if reward > self.lowbar and abs(reward -
                                        self.last_reward) < self.delta:
            self.count_same += 1
        else:
            self.count_same = 0
        self.last_reward = reward
        return self.count_same >= self.n


class GetupResetter(object):
    """ Single-policy resetter that first rolls over, then stands up."""
    def __init__(self, env, using_real_robot, standing_pose):
        self._env = env
        upright = lambda env: env._task.reward(
            env) > .95 or env.env_step_counter > 150
        self._reset_task = reset_task.ResetTask(
            terminal_conditions=(upright, RewardPlateau()), real_robot=True)
        old_task = self._env.task
        self._env.set_task(self._reset_task)

        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(graph=self._graph)
        agent_configs = {
            "actor_net": "fc_2layers_512units",
            "critic_net": "fc_2layers_512units",
            "actor_stepsize": 0.0003,
            "actor_init_output_scale": 0.01,
            "actor_batch_size": 256,
            "actor_steps": 128,
            "action_std": 0.15,
            "critic_stepsize": 0.0003,
            "critic_batch_size": 256,
            "critic_steps": 256,
            "discount": 0.99,
            "samples_per_iter": 512,
            "replay_buffer_size": int(1e6),
            "normalizer_samples": 300000,
            "enable_val_norm": False,
            "num_action_samples": 1,
            "tar_stepsize": 5e-3,
            "steps_per_tar_update": 1,
            "init_samples": 25000
        }
        self._reset_model = sac_agent.SACAgent(env=self._env,
                                               sess=self._sess,
                                               **agent_configs)
        self._reset_model.load_model(
            os.path.join(parentdir, "real/model-reset.ckpt"))
        self._action_filter = self._env.robot._BuildActionFilter(highcut=[3])
        self._env.set_task(old_task)
        self._standing_pose = standing_pose

    def _run_single_episode(self, task, policy):
        old_task = self._env.task
        self._env.set_task(task)
        obs = self._env.reset()
        done = False
        self._env.robot.running_reset_policy = True
        while not done:
            action = policy(obs)
            obs, reward, done, _ = self._env.step(action)

        self._env.robot.running_reset_policy = False
        self._env.set_task(old_task)

    def _go_to_standing_pose(self):
        env = self._env
        env.robot.ReceiveObservation()
        if env._is_render:
            env._pybullet_client.configureDebugVisualizer(
                env._pybullet_client.COV_ENABLE_RENDERING, 1)
        current_motor_angle = np.array(self._env.robot.GetMotorAngles())
        desired_motor_angle = self._standing_pose + np.clip(
            np.random.normal(size=12) * .05, -.1, .1)

        N = 100
        for t in range(N):
            blend_ratio = np.minimum(t / N, 1)
            action = (
                1 - blend_ratio
            ) * current_motor_angle + blend_ratio * desired_motor_angle
            self._env.robot.Step(action,
                                 robot_config.MotorControlMode.POSITION)
        self._env.robot.HoldCurrentPose()
        print("finished standing")

    def __call__(self):
        self._env.robot.ReceiveObservation()
        enable_action_filter = self._env.robot._enable_action_filter
        self._env.robot._enable_action_filter = True
        action_filter = self._env.robot._action_filter
        self._env.robot._action_filter = self._action_filter
        self._env.set_time_step(33)

        for i in range(1, 5):
            print("Reset attempt {}/5".format(i))
            try:
                self._run_single_episode(
                    self._reset_task,
                    lambda x: self._reset_model.sample_action(x, True)[0])
                roll, pitch, _ = self._env.robot.GetTrueBaseRollPitchYaw()
                angles = self._env.robot.GetMotorAngles()

                if abs(roll) < np.deg2rad(30) and abs(pitch) < np.deg2rad(
                        30) and np.linalg.norm(angles - GOAL_ANGLES) < .20:
                    break
            except:
                continue

        self._go_to_standing_pose()
        self._env.robot._enable_action_filter = enable_action_filter
        self._env.robot._action_filter = action_filter
        self._env.set_time_step(50)
        self._env.robot.Reset(reload_urdf=False,
                              default_motor_angles=None,
                              reset_time=0.0)
        self._env.robot.ReceiveObservation()
