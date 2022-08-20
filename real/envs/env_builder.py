# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os

import numpy as np

from real.envs import locomotion_gym_config, locomotion_gym_env
from real.envs.env_wrappers import imitation_wrapper_env
from real.envs.env_wrappers import \
    observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from real.envs.env_wrappers import (reset_task, simple_openloop,
                                    trajectory_generator_wrapper_env)
from real.envs.sensors import (environment_sensors, robot_sensors,
                               sensor_wrappers)
from real.robots import a1, a1_robot, robot_config

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


def build_imitation_env():

    curriculum_episode_length_start = curriculum_episode_length_end = 256

    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = False
    sim_params.allow_knee_contact = True
    sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION
    sim_params.num_action_repeat = 33
    sim_params.enable_action_filter = False

    gym_config = locomotion_gym_config.LocomotionGymConfig(
        simulation_parameters=sim_params)

    robot_kwargs = {"self_collision_enabled": False}
    robot_class = a1_robot.A1Robot
    robot_kwargs["reset_func_name"] = "_SafeJointsReset"
    robot_kwargs["velocity_source"] = a1.VelocitySource.IMU_FOOT_CONTACT

    num_motors = a1.NUM_MOTORS
    traj_gen = simple_openloop.A1PoseOffsetGenerator(
        action_limit=np.array(
            [0.802851455917, 4.18879020479, -0.916297857297] * 4) -
        np.array([0, 0.9, -1.8] * 4))

    sensors = [
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.MotorAngleSensor(
                num_motors=num_motors),
            num_history=3),
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=robot_sensors.IMUSensor(), num_history=3),
        sensor_wrappers.HistoricSensorWrapper(
            wrapped_sensor=environment_sensors.LastActionSensor(
                num_actions=num_motors),
            num_history=3)
    ]

    task = reset_task.ResetTask()
    randomizers = []

    env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                              robot_class=robot_class,
                                              robot_kwargs=robot_kwargs,
                                              env_randomizers=randomizers,
                                              robot_sensors=sensors,
                                              task=task)

    env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
        env, trajectory_generator=traj_gen)

    env = imitation_wrapper_env.ImitationWrapperEnv(
        env,
        episode_length_start=curriculum_episode_length_start,
        episode_length_end=curriculum_episode_length_end,
        curriculum_steps=2000000,
        num_parallel_envs=1)
    return env
