import time

import gym
import gym.spaces
import numpy as np
from absl import logging
from dm_control.utils import rewards
from motion_imitation.envs import env_builder
from motion_imitation.robots import a1, a1_robot, robot_config
from motion_imitation.utilities import pose3d
from safe_outdoor import resetters


def get_run_reward(x_velocity: float, move_speed: float,
                   cos_pitch_cos_roll: float, terminate_pitch_roll_deg: float):
    termination = np.cos(np.deg2rad(terminate_pitch_roll_deg))
    upright = rewards.tolerance(cos_pitch_cos_roll,
                                bounds=(termination, float('inf')),
                                sigmoid='linear',
                                margin=termination + 1,
                                value_at_margin=0)

    forward = rewards.tolerance(x_velocity,
                                bounds=(move_speed, 2 * move_speed),
                                margin=move_speed,
                                value_at_margin=0,
                                sigmoid='linear')

    return upright * forward  # [0, 1] => [0, 10]


class A1Real(gym.Env):
    # index 1: .5 makes it go back, .9 makes it go forward
    def __init__(self,
                 real_robot: bool = True,
                 render: bool = False,
                 zero_action: np.ndarray = np.asarray([0.05, 0.9, -1.8] * 4),
                 action_offset: np.ndarray = np.asarray([0.2, 0.4, 0.4] * 4),
                 use_onboard=True):
        if real_robot:
            logging.info(
                "WARNING: this code executes low-level control on the robot.")
            input("Press enter to continue...")
        self.env = env_builder.build_imitation_env(
            motion_files=[""],
            num_parallel_envs=1,
            mode='test',
            enable_randomizer=False,
            enable_rendering=render,
            use_real_robot=real_robot,
            reset_at_current_position=False,
            realistic_sim=True)
        self._use_onboard = use_onboard
        if not self._use_onboard:
            import openvr
            self.vr_system = openvr.init(openvr.VRApplication_Other)
            devices = []
            self.tracker_index = None
            for device_index in range(openvr.k_unMaxTrackedDeviceCount):
                is_connected = self.vr_system.isTrackedDeviceConnected(
                    device_index)
                if not is_connected:
                    continue

                prop_type = openvr.Prop_RenderModelName_String
                device_name = self.vr_system.getStringTrackedDeviceProperty(
                    device_index, prop_type)
                if "vr_tracker_vive_3_0" in device_name:
                    self.tracker_index = device_index
                    print(device_index)
                    print(is_connected)
                    print(device_name)

        self.resetter = resetters.GetupResetter(self.env,
                                                real_robot,
                                                standing_pose=zero_action)
        # self.resetter = lambda: None
        # Move the motors slowly to initial position
        self.original_kps = self.env.robot._motor_kps.copy()
        self.original_kds = self.env.robot._motor_kds.copy()

        min_actions = zero_action - action_offset
        max_actions = zero_action + action_offset

        self.action_space = gym.spaces.Box(min_actions, max_actions)
        self._estimated_velocity = np.zeros(3)
        self._reset_var()

        obs = self.observation()

        self.observation_space = gym.spaces.Box(float("-inf"),
                                                float("inf"),
                                                shape=obs.shape,
                                                dtype=np.float32)

    def _reset_var(self):
        self.prev_action = np.zeros_like(self.action_space.low)
        self.prev_qpos = None
        self._last_timestamp = time.time()
        self._prev_pose = None

    def reset(self):
        self.env._robot.SetMotorGains(kp=self.original_kps,
                                      kd=self.original_kds)
        # self.env._robot.SetTimeSteps(33, .001)
        self.resetter()
        # print("Already ran getup controller")
        self.env._robot.SetMotorGains(kp=[60.0] * 12, kd=[4.0] * 12)
        # self.env._robot.SetTimeSteps(50, .001)
        self._reset_var()
        input("Press Enter to Continue")
        # print("taking 10 steps to maybe help velocity estimation...")
        # for i in range(0):
        # action = np.zeros(12)
        # self.env.step(action)
        # print("done!")
        self._current_direction = 'forward'
        return self.observation()

    def _get_imu(self):
        rpy = self.env._robot.GetBaseRollPitchYaw()
        # print(rpy)
        drpy = self.env._robot.GetBaseRollPitchYawRate()

        assert len(rpy) >= 3, rpy
        assert len(drpy) >= 3, drpy

        channels = ["R", "P", "dR", "dP", "dY"]
        observations = np.zeros(len(channels))
        for i, channel in enumerate(channels):
            if channel == "R":
                observations[i] = rpy[0]
            if channel == "Rcos":
                observations[i] = np.cos(rpy[0])
            if channel == "Rsin":
                observations[i] = np.sin(rpy[0])
            if channel == "P":
                observations[i] = rpy[1]
            if channel == "Pcos":
                observations[i] = np.cos(rpy[1])
            if channel == "Psin":
                observations[i] = np.sin(rpy[1])
            if channel == "Y":
                observations[i] = rpy[2]
            if channel == "Ycos":
                observations[i] = np.cos(rpy[2])
            if channel == "Ysin":
                observations[i] = np.sin(rpy[2])
            if channel == "dR":
                observations[i] = drpy[0]
            if channel == "dP":
                observations[i] = drpy[1]
            if channel == "dY":
                observations[i] = drpy[2]
        return observations

    def _compute_delta_time(self, current_time):
        delta_time_s = current_time - self._last_timestamp
        self._last_timestamp = current_time
        return delta_time_s

    def _update_vel(self, delta_time_s):
        if self._use_onboard:
            self._estimated_velocity = self.env._robot.GetBaseVelocity()
        else:
            poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0.0,
                openvr.k_unMaxTrackedDeviceCount)
            vr_pose = np.ctypeslib.as_array(
                poses[self.tracker_index].mDeviceToAbsoluteTracking[:],
                shape=(3, 4))
            vr_rmat = vr_pose[:3, :3]

            if self._prev_pose is not None:
                # print('prev pose', self._prev_pose)
                self._estimated_velocity = np.linalg.inv(vr_rmat) @ (
                    vr_pose[:, -1] - self._prev_pose)
                # print('estimated velocity', self._estimated_velocity)
                self._estimated_velocity /= delta_time_s
                self._estimated_velocity = np.clip(self._estimated_velocity,
                                                   -1, 1)
            # if np.all(self._estimated_velocity == np.zeros(3)):
            # print("something went wrong")
            # else:
            self._prev_pose = vr_pose[:, -1]

    def observation(self):
        delta_time_s = self._compute_delta_time(time.time())

        self.env._robot.ReceiveObservation()
        qpos = self.env._robot.GetMotorAngles()
        qvel = self.env._robot.GetMotorVelocities()
        imu = self._get_imu()

        self._update_vel(delta_time_s)
        vel = self._estimated_velocity.copy()
        new_vel = np.array([vel[1], vel[0], vel[2]])

        self._foot_contacts = self.env._robot.GetFootContacts().astype(
            np.float32)

        return np.concatenate(
            [qpos, qvel, new_vel, imu, self.prev_action,
             self._foot_contacts]).astype(np.float32)

    def roll_reward(self):
        up = np.array([0, 0, 1])

        root_rot = self.env.robot.GetTrueBaseOrientation()
        root_up = pose3d.QuaternionRotatePoint(up, root_rot)
        cos_dist = up.dot(root_up)

        torque_penalty = np.sum(np.abs(self.env.robot.GetMotorTorques()))
        # print("torque penalty", torque_penalty)
        r_roll = (0.5 * cos_dist + 0.5)**2

        reward = r_roll

        return reward

    def step(self, action):
        assert self.env._robot._action_repeat == 50
        self.env._robot.Step(action, robot_config.MotorControlMode.POSITION)
        # print(self.env._robot._timesteps)

        obs = self.observation()
        self.prev_action[:] = action

        accel_velocity = self.env._robot.GetBaseVelocity()
        velocity = self._estimated_velocity.copy()
        roll, pitch, yaw = self.env._robot.GetTrueBaseRollPitchYaw()
        drpy = self.env._robot.GetBaseRollPitchYawRate()

        if self._use_onboard:
            lin_vel = self._estimated_velocity[0]
        else:
            lin_vel = velocity[1]
        target_vel = .5

        term_rad_roll = term_rad_pitch = np.deg2rad(30)

        # normalization_multiplier = (np.cos(pitch) - np.cos(term_rad_pitch)) / (
        #     1 - np.cos(term_rad_pitch))
        # normalization_multiplier = np.clip(normalization_multiplier, 0, 1)
        # normalization_multiplier = normalization_multiplier**2

        # reward = target_vel - np.abs(lin_vel * normalization_multiplier -
        #                              target_vel)
        # reward /= target_vel
        # # # reward += -0.1 * np.abs(drpy).mean()
        # # # reward *= np.exp(-np.abs(drpy).mean())
        # reward -= 0.1 * np.abs(drpy[-1])
        #

        # reward *= 10.0
        # reward = get_run_reward(lin_vel*np.cos(pitch), target_vel, 1.0, 30)

        reward = rewards.tolerance(lin_vel * np.cos(pitch),
                                   bounds=(target_vel, 2 * target_vel),
                                   margin=2 * target_vel,
                                   value_at_margin=0,
                                   sigmoid='linear')
        reward -= 0.1 * np.abs(drpy[-1])
        reward *= max(self._foot_contacts)
        reward *= 10.0

        # reward = get_run_reward(lin_vel, target_vel, np.cos(pitch)*np.cos(roll), 30)

        qvel = self.env._robot.GetMotorVelocities()
        torque = self.env._robot._observed_motor_torques
        energy = (qvel * torque).sum()
        # reward += -0.002 * energy

        if abs(roll) > term_rad_roll or abs(
                pitch) > term_rad_pitch or not self.env._robot._is_safe:
            print("fell")
            done = True
        else:
            done = False

        info = {
            'velocity':
            velocity,
            'acc_velocity':
            accel_velocity,
            'raw_acc':
            self.env._robot._velocity_estimator._raw_acc.copy(),
            'calibrated_acc':
            self.env._robot._velocity_estimator._calibrated_acc.copy(),
            'leg_vels':
            np.array(self.env._robot._velocity_estimator._observed_velocities),
            'rpy':
            np.array([roll, pitch, yaw]),
            'jangles':
            self.env._robot.GetMotorAngles(),
            'energy':
            energy,
            'x_vel':
            lin_vel * np.exp(-np.abs(drpy[1:]).mean()),
            'dr':
            drpy[0],
            'dp':
            drpy[1],
            'dy':
            drpy[2]
        }
        if not self._use_onboard:
            info['position'] = self._prev_pose
        return obs, reward, done, info
