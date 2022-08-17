import os
from collections import deque
from functools import cached_property
from typing import Optional

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.utils.transformations import quat_to_euler
from dm_env import specs

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'a1')
_A1_XML_PATH = os.path.join(ASSETS_DIR, 'xml', 'a1.xml')


class A1Observables(base.WalkerObservables):

    @composer.observable
    def joints_vel(self):
        return observable.MJCFFeature('qvel', self._entity.observable_joints)

    @composer.observable
    def prev_action(self):
        return observable.Generic(lambda _: self._entity.prev_action)

    @property
    def proprioception(self):
        return ([self.joints_pos, self.joints_vel] +
                self._collect_from_attachments('proprioception'))

    @composer.observable
    def sensors_velocimeter(self):
        return observable.Generic(
            lambda physics: self._entity.get_velocity(physics))

    @property
    def kinematic_sensors(self):
        return ([
            self.sensors_gyro, self.sensors_velocimeter, self.sensors_framequat
        ] + self._collect_from_attachments('kinematic_sensors'))


class A1(base.Walker):
    _INIT_QPOS = np.asarray([0.05, 0.7, -1.4] * 4)
    _QPOS_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)
    # _INIT_QPOS = np.asarray([-0.1, 0.5, -1.4] * 4)
    """A composer entity representing a Jaco arm."""

    def _build(self,
               name: Optional[str] = None,
               action_history: int = 1,
               learn_kd: bool = False):
        """Initializes the JacoArm.

    Args:
      name: String, the name of this robot. Used as a prefix in the MJCF name
        name attributes.
    """
        self._mjcf_root = mjcf.from_path(_A1_XML_PATH)
        if name:
            self._mjcf_root.model = name
        # Find MJCF elements that will be exposed as attributes.
        self._root_body = self._mjcf_root.find('body', 'trunk')
        self._root_body.pos[-1] = 0.125

        self._joints = self._mjcf_root.find_all('joint')

        self._actuators = self.mjcf_model.find_all('actuator')

        # Check that joints and actuators match each other.
        assert len(self._joints) == len(self._actuators)
        for joint, actuator in zip(self._joints, self._actuators):
            assert joint == actuator.joint

        self.kp = 60
        if learn_kd:
            self.kd = None
        else:
            self.kd = 10

        self._prev_actions = deque(maxlen=action_history)
        self.initialize_episode_mjcf(None)

    def initialize_episode_mjcf(self, random_state):
        self._prev_actions.clear()
        for _ in range(self._prev_actions.maxlen):
            self._prev_actions.append(self._INIT_QPOS)

    @cached_property
    def action_spec(self):
        minimum = []
        maximum = []
        for joint_, actuator in zip(self.joints, self.actuators):
            joint = actuator.joint
            assert joint == joint_

            minimum.append(joint.range[0])
            maximum.append(joint.range[1])

        if self.kd is None:
            minimum.append(-1.0)
            maximum.append(1.0)

        return specs.BoundedArray(
            shape=(len(minimum), ),
            dtype=np.float32,
            minimum=minimum,
            maximum=maximum,
            name='\t'.join([actuator.name for actuator in self.actuators]))

    @cached_property
    def ctrllimits(self):
        minimum = []
        maximum = []
        for actuator in self.actuators:
            minimum.append(actuator.ctrlrange[0])
            maximum.append(actuator.ctrlrange[1])

        return minimum, maximum

    def apply_action(self, physics, desired_qpos, random_state):
        # Updates previous action.
        self._prev_actions.append(desired_qpos.copy())

        joints_bind = physics.bind(self.joints)
        qpos = joints_bind.qpos
        qvel = joints_bind.qvel

        if self.kd is None:
            min_kd = 1
            max_kd = 10

            kd = (desired_qpos[-1] + 1) / 2 * (max_kd - min_kd) + min_kd
            desired_qpos = desired_qpos[:-1]
        else:
            kd = self.kd

        action = self.kp * (desired_qpos - qpos) - kd * qvel
        minimum, maximum = self.ctrllimits
        action = np.clip(action, minimum, maximum)

        physics.bind(self.actuators).ctrl = action

    def _build_observables(self):
        return A1Observables(self)

    @property
    def root_body(self):
        return self._root_body

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def observable_joints(self):
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root

    @property
    def prev_action(self):
        return np.concatenate(self._prev_actions)

    def get_roll_pitch_yaw(self, physics):
        quat = physics.bind(self.mjcf_model.sensor.framequat).sensordata
        return np.rad2deg(quat_to_euler(quat))

    def get_velocity(self, physics):
        velocimeter = physics.bind(self.mjcf_model.sensor.velocimeter)
        return velocimeter.sensordata
