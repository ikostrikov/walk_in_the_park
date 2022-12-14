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
"""Contains the terminal conditions for imitation task."""

from __future__ import absolute_import, division, print_function

import numpy as np

from pybullet_utils import transformations
from utilities import motion_util, pose3d


def imitation_terminal_condition(env,
                                 dist_fail_threshold=1.0,
                                 rot_fail_threshold=0.5 * np.pi):
    """A terminal condition for motion imitation task.

  Args:
    env: An instance of MinitaurGymEnv
    dist_fail_threshold: Max distance the simulated character's root is allowed
      to drift from the reference motion before the episode terminates.
    rot_fail_threshold: Max rotational difference between simulated character's
      root and the reference motion's root before the episode terminates.

  Returns:
    A boolean indicating if episode is over.
  """

    task = env.task

    motion_over = task.is_motion_over()
    foot_links = env.robot.GetFootLinkIDs()
    ground = env.get_ground()

    contact_fall = False
    # sometimes the robot can be initialized with some ground penetration
    # so do not check for contacts until after the first env step.
    if env.env_step_counter > 0:
        robot_ground_contacts = env.pybullet_client.getContactPoints(
            bodyA=env.robot.quadruped, bodyB=ground)

        for contact in robot_ground_contacts:
            if contact[3] not in foot_links:
                contact_fall = True
                break

    root_pos_ref = task.get_ref_base_position()
    root_rot_ref = task.get_ref_base_rotation()
    root_pos_robot = env.robot.GetBasePosition()
    root_rot_robot = env.robot.GetBaseOrientation()
    # print("root_pos_ref", root_pos_ref)
    # print("root_pos_robot", root_pos_robot)

    root_pos_diff = np.array(root_pos_ref) - np.array(root_pos_robot)
    # root_pos_fail = False
    root_pos_fail = (root_pos_diff.dot(root_pos_diff) >
                     dist_fail_threshold * dist_fail_threshold)

    root_rot_diff = transformations.quaternion_multiply(
        np.array(root_rot_ref),
        transformations.quaternion_conjugate(np.array(root_rot_robot)))
    root_rot_diff /= np.linalg.norm(root_rot_diff)
    _, root_rot_diff_angle = pose3d.QuaternionToAxisAngle(root_rot_diff)
    root_rot_diff_angle = motion_util.normalize_rotation_angle(
        root_rot_diff_angle)
    root_rot_fail = (np.abs(root_rot_diff_angle) > rot_fail_threshold)
    # print("motion_over", motion_over)
    # print("contact_fall", contact_fall)
    # print("root_pos_fail", root_pos_fail)
    # print("root_rot_fail", root_rot_fail)
    done = motion_over \
        or contact_fall \
        or root_pos_fail \
        or root_rot_fail

    if done:
        print("motion over", motion_over)
        print("contact fall", contact_fall)
        print("root pos fail", root_pos_fail)
        print("root rot fail", root_rot_fail)
    # done = motion_over
    return done
