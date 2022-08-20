import enum
import time

import numpy as np


class Terminate(enum.Enum):
    Null = 0
    Fail = 1


class RLPath(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.max_torques = []

        self.terminate = Terminate.Null

        self.clear()

        return

    def pathlength(self):
        return len(self.actions)

    def is_valid(self):
        valid = True
        l = self.pathlength()

        valid &= len(self.states) == l + 1
        valid &= len(self.actions) == l
        valid &= len(self.logps) == l
        valid &= len(self.rewards) == l
        valid &= len(self.max_torques) == l
        valid |= (l == 0)

        return valid

    def check_vals(self):
        for key, vals in vars(self).items():
            if type(vals) is list and len(vals) > 0:
                for v in vals:
                    if not np.isfinite(v).all():
                        return False
        return True

    def clear(self):
        for key, vals in vars(self).items():
            if type(vals) is list:
                vals.clear()

        self.terminate = Terminate.Null
        return

    def calc_return(self):
        return sum(self.rewards)

    def terminated(self):
        return self.terminate == Terminate.Null

    def calc_max_torque(self):
        return max(self.max_torques)
