"""Run inference with a saved policy."""

import numpy as np

import tensorflow.compat.v1 as tf


class SavedPolicy(object):
    """Load a policy saved with sac_agent.py."""
    def __init__(self, export_dir):
        """Constructor.

    Args:
      export_dir: Directory and model identifier for reloading policy. Must
        contain a metagraph file, checkpoint index, and data. E.g. if
        initialized as SavedPolicy('policies/model.ckpt'), then 'policies/'
        must contain 'model.ckpt.meta', 'model.ckpt.index', and
        'model.ckpt.data-00000-of-00001'.
    """
        self._sess = tf.Session()
        saver = tf.train.import_meta_graph("{}.meta".format(export_dir))
        saver.restore(self._sess, export_dir)

    def __call__(self, state):
        """Runs inference with the policy.

    Makes strong assumptions about the names of tensors in the policy.

    Args:
      state: Array of floats, the input to the policy. It is assumed that only
        one state is being passed in and that the state is 1-dimensional.

    Returns:
      Action from the policy as a 1D np array.
    """
        state = np.reshape(state, (1, -1))
        action = self._sess.run("add_1:0", feed_dict={"s:0": state})
        return np.reshape(action, (-1, ))
