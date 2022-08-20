import numpy as np

import tensorflow as tf


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(
        isinstance(a, int)
        for a in out), "shape function assumes that shape is fully known"
    return out


def intprod(x):
    return int(np.prod(x))


def numel(x):
    n = intprod(var_shape(x))
    return n


class SetFromFlat(object):
    def __init__(self, sess, var_list):
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self._sess = sess
        self.theta_ph = tf.compat.v1.placeholder(var_list[0].dtype,
                                                 [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(
                tf.compat.v1.assign(
                    v, tf.reshape(self.theta_ph[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self._sess.run(self.op, feed_dict={self.theta_ph: theta})


class GetFlat(object):
    def __init__(self, sess, var_list):
        self._sess = sess
        self.op = tf.concat(
            axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return self._sess.run(self.op)
