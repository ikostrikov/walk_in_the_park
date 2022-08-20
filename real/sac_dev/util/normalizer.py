import copy

import numpy as np

# import sac_dev.util.mpi_util as mpi_util
import tensorflow as tf
from real.sac_dev.util.logger import Logger


class Normalizer(object):
    CHECK_SYNC_COUNT = 50000  # check synchronization after a certain number of entries

    def __init__(self,
                 sess,
                 scope,
                 size,
                 init_mean=None,
                 init_std=None,
                 eps=0.01,
                 clip=np.inf):

        self._sess = sess
        self._scope = scope
        self._eps = eps
        self._clip = clip
        self._mean = np.zeros(size)
        self._std = np.ones(size)
        self._count = 0

        if init_mean is not None:
            if not isinstance(init_mean, np.ndarray):
                assert (size == 1)
                init_mean = np.array([init_mean])

            assert init_mean.size == size, \
            Logger.print('Normalizer init mean shape mismatch, expecting size {:d}, but got {:d}'.format(size, init_mean.size))
            self._mean = init_mean

        if init_std is not None:
            if not isinstance(init_std, np.ndarray):
                assert (size == 1)
                init_std = np.array([init_std])

            assert init_std.size == size, \
            Logger.print('Normalizer init std shape mismatch, expecting size {:d}, but got {:d}'.format(size, init_std.size))
            self._std = init_std

        self._mean_sq = self.calc_mean_sq(self._mean, self._std)

        self._new_count = 0
        self._new_sum = np.zeros_like(self._mean)
        self._new_sum_sq = np.zeros_like(self._mean_sq)

        with tf.compat.v1.variable_scope(self._scope):
            self._build_resource_tf()

        return

    def record(self, x):
        size = self.get_size()
        is_array = isinstance(x, np.ndarray)
        if not is_array:
            assert (size == 1)
            x = np.array([[x]])

        assert x.shape[-1] == size, \
            Logger.print('Normalizer shape mismatch, expecting size {:d}, but got {:d}'.format(size, x.shape[-1]))
        x = np.reshape(x, [-1, size])

        self._new_count += x.shape[0]
        self._new_sum += np.sum(x, axis=0)
        self._new_sum_sq += np.sum(np.square(x), axis=0)
        return

    def update(self):
        new_count = mpi_util.reduce_sum(self._new_count)
        new_sum = mpi_util.reduce_sum(self._new_sum)
        new_sum_sq = mpi_util.reduce_sum(self._new_sum_sq)

        if (new_count > 0):
            new_total = self._count + new_count
            if (self._count // self.CHECK_SYNC_COUNT !=
                    new_total // self.CHECK_SYNC_COUNT):
                assert self._check_synced(), Logger.print(
                    "Normalizer parameters desynchronized")

            new_mean = new_sum / new_count
            new_mean_sq = new_sum_sq / new_count
            w_old = float(self._count) / new_total
            w_new = float(new_count) / new_total

            self._mean = w_old * self._mean + w_new * new_mean
            self._mean_sq = w_old * self._mean_sq + w_new * new_mean_sq
            self._count = new_total
            self._std = self.calc_std(self._mean, self._mean_sq)

            self._new_count = 0
            self._new_sum.fill(0)
            self._new_sum_sq.fill(0)

        self._update_resource_tf()

        return

    def get_size(self):
        return self._mean.size

    def set_mean_std(self, mean, std):
        size = self.get_size()
        is_array = isinstance(mean, np.ndarray) and isinstance(std, np.ndarray)

        if not is_array:
            assert (size == 1)
            mean = np.array([mean])
            std = np.array([std])

        assert len(mean) == size and len(std) == size, \
            Logger.print('Normalizer shape mismatch, expecting size {:d}, but got {:d} and {:d}'.format(size, len(mean), len(std)))

        self._mean = mean
        self._std = std
        self._mean_sq = self.calc_mean_sq(self._mean, self._std)

        self._update_resource_tf()

        return

    def normalize(self, x):
        norm_x = (x - self._mean) / self._std
        norm_x = np.clip(norm_x, -self._clip, self._clip)
        return norm_x

    def unnormalize(self, norm_x):
        x = norm_x * self._std + self._mean
        return x

    def calc_std(self, mean, mean_sq):
        var = mean_sq - np.square(mean)
        # some time floating point errors can lead to small negative numbers
        var = np.maximum(var, 0)
        std = np.sqrt(var)
        std = np.maximum(std, self._eps)
        return std

    def calc_mean_sq(self, mean, std):
        return np.square(std) + np.square(self._mean)

    def load(self):
        count, mean, std = self._sess.run(
            [self._count_tf, self._mean_tf, self._std_tf])
        self._count = count[0]
        self._mean = mean
        self._std = std

        self._mean_sq = self.calc_mean_sq(self._mean, self._std)

        return

    def normalize_tf(self, x):
        norm_x = (x - self._mean_tf) / self._std_tf
        norm_x = tf.clip_by_value(norm_x, -self._clip, self._clip)
        return norm_x

    def unnormalize_tf(self, norm_x):
        x = norm_x * self._std_tf + self._mean_tf
        return x

    def need_update(self):
        return self._new_count > 0

    def _build_resource_tf(self):
        self._count_tf = tf.compat.v1.get_variable(dtype=tf.int32,
                                                   name="count",
                                                   initializer=np.array(
                                                       [self._count],
                                                       dtype=np.int32),
                                                   trainable=False)
        self._mean_tf = tf.compat.v1.get_variable(
            dtype=tf.float32,
            name="mean",
            initializer=self._mean.astype(np.float32),
            trainable=False)
        self._std_tf = tf.compat.v1.get_variable(dtype=tf.float32,
                                                 name="std",
                                                 initializer=self._std.astype(
                                                     np.float32),
                                                 trainable=False)

        self._count_ph = tf.compat.v1.get_variable(dtype=tf.int32,
                                                   name="count_ph",
                                                   shape=[1])
        self._mean_ph = tf.compat.v1.get_variable(dtype=tf.float32,
                                                  name="mean_ph",
                                                  shape=self._mean.shape)
        self._std_ph = tf.compat.v1.get_variable(dtype=tf.float32,
                                                 name="std_ph",
                                                 shape=self._std.shape)

        self._update_op = tf.group(self._count_tf.assign(self._count_ph),
                                   self._mean_tf.assign(self._mean_ph),
                                   self._std_tf.assign(self._std_ph))
        return

    def _update_resource_tf(self):
        feed = {
            self._count_ph: np.array([self._count], dtype=np.int32),
            self._mean_ph: self._mean,
            self._std_ph: self._std
        }
        self._sess.run(self._update_op, feed_dict=feed)
        return

    def _check_synced(self):
        synced = True
        if (mpi_util.is_root_proc()):
            vars = np.concatenate([self._mean, self._std])
            mpi_util.bcast(vars)
        else:
            vars_local = np.concatenate([self._mean, self._std])
            vars_root = np.empty_like(vars_local)
            mpi_util.bcast(vars_root)
            synced = (vars_local == vars_root).all()

        return synced
