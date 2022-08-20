import numpy as np

import sac_dev.util.mpi_util as MPIUtil
import sac_dev.util.tf_util as TFUtil
import tensorflow as tf
from sac_dev.util.logger import Logger


class MPISolver():
    CHECK_SYNC_ITERS = 1000

    def __init__(self, sess, optimizer, vars):
        self._vars = vars
        self._sess = sess
        self._optimizer = optimizer
        self._build_grad_feed(vars)
        self._update = optimizer.apply_gradients(
            zip(self._grad_ph_list, self._vars))
        self._set_flat_vars = TFUtil.SetFromFlat(sess, self._vars)
        self._get_flat_vars = TFUtil.GetFlat(sess, self._vars)

        self._iters = 0

        grad_dim = self._calc_grad_dim()
        self._flat_grad = np.zeros(grad_dim, dtype=np.float32)
        self._global_flat_grad = np.zeros(grad_dim, dtype=np.float32)

        self.reset()

        return

    def get_stepsize(self):
        stepsize = None
        if (isinstance(self._optimizer, tf.compat.v1.train.MomentumOptimizer)):
            stepsize = self._optimizer._learning_rate
        elif (isinstance(self._optimizer, tf.compat.v1.train.AdamOptimizer)):
            stepsize = self._optimizer._lr
        else:
            assert False, "Unsupported optimizer"
        return stepsize

    def update(self, grads, grad_scale=1.0):
        self._flat_grad = np.concatenate([np.reshape(g, [-1]) for g in grads],
                                         axis=0)
        return self.update_flatgrad(self._flat_grad, grad_scale)

    def update_flatgrad(self, flat_grad, grad_scale=1.0):
        if self._iters % self.CHECK_SYNC_ITERS == 0:
            assert self._check_synced(), Logger.print(
                "Network parameters desynchronized")

        if grad_scale != 1.0:
            flat_grad *= grad_scale

        MPIUtil.reduce_sum_inplace(flat_grad,
                                   destination=self._global_flat_grad)
        self._global_flat_grad /= MPIUtil.get_num_procs()

        self._load_flat_grad(self._global_flat_grad)
        self._sess.run([self._update], self._grad_feed)
        self._iters += 1

        return

    def reset(self):
        self._iters = 0
        return

    def get_iters(self):
        return self._iters

    def sync(self):
        vars = self._get_flat_vars()
        MPIUtil.bcast(vars)
        self._set_flat_vars(vars)

        assert (self._check_synced()
                ), Logger.print("Network parameters desynchronized")
        return

    def _is_root(self):
        return MPIUtil.is_root_proc()

    def _build_grad_feed(self, vars):
        self._grad_ph_list = []
        self._grad_buffers = []
        for v in self._vars:
            shape = v.get_shape()
            grad = np.zeros(shape)
            grad_ph = tf.compat.v1.placeholder(tf.float32, shape=shape)
            self._grad_buffers.append(grad)
            self._grad_ph_list.append(grad_ph)

        self._grad_feed = dict({
            g_tf: g
            for g_tf, g in zip(self._grad_ph_list, self._grad_buffers)
        })

        return

    def _calc_grad_dim(self):
        grad_dim = 0
        for grad in self._grad_buffers:
            grad_dim += grad.size
        return grad_dim

    def _load_flat_grad(self, flat_grad):
        start = 0
        for g in self._grad_buffers:
            size = g.size
            np.copyto(g, np.reshape(flat_grad[start:start + size], g.shape))
            start += size
        return

    def _check_synced(self):
        synced = True
        if self._is_root():
            vars = self._get_flat_vars()
            MPIUtil.bcast(vars)
        else:
            vars_local = self._get_flat_vars()
            vars_root = np.empty_like(vars_local)
            MPIUtil.bcast(vars_root)
            synced = (vars_local == vars_root).all()
        return synced
