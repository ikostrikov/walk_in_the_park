import time

import numpy as np
import tqdm

import gym
# import sac_dev.learning.mpi_solver as mpi_solver
import real.sac_dev.learning.nets.net_builder as net_builder
import real.sac_dev.learning.rl_agent as rl_agent
# import sac_dev.util.mpi_util as mpi_util
import real.sac_dev.util.net_util as net_util
import real.sac_dev.util.rl_path as rl_path
import tensorflow as tf
'''
Soft Actor-Critic Agent
'''


class SACAgent(rl_agent.RLAgent):
    ADV_EPS = 1e-5

    def __init__(self,
                 env,
                 sess,
                 actor_net="fc_2layers_256units",
                 critic_net="fc_2layers_256units",
                 use_MPI_solver=False,
                 parallel_ensemble=False,
                 profile=False,
                 actor_stepsize=0.0003,
                 actor_init_output_scale=0.01,
                 actor_batch_size=256,
                 actor_steps=256,
                 action_std=0.2,
                 action_l2_weight=0.0,
                 action_entropy_weight=0.0,
                 num_critic_nets=2,
                 critic_stepsize=0.0003,
                 critic_batch_size=256,
                 critic_steps=256,
                 num_ensemble_subset=2,
                 discount=0.99,
                 samples_per_iter=512,
                 replay_buffer_size=50000,
                 normalizer_samples=300000,
                 enable_val_norm=False,
                 num_action_samples=1,
                 tar_stepsize=0.01,
                 steps_per_tar_update=1,
                 init_samples=25000,
                 visualize=False):

        self._actor_net = actor_net
        self._critic_net = critic_net
        self._use_MPI_solver = use_MPI_solver
        self._parallel_ensemble = parallel_ensemble
        self._profile = profile

        self._actor_stepsize = actor_stepsize
        self._actor_init_output_scale = actor_init_output_scale
        self._actor_batch_size = actor_batch_size
        self._actor_steps = actor_steps
        self._action_std = action_std
        self._action_l2_weight = action_l2_weight
        self._action_entropy_weight = action_entropy_weight

        self._num_critic_nets = num_critic_nets
        self._critic_stepsize = critic_stepsize
        self._critic_batch_size = critic_batch_size
        self._critic_steps = critic_steps
        self._num_ensemble_subset = num_ensemble_subset

        self._num_action_samples = num_action_samples
        self._tar_stepsize = tar_stepsize
        self._steps_per_tar_update = steps_per_tar_update
        self._init_samples = init_samples

        self._actor_bound_loss_weight = 10.0

        super().__init__(env=env,
                         sess=sess,
                         discount=discount,
                         samples_per_iter=samples_per_iter,
                         replay_buffer_size=replay_buffer_size,
                         normalizer_samples=normalizer_samples,
                         enable_val_norm=enable_val_norm,
                         visualize=visualize)
        return

    def sample_action(self, s, test):
        n = len(s.shape)
        s = np.reshape(s, [-1, self.get_state_size()])

        feed = {self._s_ph: s}

        if (test):
            run_tfs = [self._mode_a_tf, self._mode_a_logp_tf]
        else:
            run_tfs = [self._sample_a_tf, self._sample_a_logp_tf]

        a, logp = self._sess.run(run_tfs, feed_dict=feed)

        if n == 1:
            a = a[0]
            logp = logp[0]

        return a, logp

    def eval_critic(self, s, a):
        n = len(s.shape)
        # s = np.reshape(s, [-1, self.get_state_size()])
        # a = np.reshape(a, [-1, self.get_action_size()])

        feed = {self._s_ph: s, self._a_ph: a}
        parallel_vs = self._sess.run(self._parallel_norm_critic_tfs,
                                     feed_dict=feed)
        seq_vs = self._sess.run(self._seq_norm_critic_tfs, feed_dict=feed)

        print("norm diff forward pass:",
              np.linalg.norm(parallel_vs - seq_vs, axis=-1))

        # if n == 1:
        #     v = v[0]
        # return v

    def get_critic_steps(self):
        if self._use_MPI_solver:
            return self._critic_solver.get_iters()
        else:
            return self._critic_updates

    def get_actor_steps(self):
        if self._use_MPI_solver:
            return self._actor_solver.get_iters()
        else:
            return self._actor_updates

    def _build_nets(self):
        s_size = self.get_state_size()
        a_size = self.get_action_size()
        action_space = self.get_action_space()
        self._build_dataset(s_size, a_size)

        self._s_ph = tf.compat.v1.placeholder(tf.float32,
                                              shape=[None, s_size],
                                              name="s")
        self._a_ph = tf.compat.v1.placeholder(tf.float32,
                                              shape=[None, a_size],
                                              name="a")
        self._tar_val_ph = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None],
                                                    name="tar_val")
        self._r_ph = tf.compat.v1.placeholder(tf.float32,
                                              shape=[None],
                                              name="r")
        self._next_s_ph = tf.compat.v1.placeholder(tf.float32,
                                                   shape=[None, s_size],
                                                   name="next_s")
        self._terminate_ph = tf.compat.v1.placeholder(tf.int32,
                                                      shape=[None],
                                                      name="terminate")

        norm_s_tf = self._s_norm.normalize_tf(self._s_ph)
        norm_a_tf = self._a_norm.normalize_tf(self._a_ph)
        norm_next_s_tf = self._s_norm.normalize_tf(self._next_s_ph)

        with tf.compat.v1.variable_scope(self.MAIN_SCOPE):
            self._norm_a_pd_tf = self._build_net_actor(
                net_name=self._actor_net, input_tfs=[norm_s_tf])
            self._sample_norm_a_tfs = [
                self._norm_a_pd_tf.sample()
                for _ in range(self._num_action_samples)
            ]
            self._sample_norm_a_tfs = tf.stack(self._sample_norm_a_tfs,
                                               axis=-2)

            self._norm_critic_tf, self._norm_critic_tfs = self._build_net_critic(
                net_name=self._critic_net,
                input_tfs=[norm_s_tf, norm_a_tf],
                num_nets=self._num_critic_nets)
            self._critic_tf = self._val_norm.unnormalize_tf(
                self._norm_critic_tf)

            critic_sample_input_tfs = [
                tf.stack([norm_s_tf] * self._num_action_samples, axis=-2),
                self._sample_norm_a_tfs
            ]
            self._sample_norm_critic_tf, _ = self._build_net_critic(
                net_name=self._critic_net,
                input_tfs=critic_sample_input_tfs,
                num_nets=self._num_critic_nets,
                reuse=True)

        with tf.compat.v1.variable_scope(self.TARGET_SCOPE):
            self._next_norm_a_pd_tf = self._build_net_actor(
                net_name=self._actor_net, input_tfs=[norm_next_s_tf])
            self._sample_next_norm_a_tfs = [
                self._next_norm_a_pd_tf.sample()
                for _ in range(self._num_action_samples)
            ]
            self._sample_next_norm_a_tfs = tf.stack(
                self._sample_next_norm_a_tfs, axis=-2)

            critic_sample_next_input_tfs = [
                tf.stack([norm_next_s_tf] * self._num_action_samples, axis=-2),
                self._sample_next_norm_a_tfs
            ]
            self._sample_next_norm_critic_tar_tf, _ = self._build_net_critic(
                net_name=self._critic_net,
                input_tfs=critic_sample_next_input_tfs,
                num_nets=self._num_critic_nets)

        sample_norm_a_tf = self._norm_a_pd_tf.sample()
        self._sample_a_tf = self._a_norm.unnormalize_tf(sample_norm_a_tf)
        self._sample_a_logp_tf = self._norm_a_pd_tf.log_prob(sample_norm_a_tf)

        mode_norm_a_tf = self._norm_a_pd_tf.mode()
        self._mode_a_tf = self._a_norm.unnormalize_tf(mode_norm_a_tf)
        self._mode_a_logp_tf = self._norm_a_pd_tf.log_prob(mode_norm_a_tf)

        main_critic_vars = self._tf_vars(self.MAIN_SCOPE + "/" +
                                         self.CRITIC_SCOPE)
        main_actor_vars = self._tf_vars(self.MAIN_SCOPE + "/" +
                                        self.ACTOR_SCOPE)
        tar_critic_vars = self._tf_vars(self.TARGET_SCOPE + "/" +
                                        self.CRITIC_SCOPE)
        tar_actor_vars = self._tf_vars(self.TARGET_SCOPE + "/" +
                                       self.ACTOR_SCOPE)
        assert len(main_critic_vars) == len(tar_critic_vars)
        assert len(main_actor_vars) == len(tar_actor_vars)

        self._sync_tar_vars_op = list(
            map(
                lambda v: v[0].assign(v[1]),
                zip(tar_critic_vars + tar_actor_vars,
                    main_critic_vars + main_actor_vars)))
        self._update_critic_tar_vars_op = list(
            map(
                lambda v: v[0].assign((1.0 - self._tar_stepsize) * v[0] + self.
                                      _tar_stepsize * v[1]),
                zip(tar_critic_vars, main_critic_vars)))
        self._update_actor_tar_vars_op = list(
            map(
                lambda v: v[0].assign((1.0 - self._tar_stepsize) * v[0] + self.
                                      _tar_stepsize * v[1]),
                zip(tar_actor_vars, main_actor_vars)))

        return

    def _build_dataset(self, s_size, a_size):
        self._all_s_ph = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, s_size],
                                                  name="all_s")
        self._all_a_ph = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, a_size],
                                                  name="all_a")
        self._all_r_ph = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None],
                                                  name="all_r")
        self._all_next_s_ph = tf.compat.v1.placeholder(tf.float32,
                                                       shape=[None, s_size],
                                                       name="all_next_s")
        self._all_terminate_ph = tf.compat.v1.placeholder(tf.int32,
                                                          shape=[None],
                                                          name="all_terminate")

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self._all_s_ph, self._all_a_ph, self._all_r_ph,
             self._all_next_s_ph, self._all_terminate_ph))
        train_dataset = train_dataset.repeat().shuffle(
            buffer_size=self._replay_buffer._buffer_size,
            reshuffle_each_iteration=True)
        num_procs = 1
        local_batch_size = int(np.ceil(self._critic_batch_size / num_procs))
        train_dataset = train_dataset.batch(local_batch_size,
                                            drop_remainder=True)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self._iterator = tf.compat.v1.data.make_initializable_iterator(
            train_dataset)
        self._input_op = self._iterator.get_next()

    def _build_losses(self):
        val_fail = 0.0

        norm_next_val_tf = tf.reduce_mean(
            input_tensor=self._sample_next_norm_critic_tar_tf, axis=-1)
        next_val_tf = self._val_norm.unnormalize_tf(norm_next_val_tf)
        next_val_tf = tf.compat.v1.where(
            tf.math.equal(self._terminate_ph, rl_path.Terminate.Fail.value),
            val_fail * tf.ones_like(next_val_tf), next_val_tf)
        next_val_tf = tf.stop_gradient(next_val_tf)

        tar_val_tf = self._r_ph + self._discount * next_val_tf
        norm_tar_val_tf = self._val_norm.normalize_tf(tar_val_tf)
        norm_tar_val_tf = tf.expand_dims(norm_tar_val_tf, axis=0)

        norm_val_diff = norm_tar_val_tf - self._norm_critic_tfs

        self._critic_loss_tf = 0.5 * tf.reduce_mean(input_tensor=tf.reduce_sum(
            input_tensor=tf.square(norm_val_diff), axis=0))
        self._actor_loss_tf = -tf.reduce_mean(
            input_tensor=self._sample_norm_critic_tf)

        if (self._actor_bound_loss_weight != 0.0):
            self._actor_loss_tf += self._actor_bound_loss_weight * self._action_bound_loss(
                self._norm_a_pd_tf)

        if (self._action_l2_weight != 0):
            self._actor_loss_tf += self._action_l2_weight * self._action_l2_loss(
                self._norm_a_pd_tf)

        self._entropy_tf = self._action_entropy(self._norm_a_pd_tf)
        if (self._action_entropy_weight != 0):
            self._actor_loss_tf += -self._action_entropy_weight * self._entropy_tf

        return

    def _build_solvers(self):
        critic_vars = self._tf_vars(self.MAIN_SCOPE + "/" + self.CRITIC_SCOPE)
        critic_opt = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._critic_stepsize)
        actor_vars = self._tf_vars(self.MAIN_SCOPE + "/" + self.ACTOR_SCOPE)
        actor_opt = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._actor_stepsize)

        if self._use_MPI_solver:
            self._critic_grad_tf = tf.gradients(ys=self._critic_loss_tf,
                                                xs=critic_vars)
            self._critic_solver = mpi_solver.MPISolver(self._sess, critic_opt,
                                                       critic_vars)

            self._actor_grad_tf = tf.gradients(ys=self._actor_loss_tf,
                                               xs=actor_vars)
            self._actor_solver = mpi_solver.MPISolver(self._sess, actor_opt,
                                                      actor_vars)
        else:
            self._critic_train_op = critic_opt.minimize(self._critic_loss_tf,
                                                        var_list=critic_vars)
            self._actor_train_op = actor_opt.minimize(self._actor_loss_tf,
                                                      var_list=actor_vars)

            self._critic_updates = 0
            self._actor_updates = 0
        return

    def _build_net_actor(self, net_name, input_tfs, reuse=False):
        with tf.compat.v1.variable_scope(self.ACTOR_SCOPE, reuse=reuse):
            h = net_builder.build_net(net_name=net_name,
                                      input_tfs=input_tfs,
                                      reuse=reuse)
            norm_a_pd_tf = self._build_action_pd(
                input_tf=h,
                init_output_scale=self._actor_init_output_scale,
                mean_activation=tf.math.tanh,
                reuse=reuse)
        return norm_a_pd_tf

    def _build_net_critic(self, net_name, input_tfs, num_nets, reuse=False):
        should_squeeze = len(
            input_tfs[0].shape) == 2 and self._parallel_ensemble
        out_size = 1
        norm_val_tfs = []

        with tf.compat.v1.variable_scope(self.CRITIC_SCOPE, reuse=reuse):
            # clean this up if it works
            if self._parallel_ensemble:
                with tf.compat.v1.variable_scope("parallel", reuse=reuse):
                    norm_val_tfs = net_util.build_fc_ensemble_net(
                        input_tfs=input_tfs,
                        layers=[512, 256, out_size],
                        ensemble_size=num_nets,
                        reuse=reuse)
            else:
                #                with tf.variable_scope("sequential", reuse=reuse):
                for i in range(num_nets):
                    with tf.compat.v1.variable_scope(str(i), reuse=reuse):
                        h = net_builder.build_net(net_name=net_name,
                                                  input_tfs=input_tfs,
                                                  reuse=reuse)
                        #with tf.variable_scope(str(2), reuse=reuse):
                        curr_norm_val_tf = tf.compat.v1.layers.dense(
                            inputs=h,
                            units=out_size,
                            activation=None,
                            kernel_initializer=tf.compat.v1.keras.initializers.
                            VarianceScaling(scale=1.0,
                                            mode="fan_avg",
                                            distribution="uniform"),
                            reuse=reuse)
                        curr_norm_val_tf = tf.squeeze(curr_norm_val_tf,
                                                      axis=-1)
                        norm_val_tfs.append(curr_norm_val_tf)
            norm_val_tfs = tf.stack(norm_val_tfs)

        if should_squeeze:
            norm_val_tfs = tf.squeeze(norm_val_tfs, axis=-1)

        ensemble_subset = norm_val_tfs
        if self._num_ensemble_subset < num_nets:
            redq_idxs = tf.range(num_nets)
            redq_ridxs = tf.random.shuffle(
                redq_idxs)[:self._num_ensemble_subset]
            ensemble_subset = tf.gather(ensemble_subset, redq_ridxs)

        norm_val_tf = tf.reduce_min(input_tensor=ensemble_subset, axis=0)

        return norm_val_tf, norm_val_tfs

    def _init_vars(self):
        super()._init_vars()

        if self._use_MPI_solver:
            self._sync_solvers()

        self._sync_tar_vars()
        return

    def _sync_solvers(self):
        self._actor_solver.sync()
        self._critic_solver.sync()
        return

    def _sync_tar_vars(self):
        self._sess.run(self._sync_tar_vars_op)
        return

    def _update_critic_tar_vars(self):
        self._sess.run(self._update_critic_tar_vars_op)
        return

    def _update_actor_tar_vars(self):
        self._sess.run(self._update_actor_tar_vars_op)
        return

    def _init_train(self):
        super()._init_train()

        self._collect_init_samples(self._init_samples)
        return

    def _collect_init_samples(self, max_samples):
        print("Collecting {} initial samples".format(max_samples))
        sample_count = 0
        next_benchmark = 1000
        update_normalizer = self._enable_normalizer_update(sample_count)
        start_time = time.time()

        while (sample_count < max_samples):
            _, _, new_sample_count, _ = self._rollout_train(
                1, update_normalizer)
            new_sample_count = mpi_util.reduce_sum(new_sample_count)
            sample_count += new_sample_count

            if (self._need_normalizer_update()):
                self._update_normalizers()
            print("samples: {}/{}".format(sample_count, max_samples))
            if sample_count >= next_benchmark:
                print("Collected {} initial samples in {} sec".format(
                    sample_count,
                    time.time() - start_time))
                next_benchmark += 1000

        return sample_count

    def _update(self, iter, new_sample_count):
        assert self._critic_batch_size == self._actor_batch_size
        num_procs = 1
        local_batch_size = int(np.ceil(self._critic_batch_size / num_procs))

        all_idx = self._replay_buffer.get_valid_idx()
        all_next_idx = self._replay_buffer.get_next_idx(all_idx)

        all_s = self._replay_buffer.get("states", all_idx)
        all_a = self._replay_buffer.get("actions", all_idx)
        all_r = self._replay_buffer.get("rewards", all_idx)
        all_next_s = self._replay_buffer.get("states", all_next_idx)
        all_terminate = self._replay_buffer.get("terminate", all_next_idx)

        feed = {
            self._all_s_ph: all_s,
            self._all_a_ph: all_a,
            self._all_r_ph: all_r,
            self._all_next_s_ph: all_next_s,
            self._all_terminate_ph: all_terminate
        }
        self._sess.run(self._iterator.initializer, feed_dict=feed)

        critic_steps = int(
            np.ceil(self._critic_steps * new_sample_count /
                    self._critic_batch_size))
        critic_info = self._update_critic(critic_steps, self._input_op)

        actor_steps = int(
            np.ceil(self._actor_steps * new_sample_count /
                    self._actor_batch_size))
        actor_info = self._update_actor(actor_steps, self._input_op)

        critic_info = mpi_util.reduce_dict_mean(critic_info)
        actor_info = mpi_util.reduce_dict_mean(actor_info)

        self._log(
            {
                "Critic/Critic_Loss": critic_info["loss"],
                "Critic/Critic_Steps": self.get_critic_steps(),
                "Critic/Critic_Time_Per_Update": critic_info["update (s)"],
                "Actor/Actor_Loss": actor_info["loss"],
                "Actor/Actor_Entropy": actor_info["entropy"],
                "Actor/Actor_Steps": self.get_actor_steps(),
            }, iter)
        info = {"critic_info": critic_info, "actor_info": actor_info}
        return info

    def _update_critic(self, steps, input_op):
        num_procs = 1
        local_batch_size = int(np.ceil(self._critic_batch_size / num_procs))
        info = None
        start = time.time()

        for b in tqdm.trange(steps):
            s, a, r, next_s, terminate = self._sess.run(self._input_op)
            curr_info = self._step_critic(s=s,
                                          a=a,
                                          r=r,
                                          next_s=next_s,
                                          terminate=terminate)

            if (self.get_critic_steps() % self._steps_per_tar_update == 0):
                self._update_critic_tar_vars()

            if (info is None):
                info = curr_info
            else:
                for k, v in curr_info.items():
                    info[k] += v

        critic_update_time = time.time() - start
        info["update (s)"] = critic_update_time

        for k in info.keys():
            info[k] /= steps

        return info

    def _update_actor(self, steps, input_op):
        info = None
        num_procs = mpi_util.get_num_procs()
        local_batch_size = int(np.ceil(self._actor_batch_size / num_procs))

        for b in tqdm.trange(steps):
            s, _, _, _, _ = self._sess.run(self._input_op)
            curr_info = self._step_actor(s=s)

            if (self.get_actor_steps() % self._steps_per_tar_update == 0):
                self._update_actor_tar_vars()

            if (info is None):
                info = curr_info
            else:
                for k, v in curr_info.items():
                    info[k] += v

        for k in info.keys():
            info[k] /= steps

        return info

    def _step_critic(self, s, a, r, next_s, terminate):
        feed = {
            self._s_ph: s,
            self._a_ph: a,
            self._r_ph: r,
            self._next_s_ph: next_s,
            self._terminate_ph: terminate
        }

        if self._use_MPI_solver:
            run_tfs = [self._critic_grad_tf, self._critic_loss_tf]
        else:
            run_tfs = [self._critic_train_op, self._critic_loss_tf]

        results = self._sess.run(run_tfs, feed)

        if self._use_MPI_solver:
            self._critic_solver.update(results[0])
        else:
            self._critic_updates += 1

        info = {"loss": results[1]}

        return info

    def _step_actor(self, s):
        feed = {
            self._s_ph: s,
        }

        if self._use_MPI_solver:
            run_tfs = [
                self._actor_grad_tf, self._actor_loss_tf, self._entropy_tf
            ]
        else:
            run_tfs = [
                self._actor_train_op, self._actor_loss_tf, self._entropy_tf
            ]

        results = self._sess.run(run_tfs, feed)

        if self._use_MPI_solver:
            self._actor_solver.update(results[0])
        else:
            self._actor_updates += 1

        info = {
            "loss": results[1],
            "entropy": results[2],
        }

        return info
