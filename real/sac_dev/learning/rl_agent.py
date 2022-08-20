import abc
import collections
import copy
import os
import pickle
import time

import numpy as np

import gym
# import moviepy.editor as mpy
import pybullet as p
import real.sac_dev.util.logger as logger
import real.sac_dev.util.mpi_util as mpi_util
import real.sac_dev.util.normalizer as normalizer
import real.sac_dev.util.replay_buffer as replay_buffer
import real.sac_dev.util.rl_path as rl_path
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tf.disable_v2_behavior()
'''
Reinforcement Learning Agent
'''


class RLAgent(abc.ABC):
    MAIN_SCOPE = "main"
    TARGET_SCOPE = "target"
    ACTOR_SCOPE = "actor"
    CRITIC_SCOPE = "critic"
    SOLVER_SCOPE = "solver"
    RESOURCE_SCOPE = "resource"

    def __init__(self,
                 env,
                 sess,
                 discount=0.99,
                 samples_per_iter=2048,
                 replay_buffer_size=50000,
                 normalizer_samples=100000,
                 enable_val_norm=False,
                 visualize=False):

        self._env = env
        self._sess = sess

        self._discount = discount
        self._samples_per_iter = samples_per_iter
        self._normalizer_samples = normalizer_samples
        self._enable_val_norm = enable_val_norm

        num_procs = mpi_util.get_num_procs()
        local_replay_buffer_size = int(np.ceil(replay_buffer_size / num_procs))
        self._replay_buffer = self._build_replay_buffer(
            local_replay_buffer_size)

        self.visualize = visualize

        self._logger = None
        self._tf_writer_train = None
        self._tf_writer_eval = None

        with self._sess.as_default(), self._sess.graph.as_default():
            with tf.compat.v1.variable_scope(self.RESOURCE_SCOPE):
                self._build_normalizers()

            self._build_nets()

            with tf.compat.v1.variable_scope(self.SOLVER_SCOPE):
                self._build_losses()
                self._build_solvers()

            self._init_vars()
            self._build_saver()

        return

    def get_state_size(self):
        state_size = np.prod(self._env.observation_space.shape)
        return state_size

    def get_action_size(self):
        action_size = 0
        action_space = self.get_action_space()

        if (isinstance(action_space, gym.spaces.Box)):
            action_size = np.prod(action_space.shape)
        elif (isinstance(action_space, gym.spaces.Discrete)):
            action_size = 1
        else:
            assert False, "Unsupported action space: " + str(
                self._env.action_space)

        return action_size

    def get_action_space(self):
        return self._env.action_space

    def get_total_samples(self):
        total_samples = self._replay_buffer.get_total_count()
        total_samples = int(mpi_util.reduce_sum(total_samples))
        return total_samples

    def eval(self, num_episodes):
        num_procs = mpi_util.get_num_procs()
        local_num_episodes = int(np.ceil(num_episodes / num_procs))
        test_return, test_path_count, metrics = self._rollout_test(
            num_episodes, print_info=True)
        test_return = mpi_util.reduce_mean(test_return)
        test_path_count = mpi_util.reduce_sum(test_path_count)
        metrics = self._reduce_metrics(metrics)

        for metric_name in sorted(metrics.keys()):
            logger.Logger.print("{}: {:.3f}".format(metric_name,
                                                    metrics[metric_name]))

        logger.Logger.print("Test_Return: {:.3f}".format(test_return))
        logger.Logger.print("Test_Paths: {:.3f}".format(test_path_count))
        return

    def _log(self, info_dict, iter, skip_tensorboard=False):
        for label, value in info_dict.items():
            self._logger.log_tabular(os.path.basename(label), value)
        if self._tf_writer_train is not None and not skip_tensorboard:
            self._write_tensorboard(info_dict, iter, self._tf_writer_train)

    def _write_tensorboard(self, info_dict, iteration, tf_writer):
        value_list = []
        for label, value in info_dict.items():
            # Strip "Train_" prefix so train and eval metrics appear on same
            # tensorboard plot.
            if label.startswith("Train_"):
                label = label[6:]
            # Catch-all category so tensorboard will show a grid rather than
            # a single column of graphs.
            if "/" not in label:
                label = "Metrics/" + label
            value_list.append(
                tf.compat.v1.Summary.Value(tag=label, simple_value=value))
        tf_writer.add_summary(tf.compat.v1.Summary(value=value_list),
                              iteration)

    def train(self,
              max_samples,
              test_episodes,
              output_dir,
              output_iters,
              variant=None):
        log_file = os.path.join(output_dir, "log.txt")
        self._logger = logger.Logger()
        self._logger.configure_output_file(log_file, variant=variant)

        video_dir = os.path.join(output_dir, "videos")
        if (mpi_util.is_root_proc()):
            os.makedirs(video_dir, exist_ok=True)
            model_dir = os.path.join(output_dir, "train")
            os.makedirs(model_dir, exist_ok=True)
        self._tf_writer_train = tf.compat.v1.summary.FileWriter(
            os.path.join(output_dir, "train"), graph=self._sess.graph)
        self._tf_writer_eval = tf.compat.v1.summary.FileWriter(
            os.path.join(output_dir, "eval"), graph=self._sess.graph)

        iter = 0
        total_train_path_count = 0
        total_test_path_count = 0
        start_time = time.time()

        self._init_train()

        num_procs = mpi_util.get_num_procs()
        local_samples_per_iter = int(
            np.ceil(self._samples_per_iter / num_procs))
        local_test_episodes = int(np.ceil(test_episodes / num_procs))

        total_samples = 0
        print("Training")

        while (total_samples < max_samples):
            if (iter % output_iters == 0):
                if (mpi_util.is_root_proc()):
                    model_file = os.path.join(model_dir,
                                              "model-{:06d}.ckpt".format(iter))
                    self.save_model(model_file)
                    #self.save_video(
                    #    os.path.join(video_dir, "iter-" + str(iter) + ".gif"))

                test_return, test_path_count, metrics = self._rollout_test(
                    local_test_episodes, print_info=False)
                test_return = mpi_util.reduce_mean(test_return)
                total_test_path_count += mpi_util.reduce_sum(test_path_count)
                log_dict = self._reduce_metrics(metrics)
                log_dict["Return"] = test_return
                log_dict["Paths"] = total_test_path_count
                self._write_tensorboard(log_dict, iter, self._tf_writer_eval)

            self._log(
                {
                    "Iteration": iter,
                    "Test_Return": test_return,
                    "Test_Paths": total_test_path_count,
                },
                iter,
                skip_tensorboard=True)
            self._logger.print_tabular()

            # For iter 0 we need to use _backfill_iter_0()
            if iter % output_iters == 0 and iter != 0:
                self._logger.dump_tabular()

            iter += 1

            update_normalizer = self._enable_normalizer_update(total_samples)
            train_return, train_path_count, new_sample_count, metrics = self._rollout_train(
                local_samples_per_iter, update_normalizer)

            train_return = mpi_util.reduce_mean(train_return)
            train_path_count = mpi_util.reduce_sum(train_path_count)
            new_sample_count = mpi_util.reduce_sum(new_sample_count)

            total_train_path_count += train_path_count

            total_samples = self.get_total_samples()
            wall_time = time.time() - start_time
            wall_time /= 60 * 60  # store time in hours

            log_dict = {
                "Iteration": iter,
                "Wall_Time": wall_time,
                "Samples": total_samples,
                "Train_Return": train_return,
                "Train_Paths": total_train_path_count
            }
            log_dict = self._reduce_metrics(metrics, log_dict)
            self._log(log_dict, iter)

            if (self._need_normalizer_update() and iter == 1):
                self._update_normalizers()

            self._update(iter, new_sample_count)

            if (self._need_normalizer_update()):
                self._update_normalizers()

            if (iter % output_iters == 0):
                test_return, test_path_count = self._rollout_test(
                    local_test_episodes, print_info=False)
                test_return = mpi_util.reduce_mean(test_return)
                total_test_path_count += mpi_util.reduce_sum(test_path_count)

                self._log(
                    {
                        "Test_Return": test_return,
                        "Test_Paths": total_test_path_count,
                    }, iter)

                if (mpi_util.is_root_proc()):
                    model_file = os.path.join(model_dir,
                                              "model-{:06d}.ckpt".format(iter))
                    self.save_model(model_file)
                    #self.save_video(
                    #    os.path.join(video_dir, "iter-" + str(iter) + ".gif"))
                    buffer_file = os.path.join(model_dir, "buffer.pkl")
                    file = open(buffer_file, "wb")
                    pickle.dump(self._replay_buffer, file)
                    file.close()

                self._logger.print_tabular()
                self._logger.dump_tabular()

            else:
                self._logger.print_tabular()

            iter += 1

        self._tf_writer_train.close()
        self._tf_writer_train = None
        self._tf_writer_eval.close()
        self._tf_writer_eval = None
        return

    def _backfill_iter_0(self, test_return, total_test_path_count):
        """Dump results from iter 0 to log file.

        The logger needs to see all keys before dump_tabular() is first called.
        Our 0th iter, when we eval the initial policy, has no train metrics. To
        log it anyway, we wait until we know all the metric names, then set the
        values to 0.
        """
        row_1 = {k: v.val for k, v in self._logger.log_current_row.items()}
        row_0 = {k: 0 for k in row_1}
        row_0["Test_Return"] = test_return
        row_0["Test_Paths"] = total_test_path_count
        self._log(row_0, iter=0, skip_tensorboard=True)
        self._logger.dump_tabular()
        self._log(row_1, iter=1, skip_tensorboard=True)

    def _reduce_metrics(self, metrics, output_dict=None):
        if output_dict is None:
            output_dict = {}
        for metric_name in sorted(metrics.keys()):
            value = metrics[metric_name]
            if metric_name == "max_torque":
                output_dict["Max_Torque"] = mpi_util.reduce_max(value)
                continue
            output_dict[metric_name] = mpi_util.reduce_mean(value)
        return output_dict

    def save_model(self, out_path):
        try:
            save_path = self._saver.save(self._sess,
                                         out_path,
                                         write_meta_graph=False,
                                         write_state=False)
            logger.Logger.print("Model saved to: " + save_path)
        except:
            logger.Logger.print("Failed to save model to: " + out_path)
        return

    def save_video(self, out_path):
        try:
            _, video_frames, _ = self._rollout_path(test=True,
                                                    return_video=True)
            video_frames.extend(
                [np.zeros_like(video_frames[0]) for _ in range(15)])
            clip = mpy.ImageSequenceClip(video_frames, fps=(1 / (.033)))
            clip.write_gif(out_path)
            logger.Logger.print("Video saved to: " + out_path)
        except:
            logger.Logger.print("Failed to save video to: " + out_path)
        return

    def load_model(self, in_path):
        self._saver.restore(self._sess, in_path)
        # load in pickled buffer
        try:
            self._replay_buffer = pickle.load(
                open(in_path[:-17] + "buffer.pkl", "rb"))
        except:
            logger.Logger.print("NO REPLAY BUFFER FOUND")
        self._load_normalizers()
        self._sync_tar_vars()
        logger.Logger.print("Model loaded from: " + in_path)
        return

    def get_state_bound_min(self):
        return self._env.observation_space.low

    def get_state_bound_max(self):
        return self._env.observation_space.high

    def get_action_bound_min(self):
        action_space = self.get_action_space()
        if (isinstance(action_space, gym.spaces.Box)):
            bound_min = self._env.action_space.low
        else:
            bound_min = -np.inf * np.ones(1)
        return bound_min

    def get_action_bound_max(self):
        action_space = self.get_action_space()
        if (isinstance(action_space, gym.spaces.Box)):
            bound_max = self._env.action_space.high
        else:
            bound_max = np.inf * np.ones(1)
        return bound_max

    def render_env(self):
        self._env.render()
        return

    def _build_normalizers(self):
        self._s_norm = self._build_normalizer_state()
        self._a_norm = self._build_normalizer_action()
        self._val_norm = self._build_normalizer_val()
        return

    def _need_normalizer_update(self):
        return self._s_norm.need_update()

    def _build_normalizer_state(self):
        size = self.get_state_size()

        high = self.get_state_bound_max().copy()
        low = self.get_state_bound_min().copy()
        inf_mask = np.logical_or((high >= np.finfo(np.float32).max),
                                 (low <= np.finfo(np.float32).min))
        high[inf_mask] = 1.0
        low[inf_mask] = -1.0

        mean = 0.5 * (high + low)
        std = 0.5 * (high - low)

        norm = normalizer.Normalizer(sess=self._sess,
                                     scope="s_norm",
                                     size=size,
                                     init_mean=mean,
                                     init_std=std)

        return norm

    def _build_normalizer_action(self):
        size = self.get_action_size()

        high = self.get_action_bound_max().copy()
        low = self.get_action_bound_min().copy()
        inf_mask = np.logical_or((high >= np.finfo(np.float32).max),
                                 (low <= np.finfo(np.float32).min))
        assert (not any(inf_mask)), "actions must be bounded"

        mean = 0.5 * (high + low)
        std = 0.5 * (high - low)

        norm = normalizer.Normalizer(sess=self._sess,
                                     scope="a_norm",
                                     size=size,
                                     init_mean=mean,
                                     init_std=std)

        return norm

    def _build_normalizer_val(self):
        mean = 0.0

        if (self._enable_val_norm):
            std = 1.0 / (1.0 - self._discount)
        else:
            std = 1.0

        norm = normalizer.Normalizer(sess=self._sess,
                                     scope="val_norm",
                                     size=1,
                                     init_mean=mean,
                                     init_std=std)
        return norm

    def _build_replay_buffer(self, buffer_size):
        buffer = replay_buffer.ReplayBuffer(buffer_size=buffer_size)
        return buffer

    @abc.abstractmethod
    def sample_action(self, s, test):
        pass

    @abc.abstractmethod
    def _build_nets(self):
        pass

    @abc.abstractmethod
    def _build_losses(self):
        pass

    @abc.abstractmethod
    def _build_solvers(self):
        pass

    @abc.abstractmethod
    def _update(self, iter, new_sample_count):
        pass

    def _init_vars(self):
        self._sess.run(tf.compat.v1.global_variables_initializer())
        return

    def _build_saver(self):
        vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        vars = [v for v in vars if self.SOLVER_SCOPE + '/' not in v.name]
        assert len(vars) > 0
        self._saver = tf.compat.v1.train.Saver(vars, max_to_keep=0)
        return

    def _init_train(self):
        self._replay_buffer.clear()
        return

    def _aggregate_metrics(self, all_metrics, last_metrics):
        aggregate_metrics = {}
        for metric_name, val_list in all_metrics.items():
            aggregate_fn = last_metrics[metric_name][1]
            aggregate_metrics[metric_name] = aggregate_fn(val_list)
        return aggregate_metrics

    def _rollout_train(self, num_samples, update_normalizer):
        new_sample_count = 0
        total_return = 0
        path_count = 0
        all_metrics = collections.defaultdict(list)

        while (new_sample_count < num_samples):
            path, _, metrics = self._rollout_path(test=False)
            path_id = self._replay_buffer.store(path)
            valid_path = path_id != replay_buffer.INVALID_IDX

            if not valid_path:
                assert False, "Invalid path detected"

            path_return = path.calc_return()

            if (update_normalizer):
                self._record_normalizers(path)

            for metric_name in metrics:
                all_metrics[metric_name].append(metrics[metric_name][0])
            all_metrics["max_torque"].append(path.calc_max_torque())
            new_sample_count += path.pathlength()
            total_return += path_return
            path_count += 1

        avg_return = total_return / path_count
        metrics["max_torque"] = (None, np.max)
        aggregate_metrics = self._aggregate_metrics(all_metrics, metrics)

        return avg_return, path_count, new_sample_count, aggregate_metrics

    def _rollout_test(self, num_episodes, print_info=False):
        total_return = 0
        all_metrics = collections.defaultdict(list)
        for e in range(num_episodes):
            path, _, metrics = self._rollout_path(test=True)
            path_return = path.calc_return()
            total_return += path_return
            for metric_name in metrics:
                all_metrics[metric_name].append(metrics[metric_name][0])

            if (print_info):
                logger.Logger.print("Episode: {:d}".format(e))
                logger.Logger.print("Curr_Return: {:.3f}".format(path_return))
                logger.Logger.print("Avg_Return: {:.3f}\n".format(
                    total_return / (e + 1)))

        avg_return = total_return / num_episodes
        aggregate_metrics = self._aggregate_metrics(all_metrics, metrics)
        return avg_return, num_episodes, aggregate_metrics

    def _rollout_path(self, test, return_video=False):
        path = rl_path.RLPath()

        s = self._env.reset()
        s = np.array(s)
        path.states.append(s)
        video_frames = []
        done = False
        while not done:
            a, logp = self.sample_action(s, test)
            s, r, done, info = self._step_env(a)
            s = np.array(s)

            path.states.append(s)
            path.actions.append(a)
            path.rewards.append(r)
            path.max_torques.append(info['max_torque'])

            path.logps.append(logp)

            if (self.visualize):
                self.render_env()
            if return_video:
                video_frames.append(self._env.render(mode="rgb_array"))

        path.terminate = self._check_env_termination()
        return path, video_frames, info.get("metrics", {})

    def _step_env(self, a):
        if (isinstance(self._env.action_space, gym.spaces.Discrete)):
            a = int(a[0])
        output = self._env.step(a)
        return output

    def _check_env_termination(self):
        if (self._env._env_step_counter >= self._env._max_episode_steps):
            term = rl_path.Terminate.Null
        else:
            term = rl_path.Terminate.Fail
        return term

    def _record_normalizers(self, path):
        states = np.array(path.states)
        self._s_norm.record(states)
        return

    def _update_normalizers(self):
        self._s_norm.update()
        return

    def _load_normalizers(self):
        self._s_norm.load()
        self._a_norm.load()
        self._val_norm.load()
        return

    def _build_action_pd(self,
                         input_tf,
                         init_output_scale,
                         mean_activation=None,
                         reuse=False):
        action_space = self.get_action_space()

        if (isinstance(action_space, gym.spaces.Box)):
            output_size = self.get_action_size()

            mean_kernel_init = tf.compat.v1.random_uniform_initializer(
                minval=-init_output_scale, maxval=init_output_scale)
            mean_bias_init = tf.compat.v1.zeros_initializer()
            logstd_kernel_init = tf.compat.v1.random_uniform_initializer(
                minval=-init_output_scale, maxval=init_output_scale)
            logstd_bias_init = np.log(self._action_std) * np.ones(output_size)
            logstd_bias_init = logstd_bias_init.astype(np.float32)

            with tf.compat.v1.variable_scope("mean", reuse=reuse):
                mean_tf = tf.compat.v1.layers.dense(
                    inputs=input_tf,
                    units=output_size,
                    kernel_initializer=mean_kernel_init,
                    bias_initializer=mean_bias_init,
                    activation=None)
                if (mean_activation is not None):
                    mean_tf = mean_activation(mean_tf)

            with tf.compat.v1.variable_scope("logstd", reuse=reuse):
                logstd_tf = tf.compat.v1.get_variable(
                    dtype=tf.float32,
                    name="bias",
                    initializer=logstd_bias_init,
                    trainable=False)
                logstd_tf = tf.broadcast_to(logstd_tf, tf.shape(input=mean_tf))
                std_tf = tf.exp(logstd_tf)

            a_pd_tf = tfp.distributions.MultivariateNormalDiag(
                loc=mean_tf, scale_diag=std_tf)

        elif (isinstance(action_space, gym.spaces.Discrete)):
            output_size = self._env.action_space.n

            kernel_init = tf.compat.v1.random_uniform_initializer(
                minval=-init_output_scale, maxval=init_output_scale)
            bias_init = tf.compat.v1.zeros_initializer()

            with tf.compat.v1.variable_scope("logits", reuse=reuse):
                logits_tf = tf.compat.v1.layers.dense(
                    inputs=input_tf,
                    units=output_size,
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    activation=None)
            a_pd_tf = tfp.distributions.Categorical(logits=logits_tf)

        else:
            assert False, "Unsupported action space: " + str(
                self._env.action_space)

        return a_pd_tf

    def _tf_vars(self, scope=""):
        vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        assert len(vars) > 0
        return vars

    def _enable_normalizer_update(self, total_samples):
        enable_update = total_samples < self._normalizer_samples
        return enable_update

    def _action_l2_loss(self, a_pd_tf):
        action_space = self.get_action_space()
        if (isinstance(action_space, gym.spaces.Box)):
            val = a_pd_tf.mean()
        elif (isinstance(action_space, gym.spaces.Discrete)):
            val = a_pd_tf.logits
        else:
            assert False, "Unsupported action space: " + str(
                self._env.action_space)

        loss = tf.reduce_sum(input_tensor=tf.square(val), axis=-1)
        loss = 0.5 * tf.reduce_mean(input_tensor=loss)
        return loss

    def _action_bound_loss(self, a_pd_tf):
        action_space = self.get_action_space()
        if (isinstance(action_space, gym.spaces.Box)):
            axis = -1
            a_bound_min = self.get_action_bound_min()
            a_bound_max = self.get_action_bound_max()
            assert (np.all(np.isfinite(a_bound_min)) and np.all(
                np.isfinite(a_bound_max))), "Actions must be bounded."

            norm_a_bound_min = self._a_norm.normalize(a_bound_min)
            norm_a_bound_max = self._a_norm.normalize(a_bound_max)

            val = a_pd_tf.mean()
            violation_min = tf.minimum(val - norm_a_bound_min, 0)
            violation_max = tf.maximum(val - norm_a_bound_max, 0)
            violation = tf.reduce_sum(input_tensor=tf.square(violation_min), axis=axis) \
                        + tf.reduce_sum(input_tensor=tf.square(violation_max), axis=axis)

            a_bound_loss = 0.5 * tf.reduce_mean(input_tensor=violation)
        else:
            a_bound_loss = tf.zeros(shape=[])

        return a_bound_loss

    def _action_entropy(self, a_pd_tf):
        loss = a_pd_tf.entropy()
        loss = tf.reduce_mean(input_tensor=loss)
        return loss
