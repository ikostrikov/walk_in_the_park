#! /usr/bin/env python
import os
import pickle
import shutil

import numpy as np
import tqdm

import gym
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'A1Run-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('wandb', True, 'Log wandb.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')
flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    wandb.init(project='a1')
    wandb.config.update(FLAGS)

    if FLAGS.real_robot:
        from real.envs.a1_env import A1Real
        env = A1Real(zero_action=np.asarray([0.05, 0.9, -1.8] * 4))
    else:
        from env_utils import make_mujoco_env
        env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history)

    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env = gym.wrappers.RecordVideo(
        env,
        f'videos/train_{FLAGS.action_filter_high_cut}',
        episode_trigger=lambda x: True)
    env.seed(FLAGS.seed)

    if not FLAGS.real_robot:
        eval_env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history)
        eval_env = wrap_gym(eval_env, rescale_actions=True)
        eval_env = gym.wrappers.RecordVideo(
            eval_env,
            f'videos/eval_{FLAGS.action_filter_high_cut}',
            episode_trigger=lambda x: True)
        eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space,
                              env.action_space, **kwargs)

    chkpt_dir = 'saved/checkpoints'
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = 'saved/buffers'

    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

    if last_checkpoint is None:
        start_i = 0
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     FLAGS.max_steps)
        replay_buffer.seed(FLAGS.seed)
    else:
        start_i = int(last_checkpoint.split('_')[-1])

        agent = checkpoints.restore_checkpoint(last_checkpoint, agent)

        with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
            replay_buffer = pickle.load(f)

    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation))
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                decode = {'r': 'return', 'l': 'length', 't': 'time'}
                wandb.log({f'training/{decode[k]}': v}, step=i)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)

        if i % FLAGS.eval_interval == 0:
            if not FLAGS.real_robot:
                eval_info = evaluate(agent,
                                     eval_env,
                                     num_episodes=FLAGS.eval_episodes)
                for k, v in eval_info.items():
                    wandb.log({f'evaluation/{k}': v}, step=i)

            checkpoints.save_checkpoint(chkpt_dir,
                                        agent,
                                        step=i + 1,
                                        keep=20,
                                        overwrite=True)

            try:
                shutil.rmtree(buffer_dir)
            except:
                pass

            os.makedirs(buffer_dir, exist_ok=True)
            with open(os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb') as f:
                pickle.dump(replay_buffer, f)


if __name__ == '__main__':
    app.run(main)
