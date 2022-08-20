import argparse
import os
import random
import sys
import time

import numpy as np

import gym
import learning.sac_agent as sac_agent
import sac_configs
import tensorflow as tf
import util.mpi_util as mpi_util

arg_parser = None


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Train or test control policies.")

    parser.add_argument("--env", dest="env", default="")

    parser.add_argument("--train",
                        dest="train",
                        action="store_true",
                        default=True)
    parser.add_argument("--test",
                        dest="train",
                        action="store_false",
                        default=True)

    parser.add_argument("--max_samples",
                        dest="max_samples",
                        type=int,
                        default=np.inf)
    parser.add_argument("--test_episodes",
                        dest="test_episodes",
                        type=int,
                        default=32)
    parser.add_argument("--output_dir", dest="output_dir", default="output")
    parser.add_argument("--output_iters",
                        dest="output_iters",
                        type=int,
                        default=20)
    parser.add_argument("--model_file", dest="model_file", default="")

    parser.add_argument("--visualize",
                        dest="visualize",
                        action="store_true",
                        default=False)
    parser.add_argument("--gpu", dest="gpu", default="")

    arg_parser = parser.parse_args()

    return arg_parser


def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return


def build_env(env_id):
    assert (env_id is not ""), "Unspecified environment."
    env = gym.make(env_id)
    return env


def build_agent(env):
    env_id = arg_parser.env
    agent_configs = {}
    if (env_id in sac_configs.SAC_CONFIGS):
        agent_configs = sac_configs.SAC_CONFIGS[env_id]

    graph = tf.Graph()
    sess = tf.compat.v1.Session(graph=graph)
    agent = sac_agent.SACAgent(env=env, sess=sess, **agent_configs)

    return agent


def set_rand_seed(seed):
    seed += 97 * mpi_util.get_proc_rank()
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


def main(args):
    global arg_parser
    arg_parser = parse_args(args)
    enable_gpus(arg_parser.gpu)

    set_rand_seed(int(time.time()))
    env = build_env(arg_parser.env)

    agent = build_agent(env)
    agent.visualize = arg_parser.visualize
    if (arg_parser.model_file is not ""):
        agent.load_model(arg_parser.model_file)

    if (arg_parser.train):
        agent.train(max_samples=arg_parser.max_samples,
                    test_episodes=arg_parser.test_episodes,
                    output_dir=arg_parser.output_dir,
                    output_iters=arg_parser.output_iters)
    else:
        agent.eval(num_episodes=arg_parser.test_episodes)

    return


if __name__ == "__main__":
    main(sys.argv)
