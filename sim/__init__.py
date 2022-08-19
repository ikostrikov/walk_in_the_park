from dm_control import composer
from dmcgym import DMCGYM
from gym.envs.registration import register
from gym.wrappers import FlattenObservation
from sim.robots import A1
from sim.tasks import Run


def make_env(task_name: str, randomize_ground: bool = True):
    assert task_name in ['run']

    robot = A1()

    task = Run(robot, randomize_ground=randomize_ground)

    env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)

    env = DMCGYM(env)
    env = FlattenObservation(env)

    return env


make_env.metadata = DMCGYM.metadata

register(id=f"A1Run-v0",
         entry_point="sim:make_env",
         max_episode_steps=400,
         kwargs=dict(task_name='run'))
