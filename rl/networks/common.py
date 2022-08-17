import flax.linen as nn
import jax
from flax.core import FrozenDict

default_init = nn.initializers.xavier_uniform


def soft_target_update(critic_params: FrozenDict,
                       target_critic_params: FrozenDict,
                       tau: float) -> FrozenDict:
    new_target_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau),
                                     critic_params, target_critic_params)

    return new_target_params
