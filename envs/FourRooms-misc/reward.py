import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    at_goal = jnp.all(state.pos == state.goal)
    reward = jnp.where(at_goal, 0, -1)
    return reward