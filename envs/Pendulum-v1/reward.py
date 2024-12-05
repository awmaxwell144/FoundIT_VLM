import jax.numpy as jnp


def compute_reward(state) -> float:
    theta = state.theta
    reward = -jnp.abs(jnp.cos(theta))  # negative because it's an inverted problem

    return reward