import numpy as np
import jax.numpy as jnp
def compute_reward(state):
    # Reward for balancing pole upright (within certain angles)
    reward_balancing = jnp.exp(-state.theta**2) + jnp.exp(-state.theta_dot**2)

    # Penalty for cart moving outside bounds or time running out
    penalty_cart = -jnp.logical_or(
        state.x < -0.5,
        state.x > 0.5
    )
    penalty_time = -jnp.logical_or(
        state.time >= 500,  # max time steps in episode (adjust this value as needed)
        jnp.abs(state.theta) > 0.1,  # adjust the threshold for angle deviation as needed
        jnp.abs(state.x_dot) > 3.0   # adjust the threshold for cart speed as needed
    )

    return reward_balancing + penalty_cart + penalty_time