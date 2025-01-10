import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    theta = state.theta
    theta_dot = state.theta_dot
    terminal_cost = jnp.square(theta % jnp.pi)
    control_cost = jnp.square(theta_dot)

    # The reward should be high for remaining in the upright position and low for deviations
    reward = -(terminal_cost + 0.1 * control_cost)
    return reward