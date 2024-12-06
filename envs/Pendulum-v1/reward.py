import jax.numpy as jnp


def compute_reward(state):
    theta = angle_normalize(state.theta)
    # Calculate distance to top
    dist_top = jnp.abs(jnp.sin(theta))
    
    # The reward is based on the distance from the pendulum's center of mass to the top and bottom points.
    reward = 2 * (1 - (dist_top ** 2))
    
    return reward.squeeze()