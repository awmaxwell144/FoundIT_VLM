import jax.numpy as jnp
def compute_reward(state):
    # Reward for keeping the pole upright within some threshold
    theta_threshold = 0.05  # adjust this value based on your environment
    reward = jnp.maximum(1 - jnp.abs(state.theta), 0)
    return reward