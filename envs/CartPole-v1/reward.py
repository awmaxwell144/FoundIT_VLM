import jax.numpy as jnp

def compute_reward(state) -> float:

    # Get the angle and angular velocity of the pole
    theta = state.theta
    theta_dot = state.theta_dot

    # Define the threshold for an upright pole
    theta_threshold_radians = 0.2

    # Calculate the reward based on the angle and angular velocity
    reward = -jnp.abs(theta)
    if jnp.abs(theta) < theta_threshold_radians:
        reward += 1.0  # Positive reward when the pole is upright

    return reward