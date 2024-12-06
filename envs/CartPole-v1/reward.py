import jax.numpy as jnp

def compute_reward(state) -> float:
    # Check if the pole is close to being upright
    theta_close = jnp.abs(state.theta) < 0.1
    
    # Check if the cart or pole are within bounds
    x_within_bounds = jnp.logical_or(
        state.x >= -1,
        state.x <= 1,
    )
    
    # Check if the cart or pole's velocities are low
    vel_low = jnp.logical_and(
        jnp.abs(state.x_dot) < 0.5,
        jnp.abs(state.theta_dot) < 0.05,
    )
    
    # Reward is -1 when the pole or cart goes out of bounds, and 0 otherwise
    reward = -jnp.float32(1)
    if x_within_bounds and vel_low:
        reward = 0
    
    return reward