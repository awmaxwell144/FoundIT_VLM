import jax.numpy as jnp

def compute_reward(state) -> float:
    
    # Normalize theta to be between -pi and pi
    theta = ((state.theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    
    # Reward for height (theta)
    height_reward = 0.5 * jnp.sin(theta)  # Normalized reward, peaks at 1 when theta is pi/2
    
    # Penalty for swing speed (theta_dot)
    speed_penalty = -0.01 * abs(state.theta_dot)  # Negative penalty to discourage high speeds
    
    # Combine the rewards and penalties into a single value
    reward = height_reward + speed_penalty
    
    return jnp.clip(reward, -1, 1)  # Normalize the reward to [-1, 1]