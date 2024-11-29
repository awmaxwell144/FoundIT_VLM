import jax.numpy as jnp

def compute_reward(state) -> float:
    theta = jnp.abs(state.theta)
    theta_dot = jnp.abs(state.theta_dot)
    
    # Reward for keeping the pole upright
    reward_theta = 1 - theta
    
    # Reward for not moving the cart too much
    reward_x_dot = 1 - abs(state.x_dot)
    
    # Combine both rewards, but with different weights
    total_reward = (reward_theta + reward_x_dot) / 2
    
    return total_reward