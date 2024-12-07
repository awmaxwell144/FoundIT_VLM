import jax.numpy as jnp


def compute_reward(state) -> float:
    params = EnvParams()  # Get default environment parameters
    
    # Calculate distance from center of gravity to fixed point
    dist_to_fixed_point = jnp.abs(state.position - params.goal_position)
    
    # Reward for being upright and having zero velocity
    reward = 100 * (
        (state.position >= params.goal_position) * (state.velocity >= params.goal_velocity)
    )
    
    # Penalize distance from center of gravity to fixed point
    reward -= 10 * dist_to_fixed_point
    
    return jnp.maximum(reward, -0.1)  # Ensure reward is in [-0.1, infinity]