import jax.numpy as jnp

def compute_reward(state) -> float:
    
    # Define a minimum distance from the goal position to discourage reaching it too quickly
    min_distance = 0.1
    
    # Calculate the distance between the car's position and the goal position
    distance = jnp.abs(state.position - 0.5)
    
    # Reward the agent for being close to the goal state
    reward = jnp.where(distance < min_distance, 10 * (1 - distance / min_distance), 0)
    
    return reward