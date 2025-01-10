import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
  # Calculate the Euclidean distance between current position and the goal
  distance_to_goal = jnp.linalg.norm(state.pos - state.goal)

  # Give a high negative reward for each time step taken
  time_penalty = -1 * state.time

  # Check if the agent has reached the goal
  is_goal_reached = jnp.all(state.pos == state.goal)
  # if goal is reached, give a large positive reward
  goal_reward = jnp.where(is_goal_reached, 200., 0.)
  
  reward = goal_reward - distance_to_goal + time_penalty
  return reward