import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    # Calculate the distance between paddle and ball on X-axis
    x_distance = jnp.abs(state.ball_x - state.paddle_x)
    
    # Add a penalty for being away from the ball
    reward = -0.1 * x_distance
  
    # Add a positive reward if ball is directly above the paddle
    if state.ball_y == state.paddle_y + 1 and x_distance == 0:
        reward += 2.0

    return reward