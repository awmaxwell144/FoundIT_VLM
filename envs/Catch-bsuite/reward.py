import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    # Check if the ball and paddle are aligned on the x-axis
    aligned_x = jnp.equal(state.ball_x, state.paddle_x)
    # Check if the ball is one position above the paddle
    just_above = jnp.equal(state.ball_y, state.paddle_y - 1)

    # If the ball is close to the paddle, give a small reward proportion to the y distance
    abs_diff_y = jnp.abs(state.ball_y - state.paddle_y)
    reward_close_to_paddle = jnp.where(abs_diff_y<=2, (3-abs_diff_y)/10, 0)
  
    # If they are aligned and the ball is right above the paddle, give a high positive reward
    reward_perfect_catch = jnp.where(jnp.logical_and(aligned_x, just_above), 1.0, 0)
  
    # If they are aligned but the ball is not right above the paddle, give a positive reward
    reward_aligned_x = jnp.where(aligned_x, 0.2, 0)
  
    # If the ball and paddle are not aligned, give a small negative reward
    reward_not_aligned = jnp.where(jnp.logical_not(aligned_x), -0.1, 0)
  
    # Final reward is a sum of the above rewards
    reward = reward_perfect_catch + reward_aligned_x + reward_not_aligned + reward_close_to_paddle
    return reward