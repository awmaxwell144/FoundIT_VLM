import numpy as np
import jax.numpy as jnp
def compute_reward(state) -> float:
    # Position match between paddle and ball in x-axis: positive reward
    if state.paddle_x == state.ball_x:
        reward = 1.0
    # Let's encourage the agent to quickly position the paddle right beneath the ball: negative reward otherwise
    else:
        reward = -0.1

    # Additional reward if the game is not finished to encourage survival
    if not state.prev_done:
        reward += 0.1

    return reward