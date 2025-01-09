import numpy as np
import jax.numpy as jnp
from typing import Any

def compute_reward(state: Any) -> float:
    w1, w2, w3, w4 = 2.5, 0.5, 0.1, 0.1  # Increased weight for positional drift
    reward = (
        1.0 +    # Reward for survival
        - w1 * jnp.abs(state.x) - 
        - w2 * jnp.abs(state.theta) - 
        - w3 * jnp.abs(state.x_dot) - 
        - w4 * jnp.abs(state.theta_dot)
    )
    return reward