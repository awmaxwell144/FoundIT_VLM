import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    joint_angle1 = state.joint_angle1
    joint_angle2 = state.joint_angle2
    velocity_1 = state.velocity_1
    velocity_2 = state.velocity_2

    target_height = 1.0
    tip_height = -jnp.cos(joint_angle1) - jnp.cos(joint_angle2 + joint_angle1)  

    velocity_penalty = jnp.abs(velocity_1) + jnp.abs(velocity_2)

    # Use jnp.where to conditionally define the reward
    height_reward = jnp.where(tip_height >= target_height, 0.0, -1.0)
    
    reward = height_reward - 0.1 * velocity_penalty   # add velocity penalty

    return reward