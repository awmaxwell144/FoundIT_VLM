import numpy as np
import jax.numpy as jnp

def compute_reward(state) -> float:
    # extract state variables
    position = state.position
    velocity = state.velocity
    
    # set target position
    target_position = 0.

    # calculate distance from the target position
    distance = jnp.abs(position - target_position)
    
    # encourage the agent to increase speed to the right
    if velocity > 0 and position > target_position:
        reward_velocity = 1.0
    else:
        reward_velocity = -1.0
    
    # we want the car to reach target position, so give lower reward if the car is far from the target
    reward_position = -distance

    # total reward combines position and velocity rewards
    reward = reward_position + reward_velocity 

    return reward