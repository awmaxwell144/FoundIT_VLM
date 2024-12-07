import jax.numpy as jnp


def compute_reward(state) -> float:

    # Extract relevant variables from the state
    position = state.position
    
    # Calculate the distance between the center of gravity and the fixed point
    height_reward = jnp.abs(position - 0.5)  # Reward for moving above the middle point

    # Calculate a term that discourages high torque values
    velocity = state.velocity
    torque_reward = -jnp.square(velocity)

    # Combine the two reward terms into a single scalar value
    reward = height_reward + torque_reward
    
    return jnp.sum(reward)  # Normalize to a fixed range of [0, 1]