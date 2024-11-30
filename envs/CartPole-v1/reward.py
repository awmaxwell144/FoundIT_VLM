def compute_reward(state):
    costheta = jnp.cos(state.theta)
    sintheta = jnp.sin(state.theta)

    # Reward for keeping the pole upright (theta close to 0)
    theta_reward = jnp.exp(-state.theta**2 / 100) * jnp.exp(-state.theta_dot**2 / 10)

    # Penalty for moving the cart far away from its initial position
    x_penalty = -abs(state.x) / 50

    return theta_reward + x_penalty
def compute_reward(state):
    costheta = jnp.cos(state.theta)
    sintheta = jnp.sin(state.theta)

    # Reward for keeping the pole upright (theta close to 0)
    theta_reward = jnp.exp(-state.theta**2 / 100) * jnp.exp(-state.theta_dot**2 / 10)

    # Penalty for moving the cart far away from its initial position
    x_penalty = -abs(state.x) / 50

    # Time-based penalty (encourages agent to balance pole quickly)
    time_penalty = -(state.time / 1000)**2

    return theta_reward + x_penalty + time_penalty