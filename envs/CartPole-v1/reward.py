def compute_reward(self, state):
    # Reward for keeping the pole upright (i.e., theta close to 0)
    upright_reward = jnp.cos(state.theta) ** 2

    # Penalize large deviations from the cart's initial position
    x_deviation_penalty = -jnp.abs(state.x)

    # Combine rewards and penalties into a single reward function
    return upright_reward + x_deviation_penalty