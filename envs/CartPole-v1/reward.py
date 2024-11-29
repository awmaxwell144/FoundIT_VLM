import jax.numpy as jnp

def compute_reward(state) -> float:
    # Check if the pole is upright
    done = jnp.logical_or(
        state.theta < -jnp.pi / 4,
        state.theta > jnp.pi / 4,
    )

    # Reward for not being terminal (i.e., pole is upright)
    reward = 1.0 - jnp.any(done)

    return reward