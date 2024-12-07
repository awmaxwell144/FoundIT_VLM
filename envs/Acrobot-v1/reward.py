import numpy as np

def reward(s, a, params):
    # Extract relevant state values
    x, y, z = s[:3]

    # Calculate distance from free end to goal height
    dist_to_goal = np.abs(z - params["height"])

    # Penalize deviations in other dimensions (x and y)
    penalization = 0.1 * (np.abs(x) + np.abs(y))

    # Reward applying torque to swing the chain upwards
    reward_for_torque = 0.5 * a

    # Combine components into overall reward value
    total_reward = -dist_to_goal + reward_for_torque - penalization

    return total_reward