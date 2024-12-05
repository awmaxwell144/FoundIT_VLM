
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import List
import jax.numpy as jnp
import os


def animate(state_seq, save_path):
    theta, theta_dot, last_u = [],[]
    for state in state_seq:
        theta.append(state.theta)
        theta_dot.append(state.theta_dot)
        last_u.append(state.last_u)
    theta_dot = jnp.array(theta_dot)
    theta = jnp.array(theta)
    last_u = jnp.append(last_u)

    theta, theta_dot, last_u = [], [], []
    for state in state_seq:
        theta.append(state.theta)
        theta_dot.append(state.theta_dot)
        last_u.append(state.last_u)
    
    theta = jnp.array(theta)
    theta_dot = jnp.array(theta_dot)
    last_u = jnp.array(last_u)

    # Pendulum parameters
    length = 1.0  # Length of the pendulum

    # Initialize figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.5 * length, 1.5 * length)
    ax.set_ylim(-1.5 * length, 1.5 * length)
    ax.set_aspect('equal')
    ax.axis('off')

    # Create pendulum line and bob
    pendulum_line, = ax.plot([], [], lw=2, color="black")
    pendulum_bob, = ax.plot([], [], 'o', markersize=10, color="red")

    # Update function for animation
    def update(frame):
        x = length * np.sin(theta[frame])
        y = -length * np.cos(theta[frame])
        pendulum_line.set_data([0, x], [0, y])
        pendulum_bob.set_data(x, y)
        return pendulum_line, pendulum_bob

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(state_seq), interval=50, blit=True)

    # Save animation as MP4
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Pendulum Animation'))
    anim.save(save_path, writer=writer)




def animate_frames(state_seq, output_dir):
        # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Setup figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    pendulum_line, = ax.plot([], [], lw=2, color="blue", marker='o')

    # Function to draw a single frame
    def draw_frame(frame, frame_idx):
        cos_theta, sin_theta, _ = frame
        x, y = sin_theta, -cos_theta
        pendulum_line.set_data([0, x], [0, y])
        fig.canvas.draw()
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        fig.savefig(frame_path, bbox_inches='tight', pad_inches=0)

    # Generate frames
    for idx, frame in enumerate(state_seq):
        draw_frame(frame, idx)

    plt.close(fig)