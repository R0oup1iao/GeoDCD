import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_static_heatmap(matrix, title, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='hot', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Source")
    ax.set_ylabel("Target")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def create_dynamic_gif(dynamic_adj, save_path, fps=10):
    N, _, T = dynamic_adj.shape

    fig, ax = plt.subplots(figsize=(8, 6))

    vmax = np.percentile(dynamic_adj, 99)
    vmin = 0

    im = ax.imshow(dynamic_adj[:, :, 0], cmap='hot', vmin=vmin, vmax=vmax, interpolation='nearest')
    title = ax.set_title(f"Dynamic Causal Strength (t=0)")
    ax.set_xlabel("Source Variable")
    ax.set_ylabel("Target Variable")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    def update(frame):
        im.set_data(dynamic_adj[:, :, frame])
        title.set_text(f"Dynamic Causal Strength (t={frame})")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=range(T), blit=True, interval=1000/fps)

    try:
        ani.save(save_path, writer='pillow', fps=fps)
        print(f"GIF saved to {save_path}")
    except Exception as e:
        print(f"Failed to save GIF (check if pillow/imagemagick is installed): {e}")

    plt.close(fig)
