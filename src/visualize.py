import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 增大全局字号
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

def plot_static_heatmap(matrix, title, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='hot', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("源变量")
    ax.set_ylabel("目标变量")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()

def create_dynamic_gif(dynamic_adj, save_path, fps=10):
    N, _, T = dynamic_adj.shape

    fig, ax = plt.subplots(figsize=(10, 8))

    vmax = np.percentile(dynamic_adj, 99)
    vmin = 0

    im = ax.imshow(dynamic_adj[:, :, 0], cmap='hot', vmin=vmin, vmax=vmax, interpolation='nearest')
    title = ax.set_title(f"动态因果强度 (t=0)")
    ax.set_xlabel("源变量")
    ax.set_ylabel("目标变量")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    def update(frame):
        im.set_data(dynamic_adj[:, :, frame])
        title.set_text(f"动态因果强度 (t={frame})")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=range(T), blit=True, interval=1000/fps)

    try:
        ani.save(save_path, writer='pillow', fps=fps)
        print(f"动图已保存至 {save_path}")
    except Exception as e:
        print(f"保存动图失败 (请检查是否安装了 pillow/imagemagick): {e}")

    plt.close(fig)

def save_dynamic_frames_pdf(dynamic_adj, save_path):
    """
    均匀抽取8帧并保存为8个独立的PDF文件。
    save_path: 基础保存路径或目录
    """
    N, _, T = dynamic_adj.shape
    # 均匀选取8个时间步
    indices = np.linspace(0, T - 1, 8, dtype=int)
    
    vmax = np.percentile(dynamic_adj, 99)
    vmin = 0
    
    # 确保保存目录存在
    base_dir = os.path.dirname(save_path)
    base_name = os.path.basename(save_path).replace('.pdf', '')
    
    for i, idx in enumerate(indices):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(dynamic_adj[:, :, idx], cmap='hot', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(f"时间步 t={idx}")
        ax.set_xlabel("源变量")
        ax.set_ylabel("目标变量")
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        plt.tight_layout()
        
        # 构造独立文件名，例如: causal_evolution_frame_0.pdf
        individual_path = os.path.join(base_dir, f"{base_name}_frame_{idx}.pdf")
        plt.savefig(individual_path, format='pdf', dpi=300)
        plt.close(fig)
        print(f"帧 {idx} 已保存至: {individual_path}")
    
    print(f"成功保存 8 个独立的 PDF 文件至目录: {base_dir}")
