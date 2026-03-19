"""
空间聚类可视化脚本
加载 SD / GD513 / GLA / GBA 数据集的坐标，使用 GeoDCD 同款 KMeans 空间聚类，
将聚类结果按经纬度画出并保存为 PDF。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans

# ── 中文字体 ──────────────────────────────────────────────────
# 直接通过字体文件路径加载中文字体
_FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_CN_FONT = fm.FontProperties(fname=_FONT_PATH)
plt.rcParams['axes.unicode_minus'] = False

# ── 数据集配置（同 scripts/*.sh 里的参数） ─────────────────────
DATASETS = {
    'SD':    {'N': 716,  'hierarchy': [32, 8]},
    'GD513': {'N': 513,  'hierarchy': [32, 8]},
    'GLA':   {'N': 3834, 'hierarchy': [128, 32, 8]},
    'GBA':   {'N': 2352, 'hierarchy': [128, 32, 8]},
}

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'real')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'cluster_plots')


def geodcd_spatial_cluster(coords: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    GeoDCD 同款空间聚类（GeometricPooler.get_structure 中所用方法）。
    1. 对坐标做 z-score 归一化
    2. 使用 KMeans 聚类
    """
    c_mean = coords.mean(axis=0)
    c_std  = coords.std(axis=0) + 1e-5
    coords_norm = (coords - c_mean) / c_std
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords_norm)
    return labels


def plot_clusters(coords: np.ndarray, labels: np.ndarray, dataset_name: str,
                  ax: plt.Axes):
    """在给定 Axes 上绘制聚类散点图，横纵坐标为经纬度，中文标签，无标题。"""
    n_clusters = len(np.unique(labels))

    # 选择符合学术顶会 (如 ICLR) 审美的色彩映射
    if n_clusters <= 10:
        cmap = plt.cm.tab10
    elif n_clusters <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.Spectral

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap=cmap,
        s=30, alpha=0.8, edgecolors='none',
    )
    
    # 增大字号
    ax.set_xlabel('经度', fontsize=18, fontproperties=_CN_FONT)
    ax.set_ylabel('纬度', fontsize=18, fontproperties=_CN_FONT)
    
    # # 添加淡色网格线提升质感
    # ax.grid(True, linestyle='--', alpha=0.3)
    
    # 移除强制的 equal 比例，以充满 4:3 的画板
    # ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(labelsize=14)

    # # 隐藏上方和右侧的边框线，更加美观
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)


def _make_figure(coords: np.ndarray, base_height: float = 6.0):
    """使用 4:3 的比例生成图片"""
    fig_w = base_height * 4 / 3  # 8.0
    fig_h = base_height          # 6.0
    return plt.subplots(figsize=(fig_w, fig_h))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, cfg in DATASETS.items():
        coords_path = os.path.join(DATA_ROOT, name, 'coords_0.npy')
        if not os.path.exists(coords_path):
            print(f'[SKIP] {name}: coords file not found at {coords_path}')
            continue

        coords = np.load(coords_path)
        print(f'[{name}] coords shape = {coords.shape}')

        # 使用 hierarchy 第一层作为聚类数（与 GeoDCD 一致）
        n_clusters = cfg['hierarchy'][0]
        labels = geodcd_spatial_cluster(coords, n_clusters)

        # ── 画图 ──────────────────────────────────────────────
        fig, ax = _make_figure(coords)
        plot_clusters(coords, labels, name, ax)
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f'{name}_clusters.pdf')
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f'[{name}] Saved to {save_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
