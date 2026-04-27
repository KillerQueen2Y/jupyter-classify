# -*- coding: utf-8 -*-
"""
CSV网格差异热力图：Difference=相同灰色/不同显示Origin颜色
调用：python compare.py --origin xxx.csv --target xxx.csv --output xxx.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

# ====================== 样式配置 ======================
FIG_SIZE = (15, 8)
GRID_COLOR = "#cccccc"
GRID_LINEWIDTH = 0.5
TITLE_FONTSIZE = 16
AXIS_FONTSIZE = 12
TICK_FONTSIZE = 10
BG_SAME_COLOR = "#f0f0f0"  # 相同值：灰色
CUSTOM_COLORS = ["#86e3ce", "#ffd1aa", "#bab3fa", "#ff9999", "#99ccff"]
# ======================================================

def load_csv(path):
    df = pd.read_csv(path, header=None)
    return df.values.astype(float)

def create_color_map(all_values):
    unique_vals = np.unique(all_values)
    n_vals = len(unique_vals)
    colors = CUSTOM_COLORS[:n_vals] if n_vals <= len(CUSTOM_COLORS) else \
             CUSTOM_COLORS + list(plt.cm.tab10(np.linspace(0,1,n_vals-len(CUSTOM_COLORS))))
    cmap = ListedColormap(colors)
    bounds = np.append(unique_vals, unique_vals.max()+1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, unique_vals

def plot_heatmap(ax, data, title, cmap, norm):
    im = ax.imshow(data, cmap=cmap, norm=norm, origin='upper', aspect='auto')
    ax.set_title(title, fontsize=TITLE_FONTSIZE, weight='bold')
    ax.set_ylabel("Layer", fontsize=AXIS_FONTSIZE)
    rows, cols = data.shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    return im

def main(origin_path, target_path, output_path):
    # 读取数据
    origin = load_csv(origin_path)
    target = load_csv(target_path)
    if origin.shape != target.shape:
        raise ValueError(f"尺寸不匹配：origin{origin.shape} vs target{target.shape}")

    # 计算差异
    diff_mask = origin != target
    print(f"总单元格：{origin.size} | 差异数：{np.sum(diff_mask)}")

    # 统一色板
    all_vals = np.concatenate([origin.flatten(), target.flatten()])
    cmap, norm, _ = create_color_map(all_vals)

    # 差异图数据：不同→origin值，相同→NaN(显示灰色)
    diff_data = np.where(diff_mask, origin, np.nan)
    diff_cmap = cmap.copy()
    diff_cmap.set_bad(color=BG_SAME_COLOR)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    plot_heatmap(axes[0], origin, "Origin", cmap, norm)
    plot_heatmap(axes[1], target, "Target", cmap, norm)
    plot_heatmap(axes[2], diff_data, "Difference", diff_cmap, norm)

    # 保存
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 已保存：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV网格差异热力图")
    parser.add_argument("--origin", "-o", required=True, help="Origin CSV路径")
    parser.add_argument("--target", "-t", required=True, help="Target CSV路径")
    parser.add_argument("--output", "-out", required=True, help="输出图片路径")
    args = parser.parse_args()
    main(args.origin, args.target, args.output)