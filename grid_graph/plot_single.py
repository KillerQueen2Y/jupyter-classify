# -*- coding: utf-8 -*-
"""
单 CSV 热力图：显示一个 CSV 的网格，加图例说明每个值对应的颜色
调用：python plot_single.py --csv xxx.csv --output xxx.png [--transpose]
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

# ====================== 样式配置 ======================
FIG_SIZE = (18, 10)
GRID_COLOR = "#cccccc"
GRID_LINEWIDTH = 0.5
TITLE_FONTSIZE = 16
AXIS_FONTSIZE = 16
TICK_FONTSIZE = 14
CUSTOM_COLORS = ["#86e3ce", "#ffd1aa", "#bab3fa", "#ff9999", "#99ccff"]
# 值到标签的映射
VALUE_LABELS = {
    -1: "Wave Head",
    1: "Anchor Head",
    2: "Veil Head"
}
# ======================================================

def load_csv(path):
    df = pd.read_csv(path, header=None)
    return df.values.astype(float)

def create_color_map(all_values):
    unique_vals = np.sort(np.unique(all_values))
    n_vals = len(unique_vals)
    colors = CUSTOM_COLORS[:n_vals] if n_vals <= len(CUSTOM_COLORS) else \
             CUSTOM_COLORS + list(plt.cm.tab10(np.linspace(0,1,n_vals-len(CUSTOM_COLORS))))
    cmap = ListedColormap(colors)
    bounds = np.append(unique_vals, unique_vals.max()+1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, unique_vals, colors

def plot_heatmap_single(ax, data, title, cmap, norm, unique_vals, colors, is_transposed=False):
    im = ax.imshow(data, cmap=cmap, norm=norm, origin='upper', aspect='auto')
    if is_transposed:
        ax.set_ylabel("Head", fontsize=AXIS_FONTSIZE)
        ax.set_xlabel("Layer", fontsize=AXIS_FONTSIZE)
    else:
        ax.set_ylabel("Layer", fontsize=AXIS_FONTSIZE)
        ax.set_xlabel("Head", fontsize=AXIS_FONTSIZE)
    rows, cols = data.shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # 添加图例：每个值对应一个颜色方块
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', linewidth=1, 
                             label=VALUE_LABELS.get(int(unique_vals[i]), f'Value: {int(unique_vals[i])}'))
                       for i in range(len(unique_vals))]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=TICK_FONTSIZE, title='Class', title_fontsize=AXIS_FONTSIZE)
    
    return im

def main(csv_path, output_path, transpose=False):
    # 读取数据
    data = load_csv(csv_path)
    
    if transpose:
        data = data.T
    
    print(f"数据形状：{data.shape} (rows={data.shape[0]}, cols={data.shape[1]})")
    
    # 创建色板
    cmap, norm, unique_vals, colors = create_color_map(data.flatten())
    
    # 绘图
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_heatmap_single(ax, data, '', cmap, norm, unique_vals, colors, is_transposed=transpose)
    fig.suptitle('Classification Result of 360 Heads', fontsize=TITLE_FONTSIZE + 2, weight='bold', y=0.98)
    
    # 保存
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 已保存：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单 CSV 热力图")
    parser.add_argument("--csv", "-c", required=True, help="CSV 路径")
    parser.add_argument("--output", "-out", required=True, help="输出图片路径")
    parser.add_argument("--transpose", action='store_true', help="是否转置图表（Head 变成行）")
    args = parser.parse_args()
    main(args.csv, args.output, args.transpose)
