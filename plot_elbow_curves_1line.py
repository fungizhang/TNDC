import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ==========================================
# 1. 全局字体与样式设置
# ==========================================
available_fonts = {f.name for f in fm.fontManager.ttflist}
if 'Times New Roman' in available_fonts:
    plt.rcParams["font.family"] = ["Times New Roman", "serif"]
else:
    plt.rcParams["font.family"] = ["DejaVu Serif", "serif"]

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12


def plot_combined_elbows(data_list, save_dir):
    """在一张图上绘制1排4个子图，底部共享图例"""
    if not data_list:
        return
        
    n_plots = len(data_list)
    # 创建 1 行 4 列的画布，宽度拉长以适应并排显示
    fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3), facecolor='white')
    
    if n_plots == 1:
        axes = [axes]
        
    for i, (ax, data) in enumerate(zip(axes, data_list)):
        class_id = data['class_id']
        distances = data['distances']
        n_keep = data['n_keep']
        
        n = len(distances)
        if n < 3:
            continue
            
        y = np.asarray(distances, dtype=np.float64)
        x = np.arange(n, dtype=np.float64)
        
        # 弦的坐标
        x_chord = [0, n - 1]
        y_chord = [y[0], y[-1]]
        
        ax.set_facecolor('white')
        
        # 浅灰色网格细线
        ax.grid(True, which='major', linestyle='-', linewidth=0.5,
                color='#D0D0D0', alpha=1.0, zorder=0)
        ax.set_axisbelow(True)
        
        # 1. 绘制平滑的实际距离曲线 (青绿色)
        ax.plot(x, y, color='#2BA89E', linestyle='-', linewidth=2, zorder=3, label='Intra-class distances')
        
        # 2. 绘制弦 (珊瑚红色)
        ax.plot(x_chord, y_chord, color='#E45A5A', linestyle='--', linewidth=2, alpha=0.7, zorder=3, label='Chord')
        
        # 3. 标记算法找到的拐点 (暖黄色)
        # 共享图例中去掉具体的 idx，保持整洁
        elbow_idx = n_keep - 1
        elbow_idx = max(0, min(elbow_idx, n - 1)) 
        
        ax.plot(elbow_idx, y[elbow_idx], marker='o', color='#F0B860', linestyle='None', markersize=8, zorder=4, label='Detected Elbow')
        
        # 4. 画垂线 (暖黄色)
        y_chord_at_elbow = y[0] + (y[-1] - y[0]) * (elbow_idx / max(n - 1, 1))
        ax.vlines(x=elbow_idx, ymin=y_chord_at_elbow, ymax=y[elbow_idx], colors='#F0B860', 
                   linestyles='dotted', zorder=3, label='Max dist to chord')
        
        # 标题和标签
        ax.set_title(f'Class {class_id}', fontsize=14, pad=8, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=13, fontweight='bold', labelpad=4)
        
        # 仅在最左侧的图显示 Y 轴标签，防止拥挤
        if i == 0:
            ax.set_ylabel('Correlation Distance', fontsize=13, fontweight='bold', labelpad=4)
        
        # 四周边框细黑线与刻度样式
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('#333333')
            
        ax.tick_params(axis='both', which='both', length=3, width=0.8,
                       color='#333333', pad=4, direction='out')

    # 提取图例标签并放在图表最底部（共享）
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=4, 
                        bbox_to_anchor=(0.5, -0.08),  # 锚点设置在图表下方
                        frameon=True, edgecolor='#333333', facecolor='white',
                        handlelength=2.0, columnspacing=1.5)
    legend.get_frame().set_linewidth(0.8)

    # 调整布局，为底部的图例留出空间
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.15)
    
    save_path = os.path.join(save_dir, "elbow_curves_combined_0_to_3.png")
    # bbox_extra_artists 确保底部移出绘图区的图例不会被裁剪掉
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', bbox_extra_artists=(legend,))
    
    # 也保存一份 PDF 方便放入论文
    pdf_path = os.path.join(save_dir, "elbow_curves_combined_0_to_3.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', bbox_extra_artists=(legend,))
    
    plt.close()
    print(f"  -> Saved {save_path}")
    print(f"  -> Saved {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot elbow curves from JSON stats.")
    parser.add_argument('--stats_file', type=str, required=True, help="Path to the stats_*.json file.")
    parser.add_argument('--save_dir', type=str, default='./elbow_plots', help="Directory to save plots.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Loading data from {args.stats_file}...")
    with open(args.stats_file, 'r') as f:
        data = json.load(f)
        
    elbow_data = data.get('results', {}).get('elbow_plot_data', {})
    
    if not elbow_data:
        print("Error: No 'elbow_plot_data' found in the JSON file. Make sure you ran with --per_class_alpha elbow.")
        return

    # 按类 ID 排序
    sorted_classes = sorted(elbow_data.keys(), key=lambda x: int(x))
    
    # 强制只取前 4 个类别（即 0-3）
    selected_classes = sorted_classes[:4]
    print(f"Plotting 1x4 layout for classes: {selected_classes} ...")

    # 收集数据
    data_list = []
    for cls_id in selected_classes:
        info = elbow_data[cls_id]
        data_list.append({
            'class_id': cls_id,
            'distances': info['distances'],
            'n_keep': info['n_keep']
        })

    # 调用合并绘图函数
    plot_combined_elbows(data_list, args.save_dir)
        
    print("All plotting finished!")

if __name__ == "__main__":
    main()