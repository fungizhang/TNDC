import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 1. 全局字体设置
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

noise_levels = [0.2, 0.4, 0.6, 0.8]

# 数据：CIFAR10
data = {
    'LRA*': {'IDN': [76.37, 57.90, 39.16, 20.65], 'Sym': [77.89, 60.99, 43.94, 26.98], 'Asym': [83.54, 72.20, 60.88, 49.26]},
    'DLD*': {'IDN': [94.33, 91.38, 43.24, 5.00], 'Sym': [94.51, 94.34, 93.60, 66.05], 'Asym': [94.34, 89.87, 54.13, 49.88]},
    'OT-Filter*': {'IDN': [86.26, 82.82, 67.86, 41.47], 'Sym': [86.83, 81.31, 72.89, 54.16], 'Asym': [88.43, 85.48, 41.39, 36.95]},
    'DMLP': {'IDN': [80.40, 66.28, 61.66, 64.55], 'Sym': [82.27, 66.75, 55.30, 49.52], 'Asym': [88.50, 80.85, 78.60, 81.74]},
    'TNDC': {'IDN': [95.15, 94.58, 90.99, 88.55], 'Sym': [94.07, 91.62, 88.27, 87.51], 'Asym': [94.61, 94.76, 93.94, 91.67]}
}

# 配色：匹配目标图柔和色调
colors = {
    'LRA*': '#8FA8B8',
    'DLD*': '#E45A5A',
    'OT-Filter*': '#F0B860',
    'DMLP': '#9B7BB8',
    'TNDC': '#2BA89E'
}

# 标记
markers = {
    'LRA*': 'o',
    'DLD*': 's',
    'OT-Filter*': '^',
    'DMLP': 'v',
    'TNDC': 'D'
}

# 创建图形：白色画布，3 个子图共享 y 轴
fig, axes = plt.subplots(1, 3, figsize=(7, 4.5), sharey=True, facecolor='white')
subplot_titles = ['Sym', 'Asym', 'IDN']

for i, ax in enumerate(axes):
    title = subplot_titles[i]

    # 白色子图背景
    ax.set_facecolor('white')

    # 浅灰色网格细线
    ax.grid(True, which='major', linestyle='-', linewidth=0.5,
            color='#D0D0D0', alpha=1.0, zorder=0)
    ax.set_axisbelow(True)

    # 标题（粗体）
    ax.set_title(title, fontsize=15, pad=8, fontweight='bold')

    # X 轴标签（粗体）
    ax.set_xlabel('Noise Levels', fontsize=13, fontweight='bold', labelpad=4)
    ax.set_xticks(noise_levels)
    ax.set_xlim(0.13, 0.87)
    ax.set_ylim(0, 100)

    # 绘制曲线
    for series in data:
        if series == 'TNDC':
            lw, ms, mew = 2.6, 10, 0.5
        else:
            lw, ms, mew = 1.5, 7, 0.4

        ax.plot(noise_levels, data[series][title],
                marker=markers[series],
                linestyle='--',
                color=colors[series],
                linewidth=lw,
                markersize=ms,
                markeredgecolor='white',
                markeredgewidth=mew,
                zorder=3,
                label=series if i == 0 else "")

    # 四周边框都保留：细黑线
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    # 刻度
    ax.tick_params(axis='x', which='both', length=3, width=0.8,
                   color='#333333', pad=4, direction='out')
    ax.tick_params(axis='y', which='both', length=3, width=0.8,
                   color='#333333', pad=4, direction='out')

    # 中间和右侧子图：显示 y 轴刻度数字（独立的框）
    if i > 0:
        # 重新启用 y 轴刻度标签可见性
        for label in ax.get_yticklabels():
            label.set_visible(True)

# 左侧 y 轴标签（粗体）
axes[0].set_ylabel('Label Accuracy (%)', fontsize=13, fontweight='bold', labelpad=4)
for ax in axes:
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([str(int(x)) for x in np.arange(0, 101, 20)])

# 中间和右侧的 y 轴刻度标签隐藏（保持只有左边显示数字，但每个子图都是独立的框）
for ax in axes[1:]:
    ax.tick_params(axis='y', labelleft=False)

# 图例：带黑色细边框
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels,
                    loc='lower center',
                    ncol=5,
                    bbox_to_anchor=(0.5, -0.05),
                    frameon=True,
                    fancybox=False,
                    edgecolor='#333333',
                    facecolor='white',
                    handlelength=2.6,
                    columnspacing=2.0,
                    handletextpad=0.6,
                    borderpad=0.6)
legend.get_frame().set_linewidth(0.8)

# 布局：子图之间留出一定间距，三个框独立
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.subplots_adjust(wspace=0.12)

# 保存：PNG 300dpi 白底
plt.savefig('./cifar10_results.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved successfully")