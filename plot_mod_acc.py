import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import findfont, FontProperties

# 设置字体（尝试使用 Times New Roman，否则用默认 serif）
try:
    font = FontProperties(family='Times New Roman')
    findfont(font)
    plt.rcParams["font.family"] = ["Times New Roman", "serif"]
except:
    plt.rcParams["font.family"] = ["serif"]

plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 噪声水平（x轴），从0.2到0.8
noise_levels = [0.2, 0.4, 0.6, 0.8]

# 数据：CIFAR100 实验结果（来自表格）
# data = {
#     'LRA*': {'IDN': [60.09, 45.66, 30.25, 15.72], 'Sym': [60.32, 45.61, 30.95, 16.24], 'Asym': [67.69, 60.35, 53.26, 45.91]},
#     'DLD*': {'IDN': [72.50, 68.24, 60.52, 43.40], 'Sym': [73.51, 72.22, 70.01, 62.26], 'Asym': [73.71, 68.08, 42.19, 37.12]},
#     'OT-Filter*': {'IDN': [69.77, 63.59, 44.29, 14.50], 'Sym': [61.04, 48.31, 33.53, 14.65], 'Asym': [77.56, 76.26, 41.55, 41.30]},
#     'DMLP': {'IDN': [81.73, 69.25, 59.77, 55.52], 'Sym': [82.30, 68.52, 59.11, 52.34], 'Asym': [89.84, 84.93, 81.66, 80.89]},
#     'TNDC': {'IDN': [89.13, 87.12, 85.11, 82.05], 'Sym': [87.58, 85.95, 83.87, 81.43], 'Asym': [89.54, 88.55, 87.58, 86.42]}
# }
# cifar10
data = {
    'LRA*': {'IDN': [76.37, 57.90, 39.16, 20.65], 'Sym': [77.89, 60.99, 43.94, 26.98], 'Asym': [83.54, 72.20, 60.88, 49.26]},
    'DLD*': {'IDN': [94.33, 91.38, 43.24, 5.00], 'Sym': [94.51, 94.34, 93.60, 66.05], 'Asym': [94.34, 89.87, 54.13, 49.88]},
    'OT-Filter*': {'IDN': [86.26, 82.82, 67.86, 41.47], 'Sym': [86.83, 81.31, 72.89, 54.16], 'Asym': [88.43, 85.48, 41.39, 36.95]},
    'DMLP': {'IDN': [80.40, 66.28, 61.66, 64.55], 'Sym': [82.27, 66.75, 55.30, 49.52], 'Asym': [88.50, 80.85, 78.60, 81.74]},
    'TNDC': {'IDN': [95.15, 94.58, 90.99, 88.55], 'Sym': [94.07, 91.62, 88.27, 87.51], 'Asym': [94.61, 94.76, 93.94, 91.67]}
}

# 颜色映射（新增方法）
colors = {
    'LRA*': '#9467bd',      # 紫色
    'DLD*': '#d62728',      # 红色
    'OT-Filter*': '#ff7f0e', # 橙色
    'DMLP': '#2ca02c',     # 绿色
    'TNDC': '#1f77b4'      # 蓝色
}

# colors = {
#     'LRA':      '#4E79A7',   # 深蓝
#     'DLD':      '#F28E2B',   # 橙黄
#     'OT-Filter': '#E15759',  # 红
#     'DMLP':     '#76B7B2',   # 青绿
#     'TNDC':     '#59A14F'    # 亮绿（突出性能）
# }



# 标记样式
markers = {
    'LRA*': '*--',
    'DLD*': 's--',
    'OT-Filter*': '^--',
    'DMLP': 'o--',
    'TNDC': 'd--'
}

# 子图背景色
# bg_colors = {'IDN': '#FFE5CB', 'Sym': '#E0F1DF', 'Asym': '#EFE3F2'}

bg_colors = {
    'Sym':   '#F5FAF5',   # 几乎白，带绿调
    'Asym':  '#FAF8FD',   # 几乎白，带紫调
    'IDN':   '#FFF9F5'    # 几乎白，带橙调
}

# 创建图形和子图（共享y轴）
fig, axes = plt.subplots(1, 3, figsize=(6, 4), sharey=True)
subplot_titles = ['Sym', 'Asym', 'IDN']

for i, ax in enumerate(axes):
    title = subplot_titles[i]
    
    # 设置背景色
    ax.set_facecolor(bg_colors[title])
    
    # 设置标题和坐标轴
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('Noise Levels', fontsize=10)
    ax.set_xticks(noise_levels)
    ax.set_ylim(0, 100)
    
    # 隐藏 Sym 和 Asym 的 y 轴刻度和标签
    if i > 0:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0)
    
    # 绘制每条曲线
    for series in data:
        ax.plot(noise_levels, data[series][title], markers[series], 
                color=colors[series], label=series if i == 0 else "")
    
    # 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 对于 Sym 和 Asym，隐藏左脊柱（y轴线）
    if i > 0:
        ax.spines['left'].set_visible(False)

# 左侧纵坐标标签（仅第一个子图）
axes[0].set_ylabel('Label Accuracy (%)', fontsize=10)

# 设置 y 轴刻度（0 到 100，步长10）
axes[0].set_yticks(np.arange(0, 100, 10))
axes[0].set_yticklabels([str(int(x)) for x in np.arange(0, 100, 10)])

# 全局图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))

# 调整布局以容纳图例
plt.tight_layout(rect=[0, 0.05, 1, 1])

# 保存图像
plt.savefig('/mnt/zfj/exp_results/paper_figs/cifar10_results.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/zfj/exp_results/paper_figs/cifar10_results.pdf', format='pdf', dpi=300, bbox_inches='tight')
