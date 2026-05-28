import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import findfont, FontProperties

# 设置字体（与前面的设置保持一致，确保学术规范）
try:
    font = FontProperties(family='Times New Roman')
    findfont(font)
    plt.rcParams["font.family"] = ["Times New Roman", "serif"]
except:
    plt.rcParams["font.family"] = ["serif"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 提取 CIFAR-100 表格数据 (Method, Noise_Type, Ratio, Baseline_Acc, TNDC_Acc)
data = [
    # CE
    ("CE", "Symmetric", "20%", 64.99, 73.15), ("CE", "Symmetric", "40%", 50.60, 72.50), ("CE", "Symmetric", "60%", 31.81, 70.92), ("CE", "Symmetric", "80%", 11.63, 69.81),
    ("CE", "Asymmetric", "20%", 72.02, 73.99), ("CE", "Asymmetric", "40%", 62.97, 74.22), ("CE", "Asymmetric", "60%", 52.23, 73.83), ("CE", "Asymmetric", "80%", 44.10, 72.93),
    ("CE", "IDN", "20%", 65.50, 74.06), ("CE", "IDN", "40%", 50.48, 72.49), ("CE", "IDN", "60%", 30.93, 72.00), ("CE", "IDN", "80%", 11.96, 69.51),
    
    # DivideMix -> D-Mix
    ("D-Mix", "Symmetric", "20%", 76.46, 76.65), ("D-Mix", "Symmetric", "40%", 73.53, 76.16), ("D-Mix", "Symmetric", "60%", 66.53, 75.95), ("D-Mix", "Symmetric", "80%", 46.05, 75.68),
    ("D-Mix", "Asymmetric", "20%", 76.47, 76.64), ("D-Mix", "Asymmetric", "40%", 70.69, 76.35), ("D-Mix", "Asymmetric", "60%", 45.93, 76.46), ("D-Mix", "Asymmetric", "80%", 40.80, 75.92),
    ("D-Mix", "IDN", "20%", 75.87, 76.61), ("D-Mix", "IDN", "40%", 67.65, 76.38), ("D-Mix", "IDN", "60%", 41.78, 76.09), ("D-Mix", "IDN", "80%", 13.09, 74.14),
    
    # OT-Filter -> OT-F
    ("OT-F", "Symmetric", "20%", 76.65, 76.90), ("OT-F", "Symmetric", "40%", 73.68, 75.91), ("OT-F", "Symmetric", "60%", 69.34, 72.76), ("OT-F", "Symmetric", "80%", 58.54, 68.31),
    ("OT-F", "Asymmetric", "20%", 75.31, 75.40), ("OT-F", "Asymmetric", "40%", 74.54, 75.72), ("OT-F", "Asymmetric", "60%", 40.67, 76.88), ("OT-F", "Asymmetric", "80%", 40.20, 75.43),
    ("OT-F", "IDN", "20%", 75.27, 76.01), ("OT-F", "IDN", "40%", 73.58, 76.07), ("OT-F", "IDN", "60%", 63.13, 73.71), ("OT-F", "IDN", "80%", 14.23, 71.81),
    
    # LRA
    ("LRA", "Symmetric", "20%", 78.65, 78.54), ("LRA", "Symmetric", "40%", 77.80, 78.94), ("LRA", "Symmetric", "60%", 71.04, 78.35), ("LRA", "Symmetric", "80%", 22.92, 78.25),
    ("LRA", "Asymmetric", "20%", 79.03, 79.38), ("LRA", "Asymmetric", "40%", 69.68, 78.91), ("LRA", "Asymmetric", "60%", 50.23, 79.12), ("LRA", "Asymmetric", "80%", 42.41, 78.88),
    ("LRA", "IDN", "20%", 79.75, 79.82), ("LRA", "IDN", "40%", 73.86, 79.65), ("LRA", "IDN", "60%", 44.47, 79.46), ("LRA", "IDN", "80%", 7.82, 77.22),
    
    # DLD
    ("DLD", "Symmetric", "20%", 81.15, 83.55), ("DLD", "Symmetric", "40%", 80.79, 83.59), ("DLD", "Symmetric", "60%", 80.17, 83.38), ("DLD", "Symmetric", "80%", 78.56, 82.80),
    ("DLD", "Asymmetric", "20%", 81.10, 84.71), ("DLD", "Asymmetric", "40%", 77.88, 84.49), ("DLD", "Asymmetric", "60%", 44.14, 84.31), ("DLD", "Asymmetric", "80%", 42.70, 83.28),
    ("DLD", "IDN", "20%", 81.24, 84.48), ("DLD", "IDN", "40%", 80.94, 84.22), ("DLD", "IDN", "60%", 55.97, 83.72), ("DLD", "IDN", "80%", 4.89, 81.70),
]

# 2. 转换为 Seaborn 容易处理的 DataFrame 格式
records = []
for method, noise_type, ratio, base_acc, tndc_acc in data:
    records.append({"Method": method, "Noise Type": noise_type, "Ratio": ratio, "Variant": "Baseline", "Accuracy": base_acc})
    records.append({"Method": method, "Noise Type": noise_type, "Ratio": ratio, "Variant": "+ TNDC", "Accuracy": tndc_acc})

df = pd.DataFrame(records)

# 3. 设置绘图风格与画布大小
sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharey=True)

noise_types = ["Symmetric", "Asymmetric", "IDN"]
ratios = ["20%", "40%", "60%", "80%"]
palette = {"Baseline": "#b0bec5", "+ TNDC": "#009688"} # 设置图例配色 (灰色代表基线，青色代表加了TNDC)

# 4. 遍历网格填充数据
for i, noise_type in enumerate(noise_types):
    for j, ratio in enumerate(ratios):
        ax = axes[i, j]
        subset = df[(df["Noise Type"] == noise_type) & (df["Ratio"] == ratio)]
        
        # 绘制分组柱状图
        sns.barplot(
            data=subset, 
            x="Method", 
            y="Accuracy", 
            hue="Variant", 
            palette=palette,
            ax=ax,
            edgecolor="black",
            linewidth=1.0
        )
        
        # 在每根柱子上方标注具体数值
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=9, padding=3)

        # 设置标题与坐标轴标签
        if i == 0:
            ax.set_title(f"Ratio: {ratio}", fontsize=18, fontweight="bold")
        
        if j == 0:
            ax.set_ylabel(f"{noise_type}\nAccuracy (%)", fontsize=16, fontweight="bold")
        else:
            ax.set_ylabel("")
            
        ax.set_xlabel("")
        
        # 将横坐标旋转角度改回0度（水平显示）
        ax.tick_params(axis='x', labelsize=12, rotation=0)
        
        # 抬高 Y 轴上限
        ax.set_ylim(0, 105) # CIFAR-100的最高精度在85左右，这里设为105留出数值空间即可
        
        # 移除所有子图自带的图例
        if ax.get_legend():
            ax.legend_.remove()

plt.tight_layout()

# 5. 提取第一个图的图例句柄和标签，用于创建全局图例
handles, labels = axes[0, 0].get_legend_handles_labels()

# 将整体图表往上挪一点，给底部的图例留出空间
plt.subplots_adjust(bottom=0.08, top=0.92)

# 在图表底部中间添加全局图例，横向排列 (ncol=2)
fig.legend(handles, labels, loc='lower center', ncol=2, 
           bbox_to_anchor=(0.5, 0), fontsize=16, frameon=True, edgecolor='black')

# 保存图片 (同时保存PNG与PDF格式)
plt.savefig("cifar100_full_barchart.png", dpi=300, bbox_inches="tight")
# plt.savefig("cifar100_full_barchart.pdf", format="pdf", bbox_inches="tight")