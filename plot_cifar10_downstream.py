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

# 1. 提取 CIFAR-10 表格数据 (Method, Noise_Type, Ratio, Baseline_Acc, TNDC_Acc)
data = [
    # CE
    ("CE", "Symmetric", "20%", 86.86, 89.97), ("CE", "Symmetric", "40%", 80.95, 87.70), ("CE", "Symmetric", "60%", 74.25, 85.22), ("CE", "Symmetric", "80%", 63.41, 84.55),
    ("CE", "Asymmetric", "20%", 90.57, 89.76), ("CE", "Asymmetric", "40%", 81.23, 89.04), ("CE", "Asymmetric", "60%", 49.11, 84.74), ("CE", "Asymmetric", "80%", 40.84, 81.47),
    ("CE", "IDN", "20%", 84.98, 90.70), ("CE", "IDN", "40%", 69.28, 90.21), ("CE", "IDN", "60%", 36.61, 86.64), ("CE", "IDN", "80%", 10.35, 85.24),
    
    # DivideMix -> D-Mix
    ("D-Mix", "Symmetric", "20%", 92.24, 92.22), ("D-Mix", "Symmetric", "40%", 90.64, 92.14), ("D-Mix", "Symmetric", "60%", 87.25, 92.19), ("D-Mix", "Symmetric", "80%", 77.66, 91.19),
    ("D-Mix", "Asymmetric", "20%", 92.50, 92.21), ("D-Mix", "Asymmetric", "40%", 91.85, 91.91), ("D-Mix", "Asymmetric", "60%", 37.67, 91.33), ("D-Mix", "Asymmetric", "80%", 37.28, 89.32),
    ("D-Mix", "IDN", "20%", 91.89, 92.48), ("D-Mix", "IDN", "40%", 90.78, 92.75), ("D-Mix", "IDN", "60%", 60.84, 92.73), ("D-Mix", "IDN", "80%", 5.53, 92.31),
    
    # OT-Filter -> OT-F
    ("OT-F", "Symmetric", "20%", 95.98, 95.72), ("OT-F", "Symmetric", "40%", 94.30, 94.63), ("OT-F", "Symmetric", "60%", 92.20, 93.14), ("OT-F", "Symmetric", "80%", 84.19, 91.23),
    ("OT-F", "Asymmetric", "20%", 96.00, 95.79), ("OT-F", "Asymmetric", "40%", 95.11, 96.17), ("OT-F", "Asymmetric", "60%", 37.15, 93.36), ("OT-F", "Asymmetric", "80%", 36.31, 92.48),
    ("OT-F", "IDN", "20%", 95.21, 96.35), ("OT-F", "IDN", "40%", 91.53, 95.09), ("OT-F", "IDN", "60%", 76.07, 93.48), ("OT-F", "IDN", "80%", 34.69, 92.25),
    
    # LRA
    ("LRA", "Symmetric", "20%", 96.68, 96.49), ("LRA", "Symmetric", "40%", 96.47, 96.12), ("LRA", "Symmetric", "60%", 93.71, 93.88), ("LRA", "Symmetric", "80%", 67.77, 92.19),
    ("LRA", "Asymmetric", "20%", 96.61, 96.28), ("LRA", "Asymmetric", "40%", 87.27, 95.39), ("LRA", "Asymmetric", "60%", 47.05, 90.74), ("LRA", "Asymmetric", "80%", 38.93, 87.93),
    ("LRA", "IDN", "20%", 96.65, 96.82), ("LRA", "IDN", "40%", 92.75, 96.77), ("LRA", "IDN", "60%", 42.43, 95.83), ("LRA", "IDN", "80%", 8.72, 94.48),
    
    # DLD
    ("DLD", "Symmetric", "20%", 96.95, 96.67), ("DLD", "Symmetric", "40%", 97.00, 95.81), ("DLD", "Symmetric", "60%", 96.92, 94.55), ("DLD", "Symmetric", "80%", 82.43, 93.39),
    ("DLD", "Asymmetric", "20%", 96.91, 96.44), ("DLD", "Asymmetric", "40%", 96.22, 95.99), ("DLD", "Asymmetric", "60%", 39.44, 92.08), ("DLD", "Asymmetric", "80%", 38.94, 88.84),
    ("DLD", "IDN", "20%", 96.95, 97.15), ("DLD", "IDN", "40%", 96.81, 97.31), ("DLD", "IDN", "60%", 44.54, 96.27), ("DLD", "IDN", "80%", 3.31, 95.01),
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
        
        # 为了让柱子上的数字不被切断，抬高 Y 轴上限
        ax.set_ylim(0, 115)
        
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

# 保存图片
plt.savefig("cifar10_full_barchart.png", dpi=300, bbox_inches="tight")
# plt.savefig("cifar10_full_barchart.pdf", format="pdf", bbox_inches="tight")