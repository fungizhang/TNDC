import matplotlib.pyplot as plt

# 浮世绘配色方案
# fugui_colors = ['#FFC300', '#A23B72', '#2E86AB'] 
fugui_colors = ['#1C3144', '#A93226', '#2ECC71']

# 数据定义
noise_ratios = ['40%', '47%', '50%', '55%', '60%', '90%']
cifar10_sym = [0, 0, 0, 0, 0, 0]
cifar10_asym = [0, 0, 0, 6, 6, 6]
cifar10_idn = [0, 0, 1, 3, 3, 6]
cifar100_sym = [0, 0, 0, 0, 0, 0]
cifar100_asym = [0, 2, 29, 50, 50, 50]
cifar100_idn = [0, 0, 0, 7, 25, 85]

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

# CIFAR-10 子图
ax1.plot(noise_ratios, cifar10_sym, label='Sym', marker='o', color=fugui_colors[0])
ax1.plot(noise_ratios, cifar10_asym, label='Asym', marker='s', color=fugui_colors[1])
ax1.plot(noise_ratios, cifar10_idn, label='IDN', marker='^', color=fugui_colors[2])
ax1.set_title('CIFAR-10')
ax1.set_xlabel('Noise Ratio')
ax1.set_ylabel('Number of NDCs')
ax1.legend()
ax1.grid(True)

# CIFAR-100 子图
ax2.plot(noise_ratios, cifar100_sym, label='Sym', marker='o', color=fugui_colors[0])
ax2.plot(noise_ratios, cifar100_asym, label='Asym', marker='s', color=fugui_colors[1])
ax2.plot(noise_ratios, cifar100_idn, label='IDN', marker='^', color=fugui_colors[2])
ax2.set_title('CIFAR-100')
ax2.set_xlabel('Noise Ratio')
ax2.set_ylabel('Number of NDCs')
ax2.legend()
ax2.grid(True)

# 显示图表
plt.tight_layout()
# 保存图表为PDF格式
plt.savefig('/mnt/zfj/ndc.pdf')