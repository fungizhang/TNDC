import sys
# sys.path.append('/mnt/zfj/projects/phd/projects_phd/CWU')
import torch
from torchvision import datasets, transforms
# from datasets import load_dataset
from tqdm import tqdm
import os

# 自定义数据集类（假设你已经正确实现了它）
# from load_parquet import *

# 参数
dataset_name = 'cifar100'  # stanford_cars
# save_path = '/home/zhangfeng/zhangfangjiao/projects/CWU/saved/saved_features/web-car-del_train_l_features.pth'
# save_path = '/home/zhangfeng/zhangfangjiao/projects/CWU/saved/saved_features/clothing1m_train_l_features.pth'
# save_path = '/home/zhangfeng/zhangfangjiao/projects/CWU/saved/saved_features/web-car-del15_train_l_features.pth'
save_path = f'/mnt/zfj/projects/phd/projects_phd/CWU/saved/saved_features/{dataset_name}_features_no_aug.pth'
# save_path = '/mnt/zfj/projects/phd/projects_phd/CWU/saved/saved_features/webvision_features_no_aug.pth'


root_path = '/mnt/zfj/projects/phd/projects_phd/CWU/saved/saved_features'
os.makedirs(root_path, exist_ok=True)


if dataset_name == 'cifar10':
    dataset_path = '/mnt/zfj/dataset/cifar-10-batches-py' 
elif dataset_name == 'cifar100':
    dataset_path = '/mnt/zfj/dataset/cifar-100-python'
elif dataset_name == 'web-aircraft':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-aircraft'
elif dataset_name == 'web-bird':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-bird'
elif dataset_name == 'web-car':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-car'
elif dataset_name == 'web-car-del15':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-car-del15'
elif dataset_name == 'web-bird-del23':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-bird-del23'
elif dataset_name == 'web-aircraft-del21':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-aircraft-del21'
elif dataset_name == 'clothing1m':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/clothing1m'
elif dataset_name == 'webvision':
    dataset_path = '/mnt/home/zfj/'
# 检查是否有可用的 GPU，并使用 cuda:1
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



########################################################################################### 数据处理
import sys
sys.path.append('/mnt/zfj/projects/phd/projects_phd/CWU')
if dataset_name == 'cifar10' or dataset_name == 'cifar100':
    from dataloader import dataloader_cifar as dataloader
    loader = dataloader.cifar_dataloader(dataset_name, noise_mode='sym', noise_ratio=0, batch_size=64, num_workers=8, 
                                        root_dir=dataset_path, model='dino')
    train_loader = loader.run('eval')
    test_loader = loader.run('test')

    noise_idx = train_loader.dataset.noise_idx
    noise_label = torch.tensor(train_loader.dataset.noise_label).to(device)
    clean_label = torch.tensor(train_loader.dataset.clean_label).to(device)
    if dataset_name == 'cifar10':
        num_class = 10
    elif dataset_name == 'cifar100':
        num_class = 100





import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 确保保存路径存在
os.makedirs('/mnt/zfj/exp_results/tmp_graph', exist_ok=True)

# 转换为 numpy
noise_label_np = noise_label.cpu().numpy()
clean_label_np = clean_label.cpu().numpy()

if dataset_name == 'cifar10':
        num_class = 10
elif dataset_name == 'cifar100':
    num_class = 100

# 构建混淆矩阵：行 = 噪声标签，列 = 真实标签
conf_mat = np.zeros((num_class, num_class), dtype=int)
for i in range(len(noise_label_np)):
    nl = int(noise_label_np[i])
    cl = int(clean_label_np[i])
    if 0 <= nl < num_class and 0 <= cl < num_class:
        conf_mat[nl, cl] += 1

# 每行归一化
row_sums = conf_mat.sum(axis=1, keepdims=True)
conf_mat_norm = np.divide(conf_mat, row_sums, out=np.zeros_like(conf_mat, dtype=float), where=row_sums != 0)

# 预定义颜色：每个真实类一个固定颜色
# colors = plt.cm.tab20(np.linspace(0, 1, num_class))
# 替换原来的 colors = plt.cm.tab20(...)
import seaborn as sns
colors = np.array(sns.color_palette("deep", n_colors=num_class))

plt.figure(figsize=(6, 4))

# 对每个噪声标签（每一行）单独处理
for nl in range(num_class):
    proportions = conf_mat_norm[nl]  # shape: (num_class,)
    # 获取排序索引：按占比降序
    sorted_indices = np.argsort(-proportions)  # 负号 → 降序
    sorted_props = proportions[sorted_indices]
    sorted_colors = colors[sorted_indices]

    left = 0.0
    for idx, prop in enumerate(sorted_props):
        if prop > 0:  # 只画非零部分（可选）
            plt.barh(
                y=nl,
                width=prop,
                left=left,
                color=sorted_colors[idx],
                edgecolor='none'  # 去掉柱子之间的细线
            )
            left += prop

# 设置坐标轴
plt.ylabel('Noisy Label', fontsize=18)
plt.xlabel('Proportion of True Labels', fontsize=18)
# plt.title(f'Distribution of True Labels within Each Noisy Label ({dataset_name}) [Horizontal, Sorted]', fontsize=20)
plt.yticks(np.arange(num_class))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.xlim(0, 1)

# 去掉边框（spines）
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 设置坐标轴刻度字体大小
ax.tick_params(axis='x', labelsize=16)   # x 轴刻度
ax.tick_params(axis='y', labelsize=16)   # y 轴刻度

# 去掉网格线（可选，你没提，但通常去边框时也去网格）
# plt.grid(False)

# 图例：为了保持颜色-类别的对应，仍用原始顺序（否则图例混乱）
legend_elements = [plt.Rectangle((0,0),1,1, color=colors[i], label=f'True {i}') for i in range(num_class)]
legend = plt.legend(
    handles=legend_elements,
    title='True Label',
    # loc='upper center',
    # bbox_to_anchor=(0.5, -0.15),  # 放在图下方中央
    # ncol=num_class,   
    loc='center left',               # 图例相对于 bbox_to_anchor 的位置
    bbox_to_anchor=(1.02, 0.5),
    ncol=1,   
    fontsize=14
)
legend.get_title().set_fontsize(16)

# 之前的代码保持不变，直到保存图像部分

plt.tight_layout()
plt.savefig(f'/mnt/zfj/exp_results/tmp_graph/{dataset_name}_idn_test.pdf', format='pdf', dpi=300, bbox_inches='tight')


# ========== 单独保存图例为一张图 ==========
# import matplotlib.pyplot as plt
# import numpy as np

# num_class = 10
# colors = plt.cm.tab20(np.linspace(0, 1, num_class))

# # 创建一个仅用于图例的空图
# fig_legend = plt.figure(figsize=(12, 2))  # 宽度足够放下10个标签

# # 构建图例元素（与主图一致）
# legend_elements = [plt.Rectangle((0,0),1,1, color=colors[i], label=f'True {i}') for i in range(num_class)]

# # 添加图例到空图（居中）
# fig_legend.legend(
#     handles=legend_elements,
#     loc='center',
#     ncol=num_class,
#     fontsize=18,
#     frameon=False,
#     title='True Label',
#     title_fontsize=20
# )

# # 去掉坐标轴（因为不需要）
# plt.axis('off')

# # 之前的代码保持不变，直到保存图像部分

# plt.tight_layout()
# plt.savefig(f'/mnt/zfj/exp_results/tmp_graph/legend.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.show()



# ========== 新增：统计每个噪声标签下真实标签占比前三 ==========
print(f"\nTop-3 true label proportions for each noisy label in {dataset_name}:\n")
top3_stats = {}

for nl in range(num_class):
    proportions = conf_mat_norm[nl]  # shape: (num_class,)
    # 获取排序索引（降序）
    sorted_indices = np.argsort(-proportions)
    top3_indices = sorted_indices[:3]
    top3_props = proportions[top3_indices]

    # 过滤掉占比为0的项（可选）
    valid_mask = top3_props > 0
    top3_indices = top3_indices[valid_mask]
    top3_props = top3_props[valid_mask]

    top3_stats[nl] = list(zip(top3_indices, top3_props))

    # 打印结果
    print(f"Noisy Label {nl}:")
    for rank, (true_label, prop) in enumerate(top3_stats[nl], 1):
        print(f"  Rank {rank}: True Label {true_label} ({prop:.3f})")
    print()


# ========== 新增：统计 noise-dominant classes ==========
noise_dominant_count = 0
noise_dominant_classes = []

for nl in range(num_class):
    proportions = conf_mat_norm[nl]
    # 找到占比最高的真实标签
    top_true_label = np.argmax(proportions)  # 返回最大值的索引
    top_prop = proportions[top_true_label]

    # 判断：如果最高占比的真实标签 ≠ 噪声标签 nl，则为 noise-dominant
    if top_true_label != nl:
        noise_dominant_count += 1
        noise_dominant_classes.append((nl, top_true_label, top_prop))

print(f"\nNumber of noise-dominant classes: {noise_dominant_count} / {num_class}")
print("Noise-dominant classes (noisy_label → dominant_true_label [prop]):")
for nl, true_lbl, prop in noise_dominant_classes:
    print(f"  Noisy {nl} → True {true_lbl} ({prop:.3f})")