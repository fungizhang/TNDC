import os
import torch
import numpy as np
import json
import sys
from tqdm import tqdm
import random
from collections import defaultdict
import hdbscan
from scipy.spatial.distance import correlation
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")




# -------------------------- 命令行参数解析 --------------------------
parser = argparse.ArgumentParser(description="Noise Label Correction with HDBSCAN")
parser.add_argument('--dataset_name', type=str, default='cifar100',
                    choices=['cifar10', 'cifar100', 'tiny_imagenet', 'web-aircraft', 'web-bird', 'web-car'],
                    help='Name of the dataset')
parser.add_argument('--noise_mode', type=str, default='idn',
                    choices=['sym', 'asym', 'idn', 'asym_var'],
                    help='Type of noise: symmetric (sym) or asymmetric (asym)')
parser.add_argument('--noise_ratio', type=float, default=0.8,
                    help='Noise ratio (e.g., 0.2 for 20%)')

args = parser.parse_args()

# 使用命令行参数
dataset_name = args.dataset_name
noise_mode = args.noise_mode
noise_ratio = args.noise_ratio



# 全局参数配置
# sys.path.append('/mnt/zfj/projects/TNDC')
# dataset_name = 'cifar100'  # 可切换为 'cifar10'/'stanford_cars' 等
feature_dir = f'./saved_features/{dataset_name}_features_no_aug.pth'
root_path = '/mnt/zfj/projects/phd/projects_phd/CWU/saved/saved_features'
os.makedirs(root_path, exist_ok=True)

# -------------------------- 1. 数据集路径与设备配置 --------------------------
# 数据集路径映射
dataset_path_map = {
    'cifar10': './datasets/cifar-10-batches-py',
    'cifar100': './datasets/cifar-100-python',
    'tiny_imagenet': '/mnt/zfj/dataset/tiny-imagenet-200',
    'web-aircraft': '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-aircraft',
    'web-bird': '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-bird',
    'web-car': '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-car'
}
dataset_path = dataset_path_map.get(dataset_name, '')
if not dataset_path:
    raise ValueError(f"未配置 {dataset_name} 的数据集路径")

# 设备配置（优先GPU）
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# -------------------------- 2. 数据加载（含噪声标签与干净标签） --------------------------
if dataset_name in ['cifar10', 'cifar100']:
    from dataloader import dataloader_cifar as dataloader
    # 加载带噪声的CIFAR数据集（asym模式，20%噪声率）
    loader = dataloader.cifar_dataloader(
        dataset_name,
        noise_mode=noise_mode,          # ← 使用 args.noise_mode
        noise_ratio=noise_ratio,        # ← 使用 args.noise_ratio
        batch_size=64,
        num_workers=8,
        root_dir=dataset_path,
        model='dino'
    )
    train_loader = loader.run('train')
    test_loader = loader.run('test')

    # 提取关键标签信息（噪声标签/干净标签/类别数）
    noise_label = torch.tensor(train_loader.dataset.noise_label).to(device)
    clean_label = torch.tensor(train_loader.dataset.clean_label).to(device)
    num_class = 10 if dataset_name == 'cifar10' else 100
    print(f"数据集加载完成：{dataset_name}，样本数：{len(train_loader.dataset)}，类别数：{num_class}\n")

elif dataset_name == "tiny_imagenet":
    print("Loading Tiny-ImageNet...")
    import sys
    sys.path.append('/mnt/zfj/projects/phd/projects_phd/DeFT-main')
    from utils.config import _C as cfg
    from dataloader import dataloader_tiny_imagenet as dataloader

    cfg.defrost()
    cfg.merge_from_file('/mnt/zfj/projects/phd/projects_phd/DeFT-main/config/PEFT/tiny_imagenet.yaml')
    train_loader, eval_loader, test_loader = dataloader.build_loader(cfg)

    noise_idx = eval_loader.dataset.noise_idx
    noise_label = torch.tensor(eval_loader.dataset.noise_label).to(device)
    clean_label = torch.tensor(eval_loader.dataset.clean_label).to(device)
    num_class = 200
    print(f"数据集加载完成：{dataset_name}，样本数：{len(train_loader.dataset)}，类别数：{num_class}\n")

else:
    raise NotImplementedError(f"{dataset_name} 的数据加载逻辑未实现")

# -------------------------- 3. 特征加载 --------------------------
checkpoint = torch.load(feature_dir)
features_tensor = checkpoint['features'].to(device)  # 形状：(样本数, 特征维度)
labels_tensor = checkpoint['labels'].to(device)      # 原始标签（备用）
# labels_tensor = checkpoint['clean_labels'].to(device)      # 原始标签（备用）
print(f"特征加载完成：特征形状 {features_tensor.shape}，标签形状 {labels_tensor.shape}\n")

# -------------------------- 4. 功能1：噪声标签类的正确标签占比分析 --------------------------
def analyze_noise_class_accuracy(clean_labels, noise_labels, num_classes):
    """统计每个噪声标签类中，真实标签与噪声标签一致的样本占比"""
    print("===== 开始噪声标签类正确标签占比分析 =====")
    clean_labels_np = clean_labels.cpu().numpy()
    noise_labels_np = noise_labels.cpu().numpy()
    
    # 按噪声标签分组统计
    noise_class_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for true_lbl, noisy_lbl in zip(clean_labels_np, noise_labels_np):
        noise_class_stats[noisy_lbl]['total'] += 1
        if true_lbl == noisy_lbl:
            noise_class_stats[noisy_lbl]['correct'] += 1
    
    # 输出每个噪声标签类的结果
    for noisy_class in range(num_classes):
        stats = noise_class_stats[noisy_class]
        total = stats['total']
        if total == 0:
            print(f"噪声标签类 {noisy_class}：无样本")
            continue
        correct_ratio = (stats['correct'] / total) * 100
        print(f"噪声标签类 {noisy_class}：总样本数 {total}，正确标签占比 {correct_ratio:.2f}%")
    print("===== 噪声标签类分析完成 =====\n")



# -------------------------- 5. 功能2：HDBSCAN聚类分析（含簇内标签正确比例） --------------------------

def hdbscan_cluster_analysis(features, noise_labels, clean_labels, num_classes, K=10, N=3, THRESHOLD=0.6, MIN_REPRESENTATIVE_SAMPLES=1000):
    """
    对每个噪声标签类执行HDBSCAN聚类，检查 top-K 簇，
    合并所有满足 THRESHOLD 的簇，直到总样本 ≥ MIN_REPRESENTATIVE_SAMPLES。
    """
    print("===== 开始HDBSCAN聚类分析（多簇合并模式） =====")
    noise_labels_np = noise_labels.cpu().numpy()
    clean_labels_np = clean_labels.cpu().numpy()
    features_np = features.cpu().numpy()
    
    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(noise_labels_np):
        label_to_indices[lbl].append(idx)
    
    cluster_analysis_results = {}
    total_checked_samples_per_class = {}

    for noisy_label in tqdm(label_to_indices.keys(), desc="Processing classes for HDBSCAN"):
        sample_indices = label_to_indices[noisy_label]
        if len(sample_indices) < K * N:
            print(f"跳过类 {noisy_label}（样本数不足：{len(sample_indices)} < {K*N}）")
            total_checked_samples_per_class[noisy_label] = 0
            continue
        
        class_features = features_np[sample_indices]
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            metric='correlation',
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(class_features)
        
        cluster_counts = defaultdict(int)
        for lbl in cluster_labels:
            if lbl != -1:
                cluster_counts[lbl] += 1
        
        if not cluster_counts:
            print(f"类 {noisy_label} 无有效簇")
            total_checked_samples_per_class[noisy_label] = 0
            continue
        
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:K]
        top_cluster_ids = [cid for cid, _ in sorted_clusters]
        
        cluster_details = []
        checked_samples_count = 0
        qualified_clusters = []  # 存储合格簇的全局索引
        total_qualified_samples = 0

        for cluster_id in top_cluster_ids:
            cluster_mask = (cluster_labels == cluster_id)
            cluster_in_class_indices = np.where(cluster_mask)[0]
            cluster_global_indices = [sample_indices[i] for i in cluster_in_class_indices]
            
            sample_count = min(N, len(cluster_global_indices))
            if sample_count == 0:
                continue
            sampled_global_indices = np.random.choice(cluster_global_indices, size=sample_count, replace=False)
            sampled_true_labels = [clean_labels_np[idx] for idx in sampled_global_indices]
            sampled_correct = sum(1 for tl in sampled_true_labels if tl == noisy_label)
            sampled_correct_ratio = sampled_correct / sample_count
            
            checked_samples_count += sample_count

            cluster_details.append({
                "cluster_id": cluster_id,
                "total_samples": len(cluster_global_indices),
                "sampled_count": sample_count,
                "sampled_correct_ratio": sampled_correct_ratio,
                "qualified": sampled_correct_ratio >= THRESHOLD
            })

            # 如果合格，加入候选
            if sampled_correct_ratio >= THRESHOLD:
                qualified_clusters.append(cluster_global_indices)
                total_qualified_samples += len(cluster_global_indices)
                # 提前终止条件：足够多的代表样本
                if total_qualified_samples >= MIN_REPRESENTATIVE_SAMPLES:
                    break

        found_representative = total_qualified_samples > 0

        cluster_analysis_results[noisy_label] = {
            "total_samples_in_class": len(sample_indices),
            "cluster_details": cluster_details,
            "found_representative": found_representative,
            "total_checked_samples": checked_samples_count,
            "total_qualified_samples": total_qualified_samples,
            "num_qualified_clusters": len(qualified_clusters)
        }
        total_checked_samples_per_class[noisy_label] = checked_samples_count
    
    # 输出结果
    for noisy_label, result in cluster_analysis_results.items():
        print(f"\n类（噪声标签）{noisy_label}：总样本数 {result['total_samples_in_class']}")
        print(f"  ✅ 找到合格簇: {'是' if result['found_representative'] else '否'}")
        print(f"  📦 合格簇总样本数: {result['total_qualified_samples']}（来自 {result['num_qualified_clusters']} 个簇）")
        print(f"  🔍 总共检查样本数: {result['total_checked_samples']}")
        for i, cluster in enumerate(result['cluster_details'], 1):
            mark = "✅" if cluster['qualified'] else "❌"
            print(f"    {mark} 簇 {i}（ID: {cluster['cluster_id']}）：总样本 {cluster['total_samples']}，抽样 {cluster['sampled_count']} 个，正确比例 {cluster['sampled_correct_ratio']:.4f}")
    
    print("\n===== 各类检查样本量汇总 =====")
    total_overall_checked = sum(total_checked_samples_per_class.values())
    print(f"总计检查样本数: {total_overall_checked}")
    for lbl in sorted(total_checked_samples_per_class.keys()):
        print(f"  类 {lbl}: {total_checked_samples_per_class[lbl]} 个样本")
    print("===== HDBSCAN聚类分析完成 =====\n")
    
    return cluster_analysis_results, total_checked_samples_per_class

# -------------------------- 6. 功能3：基于质心的标签重分配与准确率统计（新增标签修正后样本数） --------------------------
### 取每类前top_ratio

# def reassign_labels_by_centroid(features, noise_labels, clean_labels, K=10, N=10, THRESHOLD=0.6, top_ratio=0.5, MIN_REPRESENTATIVE_SAMPLES=1000):
#     """
#     合并多个满足条件的簇作为代表样本，直到总样本 ≥ MIN_REPRESENTATIVE_SAMPLES。
#     """
#     print("===== 开始基于质心的标签重分配（多簇合并 + 高置信度修正） =====")
#     noise_labels_np = noise_labels.cpu().numpy()
#     clean_labels_np = clean_labels.cpu().numpy()
#     features_np = features.cpu().numpy()
#     total_samples = features_np.shape[0]

#     new_labels_np = noise_labels_np.copy()
#     label_to_indices = defaultdict(list)
#     for idx, lbl in enumerate(noise_labels_np):
#         label_to_indices[lbl].append(idx)

#     class_centroids = {}
#     class_representative_count = {}

#     for noisy_label in tqdm(label_to_indices.keys(), desc="Finding representative clusters (multi-cluster)"):
#         sample_indices = label_to_indices[noisy_label]
#         if len(sample_indices) < K * N:
#             print(f"跳过类 {noisy_label}（样本数不足：{len(sample_indices)}）")
#             continue

#         class_features = features_np[sample_indices]
#         clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='correlation', cluster_selection_method='eom')
#         cluster_labels = clusterer.fit_predict(class_features)

#         cluster_counts = defaultdict(int)
#         for lbl in cluster_labels:
#             if lbl != -1:
#                 cluster_counts[lbl] += 1

#         if not cluster_counts:
#             centroid = np.mean(class_features, axis=0)
#             class_centroids[noisy_label] = centroid
#             class_representative_count[noisy_label] = len(sample_indices)
#             continue

#         sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:K]
#         top_cluster_ids = [cid for cid, _ in sorted_clusters]

#         qualified_global_indices = []
#         total_qualified = 0

#         for cluster_id in top_cluster_ids:
#             cluster_mask = (cluster_labels == cluster_id)
#             cluster_in_class_indices = np.where(cluster_mask)[0]
#             cluster_global_indices = [sample_indices[i] for i in cluster_in_class_indices]
#             rep_count = len(cluster_global_indices)

#             sample_count = min(N, rep_count)
#             if sample_count == 0:
#                 continue
#             sampled_indices = np.random.choice(cluster_global_indices, size=sample_count, replace=False)
#             sampled_true_labels = [clean_labels_np[idx] for idx in sampled_indices]
#             correct_ratio = sum(1 for tl in sampled_true_labels if tl == noisy_label) / sample_count

#             if correct_ratio >= THRESHOLD:
#                 qualified_global_indices.extend(cluster_global_indices)
#                 total_qualified += rep_count
#                 if total_qualified >= MIN_REPRESENTATIVE_SAMPLES:
#                     break  # 达到目标，提前退出

#         if qualified_global_indices:
#             # 使用所有合格簇的样本计算质心
#             centroid = np.mean(features_np[qualified_global_indices], axis=0)
#             class_centroids[noisy_label] = centroid
#             class_representative_count[noisy_label] = len(qualified_global_indices)
#         else:
#             # 无合格簇，回退到全类均值
#             centroid = np.mean(class_features, axis=0)
#             class_centroids[noisy_label] = centroid
#             class_representative_count[noisy_label] = len(sample_indices)

#     if not class_centroids:
#         raise ValueError("无任何类找到代表簇，无法进行标签重分配")

#     all_classes = list(class_centroids.keys())
#     print(f"\n找到 {len(all_classes)} 个类的代表簇（可能由多簇合并），开始计算置信度...")

#     # 后续逻辑不变：计算距离、重分配等
#     sample_distances = []
#     for i in range(total_samples):
#         feat = features_np[i]
#         min_dist = float('inf')
#         best_class = -1
#         for cls in all_classes:
#             dist = correlation(feat, class_centroids[cls])
#             if dist < min_dist:
#                 min_dist = dist
#                 best_class = cls
#         sample_distances.append((i, best_class, min_dist))

#     pred_class_to_samples = defaultdict(list)
#     for idx, pred_cls, dist in sample_distances:
#         pred_class_to_samples[pred_cls].append((idx, dist))

#     for pred_cls, samples in pred_class_to_samples.items():
#         if not samples:
#             continue
#         samples_sorted = sorted(samples, key=lambda x: x[1])
#         n_keep = max(1, int(len(samples_sorted) * top_ratio))
#         high_conf_samples = samples_sorted[:n_keep]
#         for idx, _ in high_conf_samples:
#             new_labels_np[idx] = pred_cls

#     # 统计指标（略，与原逻辑一致）
#     overall_correct = sum(1 for i in range(total_samples) if new_labels_np[i] == clean_labels_np[i])
#     overall_accuracy = overall_correct / total_samples

#     corrected_class_sample_count = defaultdict(int)
#     for lbl in new_labels_np:
#         corrected_class_sample_count[lbl] += 1

#     class_accuracies = defaultdict(lambda: {
#         'correct': 0, 'total': 0, 'accuracy': 0.0,
#         'representative_sample_count': 0,
#         'corrected_sample_count': 0
#     })

#     for i in range(total_samples):
#         true_lbl = clean_labels_np[i]
#         pred_lbl = new_labels_np[i]
#         class_accuracies[true_lbl]['total'] += 1
#         if true_lbl == pred_lbl:
#             class_accuracies[true_lbl]['correct'] += 1
#         if true_lbl in class_representative_count:
#             class_accuracies[true_lbl]['representative_sample_count'] = class_representative_count[true_lbl]
#         class_accuracies[true_lbl]['corrected_sample_count'] = corrected_class_sample_count.get(true_lbl, 0)

#     for class_label in class_accuracies:
#         total = class_accuracies[class_label]['total']
#         if total > 0:
#             class_accuracies[class_label]['accuracy'] = class_accuracies[class_label]['correct'] / total

#     print(f"\n=== 多簇合并 + 高置信度修正统计（top {int(top_ratio*100)}%）===")
#     print(f"整体准确率: {overall_accuracy:.4f}")
#     print(f"总样本数: {total_samples}")

#     print("\n每个类的详细统计:")
#     for class_label in sorted(class_accuracies.keys()):
#         acc = class_accuracies[class_label]['accuracy']
#         correct = class_accuracies[class_label]['correct']
#         total = class_accuracies[class_label]['total']
#         rep_count = class_accuracies[class_label]['representative_sample_count']
#         corrected_count = class_accuracies[class_label]['corrected_sample_count']
#         rep_count_str = str(rep_count) if rep_count > 0 else "无"
#         print(f"类 {class_label}: 准确率 {acc:.4f} ({correct}/{total})，代表样本数 {rep_count_str}，修正后样本数 {corrected_count}")

#     results = {
#         'overall_accuracy': overall_accuracy,
#         'total_samples': total_samples,
#         'class_accuracies': dict(class_accuracies),
#         'class_centroids': {k: v.tolist() for k, v in class_centroids.items()},
#         'class_representative_counts': class_representative_count,
#         'corrected_class_sample_counts': corrected_class_sample_count
#     }

#     print("===== 标签重分配（多簇合并）完成 =====\n")
#     return results, new_labels_np, noise_labels_np



def reassign_labels_by_centroid(features, noise_labels, clean_labels, K=10, N=10, THRESHOLD=0.6, top_ratio=0.5, MIN_REPRESENTATIVE_SAMPLES=1000):
    print("===== 开始基于质心的标签重分配（多簇合并 + 高置信度修正） =====")
    noise_labels_np = noise_labels.cpu().numpy()
    clean_labels_np = clean_labels.cpu().numpy()
    features_np = features.cpu().numpy()
    total_samples = features_np.shape[0]

    new_labels_np = noise_labels_np.copy()
    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(noise_labels_np):
        label_to_indices[lbl].append(idx)

    class_centroids = {}
    class_representative_count = {}

    for noisy_label in tqdm(label_to_indices.keys(), desc="Finding representative clusters (multi-cluster)"):
        sample_indices = label_to_indices[noisy_label]
        if len(sample_indices) < K * N:
            print(f"跳过类 {noisy_label}（样本数不足：{len(sample_indices)}）")
            continue

        class_features = features_np[sample_indices]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='correlation', cluster_selection_method='eom')
        cluster_labels = clusterer.fit_predict(class_features)

        cluster_counts = defaultdict(int)
        for lbl in cluster_labels:
            if lbl != -1:
                cluster_counts[lbl] += 1

        if not cluster_counts:
            centroid = np.mean(class_features, axis=0)
            class_centroids[noisy_label] = centroid
            class_representative_count[noisy_label] = len(sample_indices)
            continue

        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:K]
        top_cluster_ids = [cid for cid, _ in sorted_clusters]

        qualified_global_indices = []
        for cluster_id in top_cluster_ids:
            cluster_mask = (cluster_labels == cluster_id)
            cluster_in_class_indices = np.where(cluster_mask)[0]
            cluster_global_indices = [sample_indices[i] for i in cluster_in_class_indices]
            rep_count = len(cluster_global_indices)

            sample_count = min(N, rep_count)
            if sample_count == 0:
                continue
            sampled_indices = np.random.choice(cluster_global_indices, size=sample_count, replace=False)
            sampled_true_labels = [clean_labels_np[idx] for idx in sampled_indices]
            correct_ratio = sum(1 for tl in sampled_true_labels if tl == noisy_label) / sample_count

            if correct_ratio >= THRESHOLD:
                qualified_global_indices.extend(cluster_global_indices)

        # >>>>>>>>>> 新增：限制代表样本最多为 MIN_REPRESENTATIVE_SAMPLES <<<<<<<<<<
        if len(qualified_global_indices) > MIN_REPRESENTATIVE_SAMPLES:
            qualified_global_indices = np.random.choice(
                qualified_global_indices, size=MIN_REPRESENTATIVE_SAMPLES, replace=False
            ).tolist()

        if qualified_global_indices:
            centroid = np.mean(features_np[qualified_global_indices], axis=0)
            class_centroids[noisy_label] = centroid
            class_representative_count[noisy_label] = len(qualified_global_indices)
        else:
            centroid = np.mean(class_features, axis=0)
            class_centroids[noisy_label] = centroid
            class_representative_count[noisy_label] = len(sample_indices)

    # ... 后续逻辑保持不变 ...
    if not class_centroids:
        raise ValueError("无任何类找到代表簇，无法进行标签重分配")

    all_classes = list(class_centroids.keys())
    print(f"\n找到 {len(all_classes)} 个类的代表簇（可能由多簇合并），开始计算置信度...")

    # 后续逻辑不变：计算距离、重分配等
    sample_distances = []
    for i in range(total_samples):
        feat = features_np[i]
        min_dist = float('inf')
        best_class = -1
        for cls in all_classes:
            dist = correlation(feat, class_centroids[cls])
            if dist < min_dist:
                min_dist = dist
                best_class = cls
        sample_distances.append((i, best_class, min_dist))

    pred_class_to_samples = defaultdict(list)
    for idx, pred_cls, dist in sample_distances:
        pred_class_to_samples[pred_cls].append((idx, dist))

    for pred_cls, samples in pred_class_to_samples.items():
        if not samples:
            continue
        samples_sorted = sorted(samples, key=lambda x: x[1])
        n_keep = max(1, int(len(samples_sorted) * top_ratio))
        high_conf_samples = samples_sorted[:n_keep]
        for idx, _ in high_conf_samples:
            new_labels_np[idx] = pred_cls

    # 统计指标（略，与原逻辑一致）
    overall_correct = sum(1 for i in range(total_samples) if new_labels_np[i] == clean_labels_np[i])
    overall_accuracy = overall_correct / total_samples

    corrected_class_sample_count = defaultdict(int)
    for lbl in new_labels_np:
        corrected_class_sample_count[lbl] += 1

    class_accuracies = defaultdict(lambda: {
        'correct': 0, 'total': 0, 'accuracy': 0.0,
        'representative_sample_count': 0,
        'corrected_sample_count': 0
    })

    for i in range(total_samples):
        true_lbl = clean_labels_np[i]
        pred_lbl = new_labels_np[i]
        class_accuracies[true_lbl]['total'] += 1
        if true_lbl == pred_lbl:
            class_accuracies[true_lbl]['correct'] += 1
        if true_lbl in class_representative_count:
            class_accuracies[true_lbl]['representative_sample_count'] = class_representative_count[true_lbl]
        class_accuracies[true_lbl]['corrected_sample_count'] = corrected_class_sample_count.get(true_lbl, 0)

    for class_label in class_accuracies:
        total = class_accuracies[class_label]['total']
        if total > 0:
            class_accuracies[class_label]['accuracy'] = class_accuracies[class_label]['correct'] / total

    print(f"\n=== 多簇合并 + 高置信度修正统计（top {int(top_ratio*100)}%）===")
    print(f"整体准确率: {overall_accuracy:.4f}")
    print(f"总样本数: {total_samples}")

    print("\n每个类的详细统计:")
    for class_label in sorted(class_accuracies.keys()):
        acc = class_accuracies[class_label]['accuracy']
        correct = class_accuracies[class_label]['correct']
        total = class_accuracies[class_label]['total']
        rep_count = class_accuracies[class_label]['representative_sample_count']
        corrected_count = class_accuracies[class_label]['corrected_sample_count']
        rep_count_str = str(rep_count) if rep_count > 0 else "无"
        print(f"类 {class_label}: 准确率 {acc:.4f} ({correct}/{total})，代表样本数 {rep_count_str}，修正后样本数 {corrected_count}")

    results = {
        'overall_accuracy': overall_accuracy,
        'total_samples': total_samples,
        'class_accuracies': dict(class_accuracies),
        'class_centroids': {k: v.tolist() for k, v in class_centroids.items()},
        'class_representative_counts': class_representative_count,
        'corrected_class_sample_counts': corrected_class_sample_count
    }

    print("===== 标签重分配（多簇合并）完成 =====\n")
    return results, new_labels_np, noise_labels_np


# -------------------------- 7. 功能4：统计修正后每个类中真实类别的分布（前三占比） --------------------------
def analyze_corrected_class_composition(new_labels, clean_labels, num_classes):
    """
    分析标签修正后，每个新类中包含哪些真实类别的样本，输出占比前三的真实类别，
    并统计 Noise-dominant class 的数量（即：修正后类中占比最高的真实类 ≠ 修正后类标签）。
    """
    print("===== 开始分析标签修正后各类的组成（真实类别分布） =====")
    
    # 构建：每个修正后的类 -> 其包含的所有真实标签
    corrected_class_to_true = defaultdict(list)
    for pred_lbl, true_lbl in zip(new_labels, clean_labels):
        corrected_class_to_true[pred_lbl].append(true_lbl)
    
    noise_dominant_count = 0  # 计数器

    # 遍历每个修正后的类
    for corrected_label in sorted(corrected_class_to_true.keys()):
        true_labels_in_class = corrected_class_to_true[corrected_label]
        total = len(true_labels_in_class)
        
        # 统计各真实类别的数量
        true_label_counter = defaultdict(int)
        for tl in true_labels_in_class:
            true_label_counter[tl] += 1
        
        # 按数量排序，取第一（主导真实类）
        sorted_true_labels = sorted(true_label_counter.items(), key=lambda x: x[1], reverse=True)
        top1_true_label, top1_count = sorted_true_labels[0]
        top1_ratio = top1_count / total

        # 判断是否为 Noise-dominant class
        if top1_true_label != corrected_label:
            noise_dominant_count += 1
            is_noise_dominant = "🔴 Noise-dominant"
        else:
            is_noise_dominant = "🟢 Clean-dominant"

        # 输出前三
        top3 = sorted_true_labels[:3]
        print(f"\n修正后类 {corrected_label}（共 {total} 个样本）{is_noise_dominant}")
        for i, (true_lbl, count) in enumerate(top3):
            ratio = count / total
            print(f"  第{i+1}大成分: 真实类 {true_lbl}, 数量 {count}, 占比 {ratio:.3f} ({ratio*100:.1f}%)")
        
        # 其他类提示
        if len(sorted_true_labels) > 3:
            others_count = sum(item[1] for item in sorted_true_labels[3:])
            others_ratio = others_count / total
            print(f"  其他 {len(sorted_true_labels) - 3} 个类合计: {others_count} 样本, 占比 {others_ratio:.3f} ({others_ratio*100:.1f}%)")
    
    print(f"\n📊 总结：共有 {noise_dominant_count} 个 Noise-dominant class（主导真实类 ≠ 修正后类标签）")
    print("===== 修正后类组成分析完成 =====\n")


# -------------------------- 新增功能5：按置信度区间统计修正前后准确率 --------------------------
def analyze_accuracy_by_confidence_intervals(
    features, 
    noise_labels, 
    clean_labels, 
    new_labels, 
    class_centroids,
    num_classes
):
    """
    将所有样本按到预测类质心的距离（置信度）排序，划分成10个等比例区间，
    统计每个区间内：
        - 修正前（noise_labels）的准确率
        - 修正后（new_labels）的准确率
    
    并额外统计：前10%、20%、...、100% 累计样本的修正后整体准确率。
    """
    print("===== 开始按置信度区间分析修正前后准确率 =====")
    
    features_np = features.cpu().numpy()
    noise_labels_np = noise_labels.cpu().numpy()
    clean_labels_np = clean_labels.cpu().numpy()
    new_labels_np = np.array(new_labels)
    
    all_classes = list(class_centroids.keys())
    sample_distances = []  # [(index, distance), ...]

    for i in range(len(features_np)):
        feat = features_np[i]
        min_dist = float('inf')
        for cls in all_classes:
            dist = correlation(feat, class_centroids[cls])
            if dist < min_dist:
                min_dist = dist
        sample_distances.append((i, min_dist))
    
    # 按距离升序排序（高置信度在前）
    sample_distances.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in sample_distances]
    
    total_samples = len(sorted_indices)
    interval_size = total_samples // 10
    results_per_interval = []

    print(f"总样本数: {total_samples}，每区间约 {interval_size} 样本")
    
    for i in range(10):
        start = i * interval_size
        end = start + interval_size if i < 9 else total_samples
        interval_indices = sorted_indices[start:end]
        
        if len(interval_indices) == 0:
            continue
            
        clean_sub = clean_labels_np[interval_indices]
        noise_sub = noise_labels_np[interval_indices]
        new_sub = new_labels_np[interval_indices]
        
        acc_before = np.mean(clean_sub == noise_sub)
        acc_after = np.mean(clean_sub == new_sub)
        
        results_per_interval.append({
            'interval': f"{i*10}%–{(i+1)*10}%",
            'sample_count': len(interval_indices),
            'acc_before': acc_before,
            'acc_after': acc_after
        })
        
        print(f"区间 {i+1:2d} ({i*10:2d}–{(i+1)*10:2d}%): "
              f"样本数={len(interval_indices):4d}, "
              f"修正前准确率={acc_before:.4f}, "
              f"修正后准确率={acc_after:.4f}")
    
    # ==================== 新增：累计前 N% 的整体准确率 ====================
    print("\n===== 累计前 N% 高置信度样本的修正后整体准确率 =====")
    cumulative_accuracies = {}
    for p in range(10, 101, 10):
        n_samples = int(total_samples * p / 100)
        if n_samples == 0:
            acc = 0.0
        else:
            top_indices = sorted_indices[:n_samples]
            acc = np.mean(clean_labels_np[top_indices] == new_labels_np[top_indices])
        cumulative_accuracies[f"top_{p}pct"] = float(acc)
        print(f"前 {p:3d}% ({n_samples:6d} 样本): 修正后准确率 = {acc:.4f}")
    
    print("===== 置信度区间与累计准确率分析完成 =====\n")
    
    return {
        'interval_results': results_per_interval,
        'cumulative_accuracies': cumulative_accuracies
    }




# 执行功能1
analyze_noise_class_accuracy(clean_labels=clean_label, noise_labels=noise_label, num_classes=num_class)

# # 执行功能2（K=10个最大簇，N=10个抽样样本）
hdbscan_results = hdbscan_cluster_analysis(
    features=features_tensor,
    noise_labels=noise_label,
    clean_labels=clean_label,
    num_classes=num_class,
    K=10,
    N=1,
    THRESHOLD=0.6
)
# 执行功能3（阈值THRESHOLD=0.6，即抽样正确比例≥60%的簇为代表簇）
reassignment_results, new_labels_np, noise_labels_np = reassign_labels_by_centroid(
    features=features_tensor,
    noise_labels=noise_label,
    clean_labels=clean_label,
    K=10,
    N=1,
    THRESHOLD=0.6,
    top_ratio=0.9
)
# 执行功能4
# analyze_corrected_class_composition(
#     new_labels=np.array(new_labels_np),  # 来自功能3的 new_labels_np
#     clean_labels=clean_label.cpu().numpy(),
#     num_classes=num_class
# )

# 执行功能5
# 调用新增功能5（需从 reassign_labels_by_centroid 返回 class_centroids）
# 注意：reassign_labels_by_centroid 当前返回的是 centroids 的 list 形式，需转回 numpy
# 从 reassignment_results 中恢复 centroids 为 numpy 数组



# 执行区间准确率分析
# class_centroids_np = {
#     k: np.array(v) for k, v in reassignment_results['class_centroids'].items()
# }
# interval_analysis_results = analyze_accuracy_by_confidence_intervals(
#     features=features_tensor,
#     noise_labels=noise_label,
#     clean_labels=clean_label,
#     new_labels=new_labels_np,
#     class_centroids=class_centroids_np,
#     num_classes=num_class
# )


# print("===== 原噪声组成 =====\n")


# analyze_corrected_class_composition(
#     new_labels=np.array(noise_labels_np),  # 来自功能3的 new_labels_np
#     clean_labels=clean_label.cpu().numpy(),
#     num_classes=num_class
# )



# -------------------------- 8. 功能5：保存修正后的标签为JSON文件 --------------------------
import json
import os

# 确保目标目录存在
save_dir = f'./datasets/dino_mod/{dataset_name}'
os.makedirs(save_dir, exist_ok=True)

# 将 new_labels_np 转为列表
new_labels_list = new_labels_np.tolist()

# 保存路径
save_path = os.path.join(save_dir, f'dino_mod_labels_{noise_mode}_{noise_ratio}.json')

# 写入 JSON 文件
with open(save_path, 'w') as f:
    json.dump(new_labels_list, f)

print(f"\n✅ 修正后的标签已保存至: {save_path}")