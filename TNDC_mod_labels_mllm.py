import os
import torch
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
import hdbscan
from scipy.spatial.distance import correlation
import argparse
import warnings
from PIL import Image
import gc
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── 命令行参数 ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Noise Label Correction with HDBSCAN + Qwen-VL")
parser.add_argument('--dataset_name', type=str, default='cifar100',
                    choices=['cifar10', 'cifar100'],
                    help='Name of the dataset')
parser.add_argument('--noise_mode', type=str, default='idn',
                    choices=['sym', 'asym', 'idn', 'asym_var'])
parser.add_argument('--noise_ratio', type=float, default=0.2)
args = parser.parse_args()



# 使用命令行参数
dataset_name = args.dataset_name
noise_mode = args.noise_mode
noise_ratio = args.noise_ratio


# 全局参数配置
# sys.path.append('/mnt/zfj/projects/TNDC')
# dataset_name = 'cifar100'  # 可切换为 'cifar10'/'stanford_cars' 等
feature_dir = f'./saved_features/{dataset_name}_features_no_aug.pth'
# root_path = '/mnt/zfj/projects/phd/projects_phd/CWU/saved/saved_features'
# os.makedirs(root_path, exist_ok=True)

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

else:
    raise NotImplementedError(f"{dataset_name} 的数据加载逻辑未实现")

# -------------------------- 3. 特征加载 --------------------------
checkpoint = torch.load(feature_dir)
features_tensor = checkpoint['features'].to(device)  # 形状：(样本数, 特征维度)
labels_tensor = checkpoint['labels'].to(device)      # 原始标签（备用）
# labels_tensor = checkpoint['clean_labels'].to(device)      # 原始标签（备用）
print(f"特征加载完成：特征形状 {features_tensor.shape}，标签形状 {labels_tensor.shape}\n")

# ─── 路径与设备 ──────────────────────────────────────────────────────────────
root = "./datasets"
if dataset_name == 'cifar10':
    data_root = os.path.join(root, "cifar-10-batches-py")
    num_class = 10
elif dataset_name == 'cifar100':
    data_root = os.path.join(root, "cifar-100-python")
    num_class = 100
else:
    raise ValueError("Only support cifar10/cifar100 now")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── 加载类别名称（英文，最重要！） ──────────────────────────────────────────
if dataset_name == 'cifar10':
    cifar10_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    class_names = cifar10_names
else:  # cifar100
    # 你需要准备完整的 CIFAR-100 100个类名（这里只示例前几个）
    # 完整列表建议从官方 meta 文件读取，或直接硬编码
    cifar100_names = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm"
    ]
    class_names = cifar100_names

# ─── 载入 Qwen2.5-VL-0.5B-Instruct ───────────────────────────────────
# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/zfj/dataset/incremental_metrics.png"},
            {"type": "text", "text": "explain this photo"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
print("Qwen2.5-VL-7B-Instruct 载入完成！\n")

# ─── 准备所有训练图片路径 ───────────────────────────────────────────────────
# 方法1：如果你用的是官方 cifar 数据加载器，可以这样获取路径
# 但更常见的是你已经把 cifar 数据解压/转换成了图片文件

# 这里假设你已经把 cifar 转成图片格式，目录结构例如：
# ./datasets/cifar100_images/train/00000.png
# ./datasets/cifar100_images/train/00001.png
# ...

image_dir = f"./datasets/{dataset_name}_images/train"
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png','.jpg'))])

if len(image_paths) == 0:
    print("警告：没有找到图片文件！请先将 cifar 数据集解压成图片格式")
    # 你也可以选择直接从 pickle 加载数据并转为 PIL Image（见下方备选方案）

print(f"Found {len(image_paths)} training images.")

# 如果你还没有转成图片，可以使用下面这种内存加载方式（推荐）：
# （需要修改 dataloader 部分，让它返回 image 对象列表）

# ─── 功能：用 Qwen-VL 判断单张图是否属于该类 ────────────────────────────────
def qwen_judge_is_class(image: Image.Image, class_name: str, image_path: str = None, idx: int = None) -> bool:
    """
    增加图片路径/索引信息打印，方便追踪哪张图被判断为什么
    """
    try:
        prompt = f"""Question: This image shows a photo of {class_name}?
                    Please answer with only one word: True or False

                    Answer:"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        output_lower = output_text.lower()

        # ─── 打印完整信息 ───────────────────────────────────────────────
        info = []
        if idx is not None:
            info.append(f"idx={idx:6d}")
        if image_path:
            info.append(f"path={os.path.basename(image_path)}")
        info_str = " | ".join(info) if info else ""
        
        print(f"[Qwen-VL] {info_str}  →  class='{class_name}'  →  raw answer: {output_text!r}")
        # print(f"   lower: {output_lower}")

        # 判断逻辑（稍微更稳健一些）
        if "true" in output_lower and "false" not in output_lower:
            print("   → 判断结果: True\n")
            return True
        if "false" in output_lower and "true" not in output_lower:
            print("   → 判断结果: False\n")
            return False

        # 边界情况处理
        first_word = output_lower.split()[0] if output_lower.split() else ""
        if first_word in ["true", "yes", "1"]:
            print("   → 边界处理: True (first word)\n")
            return True
        if first_word in ["false", "no", "0"]:
            print("   → 边界处理: False (first word)\n")
            return False

        print("   → 无法明确判断，默认返回 False\n")
        return False

    except Exception as e:
        print(f"[Qwen-VL ERROR] {info_str}  →  {e}")
        return False
    finally:
        if 'inputs' in locals():
            del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─── 修改后的簇代表性判断函数 ────────────────────────────────────────────────
def is_representative_by_mllm(sample_indices, noisy_label_idx, clean_labels_np, n_sample=3, threshold=0.6):
    if len(sample_indices) == 0:
        return 0.0

    sampled_indices = np.random.choice(sample_indices, size=min(n_sample, len(sample_indices)), replace=False)

    true_count = 0           # MLLM 判断为 True 的数量
    match_count = 0          # MLLM判断 与 Actual 是否一致
    actual_correct = 0       # 真实正确的数量（用于对照）

    class_name = class_names[noisy_label_idx]

    print(f"\n=== 检查噪声标签类 {noisy_label_idx} ({class_name}) 的代表性 ===")
    print(f"  簇规模：总样本 {len(sample_indices)} 张   →   本次抽样 {len(sampled_indices)} 张\n")

    print("[Qwen-VL] 判断详情（抽样样本）:")

    results = []

    for idx in sampled_indices:
        actual_is_correct = (clean_labels_np[idx] == noisy_label_idx)
        if actual_is_correct:
            actual_correct += 1

        try:
            if len(image_paths) > 0 and idx < len(image_paths):
                img_path = image_paths[idx]
                img = Image.open(img_path).convert("RGB")
                is_class = qwen_judge_is_class(img, class_name, image_path=img_path, idx=idx)
            else:
                is_class = False  # 或者跳过
        except:
            is_class = False

        match = (is_class == actual_is_correct)
        if match:
            match_count += 1
        if is_class:
            true_count += 1

        status = "一致 ✅" if match else "不一致 ❌"
        filename = os.path.basename(image_paths[idx]) if idx < len(image_paths) else "???"
        
        line = f"  idx={idx:6d} | {filename:<12} →  MLLM={str(is_class):<5} | Actual={str(actual_is_correct):<5} → {status}"
        print(line)
        results.append((idx, is_class, actual_is_correct, match))

    mllm_ratio = true_count / len(sampled_indices) if sampled_indices.size > 0 else 0.0
    actual_ratio = actual_correct / len(sampled_indices) if sampled_indices.size > 0 else 0.0
    match_ratio = match_count / len(sampled_indices) if sampled_indices.size > 0 else 0.0

    print(f"\n→ MLLM 判断为 True 的比例: {mllm_ratio}")
    print(f"→ 实际 Actual 的比例: {actual_ratio}")
    
    if mllm_ratio >= threshold and actual_ratio >= 0.8:
        pass
    elif mllm_ratio < threshold and actual_ratio <= 0.2:
        pass
    elif mllm_ratio >= threshold and actual_ratio < 0.8:
        print(f"❌❌❌❌❌❌❌❌❌❌❌❌ 可能错领：{len(sample_indices)}")
    elif mllm_ratio < threshold and actual_ratio > 0.2:
        print(f"❌❌❌❌❌❌❌❌❌❌❌❌ 可能丢失：{len(sample_indices)}")
    
    # 大多数论文/方法用 MLLM 说“是该类”的比例 → 这里保持原样返回 mllm_ratio
    return mllm_ratio


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

# ─── HDBSCAN 聚类分析（唯一一次执行 MLLM 判断） ────────────────────────────────
def hdbscan_cluster_analysis(
    features,
    noise_labels,
    clean_labels,
    num_classes,
    K=10,
    N=5,
    THRESHOLD=0.6,
    MIN_REPRESENTATIVE_SAMPLES=1000
):
    """
    对每个噪声标签类进行 HDBSCAN 聚类 + MLLM 代表性判断，
    为每个类返回一个代表性样本索引列表（合格簇合并 or 全类样本）。
    这是唯一会调用 MLLM 的地方。
    """
    print("===== 开始 HDBSCAN 聚类分析（唯一一次 MLLM 判断） =====")
    noise_labels_np = noise_labels.cpu().numpy()
    clean_labels_np = clean_labels.cpu().numpy()
    features_np = features.cpu().numpy()
    
    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(noise_labels_np):
        label_to_indices[lbl].append(idx)
    
    cluster_analysis_results = {}
    total_checked_samples_per_class = {}
    representative_indices_per_class = {}  # noisy_label -> List[int] 代表性样本索引

    for noisy_label in tqdm(label_to_indices.keys(), desc="Processing classes for HDBSCAN"):
        sample_indices = label_to_indices[noisy_label]
        qualified_global_indices_all = []

        # 样本太少 → 直接使用全类
        if len(sample_indices) < K * N:
            print(f"类 {noisy_label} 样本太少({len(sample_indices)})，直接使用全类作为代表样本")
            representative_indices_per_class[noisy_label] = sample_indices.copy()
            total_checked_samples_per_class[noisy_label] = 0
            cluster_analysis_results[noisy_label] = {
                "total_samples_in_class": len(sample_indices),
                "found_representative": True,
                "total_qualified_samples": len(sample_indices),
                "note": "样本不足，直接使用全类"
            }
            continue

        # 正常聚类流程
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
            print(f"类 {noisy_label} 无有效簇，使用全类作为代表样本")
            representative_indices_per_class[noisy_label] = sample_indices.copy()
            total_checked_samples_per_class[noisy_label] = 0
            continue

        # 取 top-K 最大簇
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:K]
        top_cluster_ids = [cid for cid, _ in sorted_clusters]

        checked_samples_count = 0
        qualified_clusters = []
        total_qualified_samples = 0

        for cluster_id in top_cluster_ids:
            cluster_mask = (cluster_labels == cluster_id)
            cluster_in_class_indices = np.where(cluster_mask)[0]
            cluster_global_indices = [sample_indices[i] for i in cluster_in_class_indices]

            sample_count = min(N, len(cluster_global_indices))
            if sample_count == 0:
                continue

            correct_ratio = is_representative_by_mllm(
                cluster_global_indices,
                noisy_label,
                clean_labels_np,
                n_sample=N,
                threshold=THRESHOLD
            )

            checked_samples_count += sample_count

            if correct_ratio >= THRESHOLD:
                qualified_global_indices_all.extend(cluster_global_indices)
                qualified_clusters.append(cluster_global_indices)
                total_qualified_samples += len(cluster_global_indices)

        # 最终处理代表样本
        if qualified_global_indices_all:
            # 有合格簇 → 限制最大数量
            if len(qualified_global_indices_all) > MIN_REPRESENTATIVE_SAMPLES:
                qualified_global_indices_all = np.random.choice(
                    qualified_global_indices_all,
                    size=MIN_REPRESENTATIVE_SAMPLES,
                    replace=False
                ).tolist()
        else:
            # 没有任何合格簇 → 使用全类（最保守 fallback）
            print(f"类 {noisy_label} 没有任何合格代表簇，使用全类样本作为代表...")
            qualified_global_indices_all = sample_indices.copy()

        representative_indices_per_class[noisy_label] = qualified_global_indices_all

        # 记录统计信息
        cluster_analysis_results[noisy_label] = {
            "total_samples_in_class": len(sample_indices),
            "found_representative": len(qualified_global_indices_all) > 0,
            "total_checked_samples": checked_samples_count,
            "total_qualified_samples": len(qualified_global_indices_all),
            "num_qualified_clusters": len(qualified_clusters)
        }
        total_checked_samples_per_class[noisy_label] = checked_samples_count

    # 打印总结（可选，根据需要保留或注释）
    print("\n===== HDBSCAN 聚类分析完成 =====")
    print(f"总检查样本数: {sum(total_checked_samples_per_class.values())}")
    for lbl in sorted(total_checked_samples_per_class):
        print(f"  类 {lbl}: 检查 {total_checked_samples_per_class[lbl]} 个样本")

    return cluster_analysis_results, total_checked_samples_per_class, representative_indices_per_class


# ─── 标签重分配（只使用预计算的代表样本，绝不重复调用 MLLM） ──────────────────
def reassign_labels_by_centroid(
    features,
    noise_labels,
    clean_labels,
    representative_indices_per_class,  # 必须传入
    top_ratio=0.9,
):
    """
    使用预先计算好的代表性样本构建类质心，进行标签重分配。
    此函数**不会**再进行任何聚类或 MLLM 调用。
    """
    print("===== 开始基于质心的标签重分配（使用预计算代表样本） =====")

    if representative_indices_per_class is None or len(representative_indices_per_class) == 0:
        raise ValueError("必须传入 representative_indices_per_class 参数！")

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

    for noisy_label in tqdm(label_to_indices.keys(), desc="构建类质心"):

        if noisy_label not in representative_indices_per_class:
            raise ValueError(f"类别 {noisy_label} 在预计算结果中缺失！")

        qualified_global_indices = representative_indices_per_class[noisy_label]

        if len(qualified_global_indices) == 0:
            raise RuntimeError(f"类别 {noisy_label} 的代表样本为空，这不应该发生")

        print(f"类 {noisy_label} 使用预计算代表样本：{len(qualified_global_indices)} 张")
        centroid = np.mean(features_np[qualified_global_indices], axis=0)
        class_centroids[noisy_label] = centroid
        class_representative_count[noisy_label] = len(qualified_global_indices)

    if not class_centroids:
        raise ValueError("没有任何类的代表样本，无法构建质心")

    # ── 后续距离计算、标签修正、统计部分保持不变 ─────────────────────────────
    all_classes = list(class_centroids.keys())
    print(f"\n找到 {len(all_classes)} 个类的代表质心，开始计算距离...")

    # GPU 批量计算 correlation distance
    features_t = torch.from_numpy(features_np).float().to(device)
    centroid_values = [torch.from_numpy(class_centroids[cls]).float() for cls in all_classes]
    centroids_t = torch.stack(centroid_values).to(device)

    class_map = {i: cls for i, cls in enumerate(all_classes)}

    # 中心化 & L2 归一化
    features_mean = features_t.mean(dim=1, keepdim=True)
    centroids_mean = centroids_t.mean(dim=1, keepdim=True)
    features_centered = features_t - features_mean
    centroids_centered = centroids_t - centroids_mean

    features_norm = torch.nn.functional.normalize(features_centered, p=2, dim=1)
    centroids_norm = torch.nn.functional.normalize(centroids_centered, p=2, dim=1)

    cosine_sim = torch.mm(features_norm, centroids_norm.T)
    distances = 1 - cosine_sim  # correlation distance

    min_dist, best_class_idx = distances.min(dim=1)

    min_dist_cpu = min_dist.cpu().numpy()
    best_class_idx_cpu = best_class_idx.cpu().numpy()

    sample_distances = [
        (i, class_map[best_class_idx_cpu[i]], min_dist_cpu[i])
        for i in range(total_samples)
    ]

    # 高置信度样本修正标签
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

    # ── 统计结果（保持原逻辑） ────────────────────────────────────────────────
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

    print(f"\n=== 标签重分配结果（top {int(top_ratio*100)}% 高置信度修正）===")
    print(f"整体准确率: {overall_accuracy:.4f}")
    print(f"总样本数: {total_samples}\n")

    print("每个类的详细统计:")
    for class_label in sorted(class_accuracies.keys()):
        acc = class_accuracies[class_label]['accuracy']
        correct = class_accuracies[class_label]['correct']
        total = class_accuracies[class_label]['total']
        rep_count = class_accuracies[class_label]['representative_sample_count']
        corrected_count = class_accuracies[class_label]['corrected_sample_count']
        rep_str = str(rep_count) if rep_count > 0 else "无"
        print(f"类 {class_label}: 准确率 {acc:.4f} ({correct}/{total})，代表样本 {rep_str}，修正后样本 {corrected_count}")

    results = {
        'overall_accuracy': overall_accuracy,
        'total_samples': total_samples,
        'class_accuracies': dict(class_accuracies),
        'class_centroids': {k: v.tolist() for k, v in class_centroids.items()},
        'class_representative_counts': class_representative_count,
        'corrected_class_sample_counts': corrected_class_sample_count
    }

    print("===== 标签重分配完成 =====\n")
    return results, new_labels_np, noise_labels_np

# 执行功能1
analyze_noise_class_accuracy(clean_labels=clean_label, noise_labels=noise_label, num_classes=num_class)



# 1. 先跑一次详细的聚类分析（会做 MLLM 判断）
cluster_results, checked_count, rep_indices_dict = hdbscan_cluster_analysis(
    features=features_tensor,
    noise_labels=noise_label,
    clean_labels=clean_label,
    num_classes=num_class,
    K=10,
    N=5,
    THRESHOLD=0.4,
    MIN_REPRESENTATIVE_SAMPLES=1000
)

# 2. 直接把代表性样本索引传给标签修正函数，大幅节省时间
# 后续只用预计算结果，绝不再调用 MLLM
reassignment_results, new_labels_np, _ = reassign_labels_by_centroid(
    features=features_tensor,
    noise_labels=noise_label,
    clean_labels=clean_label,
    representative_indices_per_class=rep_indices_dict,
    top_ratio=0.9
)

