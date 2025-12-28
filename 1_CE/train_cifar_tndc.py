import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 项目特定路径
import sys
# sys.path.append('/mnt/zfj/projects/SODA-main')
# sys.path.append('/mnt/zfj/projects/phd/projects_phd/CWU')

import models
from models.split_model import get_num_trainable_parameters

# ==================== 参数解析 ====================
parser = argparse.ArgumentParser(description='Basic Training on CIFAR-10/100 with Custom Labels')
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--data_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--root_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='/mnt/zfj/exp_results/cifar10/resnet18_basic')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--label_path', type=str, default=None,
                    help='Path to custom label JSON file')
parser.add_argument('--noise_mode', type=str, default=None, choices=['sym', 'asym', 'idn'],
                    help='Noise mode for label corruption')
parser.add_argument('--noise_ratio', type=float, default=None, choices=[0.2, 0.4, 0.6, 0.8],
                    help='Noise ratio for label corruption')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', action='store_true', default=True)
parser.add_argument('--gradient_clip', type=float, default=0.0)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--gpu', type=int, default=5, help='GPU device id to use (e.g., 0, 1, ..., 7)')

args = parser.parse_args()

# ==================== 自动 label_path 和参数校验 ====================
if args.label_path is None:
    if args.noise_mode is not None and args.noise_ratio is not None:
        args.label_path = f"./datasets/dino_mod/{args.data_name}/dino_mod_labels_{args.noise_mode}_{args.noise_ratio}.json"
    elif args.noise_mode is not None or args.noise_ratio is not None:
        raise ValueError("Both --noise_mode and --noise_ratio must be specified together.")
else:
    if args.noise_mode is not None or args.noise_ratio is not None:
        print("Warning: --label_path is explicitly provided; --noise_mode and --noise_ratio are ignored.")

# ==================== 自动路径设置 ====================
if args.data_name == 'cifar10':
    default_root = './datasets/'
    default_num_class = 10
elif args.data_name == 'cifar100':
    default_root = './datasets/'
    default_num_class = 100
else:
    raise ValueError("Unsupported data_name")

args.root_dir = default_root
args.num_class = default_num_class

# 替换 dataset name in output_dir
args.output_dir = args.output_dir.replace('cifar10', args.data_name).replace('cifar100', args.data_name)

# Append noise tag if applicable
if args.noise_mode is not None and args.noise_ratio is not None:
    noise_tag = f"_{args.noise_mode}_{int(args.noise_ratio * 100)}"
    if not args.output_dir.endswith(noise_tag):
        args.output_dir += noise_tag

os.makedirs(args.output_dir, exist_ok=True)

device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==================== 数据增强 ====================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# ==================== 测试函数 ====================
def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ==================== 主函数 ====================
def main():
    test_accuracies = []  # 新增：用于记录每个 epoch 的 test acc
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'train.log')),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    # Log noise info
    if args.noise_mode is not None and args.noise_ratio is not None:
        logger.info(f"Using noisy labels: mode={args.noise_mode}, ratio={args.noise_ratio}")
    elif args.label_path is not None:
        logger.info(f"Using custom labels from: {args.label_path}")
    else:
        logger.info("Using original clean labels")

    # ------------------ 使用标准 CIFAR 数据集 ------------------
    if args.data_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.root_dir, train=True, download=False, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=args.root_dir, train=False, download=False, transform=transform_test)
    else:
        train_dataset = datasets.CIFAR100(root=args.root_dir, train=True, download=False, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=args.root_dir, train=False, download=False, transform=transform_test)

    # ------------------ 自定义标签替换（关键：在 DataLoader 之前！）------------------
    if args.label_path is not None:
        if not os.path.exists(args.label_path):
            raise FileNotFoundError(f"Label file not found: {args.label_path}")
        
        with open(args.label_path, 'r') as f:
            custom_labels = json.load(f)
        
        custom_labels = np.array(custom_labels, dtype=np.int64)
        if len(custom_labels) != len(train_dataset):
            raise ValueError(f"Custom labels length {len(custom_labels)} != train set {len(train_dataset)}")
        
        train_dataset.targets = custom_labels.tolist()  # 直接替换 .targets
        logger.info(f"Loaded custom labels from {args.label_path}")

    # ------------------ 创建 DataLoaders ------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # ------------------ 模型 ------------------
    net = getattr(models, args.arch)(num_classes=args.num_class).to(device)
    if args.resume and args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print(f"Trainable params: {get_num_trainable_parameters(net)}")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    logger.info('Epoch \t LR \t Time \t TrainLoss \t TrainACC \t TestACC')

    for epoch in range(args.epoch):
        start_time = time.time()
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clip)

            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        test_acc = test(net, test_loader)
        test_accuracies.append(test_acc)  # ← 新增这一行
        current_lr = optimizer.param_groups[0]['lr']

        logger.info('{:03d} \t {:.4f} \t {:.1f}s \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            epoch, current_lr, time.time() - start_time, train_loss, train_acc, test_acc))
    
    # ==================== 保存测试结果到 JSON ====================
    # 构造文件名
    if args.noise_mode is not None and args.noise_ratio is not None:
        filename = f"{args.data_name}_{args.noise_mode}_{args.noise_ratio}_tndc.json"
    else:
        filename = f"{args.data_name}_clean.json"

    ce_dir = "./exp_results/CE"
    os.makedirs(ce_dir, exist_ok=True)
    json_path = os.path.join(ce_dir, filename)

    # 转换为 { "0": 95.12, "1": 96.34, ... } 格式（保留两位小数，乘以100转为百分比）
    result_dict = {str(epoch): round(acc * 100, 2) for epoch, acc in enumerate(test_accuracies)}

    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=4)

    logger.info(f"Test accuracies saved to {json_path}")


if __name__ == '__main__':
    main()