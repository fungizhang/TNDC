import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

import models
import matplotlib
matplotlib.use('Agg')

from dataloader import dataloader_cifar as dataloader


# ------------------ 参数解析 ------------------
parser = argparse.ArgumentParser(description='Train and save sample losses')

# 基础参数（只保留训练必须的参数）
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--data_set', type=str, default='./datasets/cifar-10-batches-py')
parser.add_argument('--output_dir', type=str, default='./exp_results/loss_analysis1')
parser.add_argument('--noise_ratio', type=float, default=0.6, help='label noise ratio')
parser.add_argument('--noise_mode', type=str, default='idn', choices=['sym', 'asym', 'idn'])
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', action='store_true', default=True)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--cutmix_alpha', type=float, default=1.0)
parser.add_argument('--gradient_clip', type=float, default=5.0)
parser.add_argument('--warmup_epochs', type=int, default=5)

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 日志
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, 'train.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

# ------------------ 标签平滑 & CutMix ------------------
class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        n_classes = logits.size(1)
        one_hot = F.one_hot(labels, n_classes).float()
        soft_labels = (1 - self.epsilon) * one_hot + self.epsilon / n_classes
        log_probs = F.log_softmax(logits, dim=1)
        return -torch.sum(soft_labels * log_probs, dim=1)


def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0 or torch.rand(1).item() >= 0.5:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    y_a, y_b = y, y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


# ------------------ 主训练流程 ------------------
def main():
    # 数据加载
    loader = dataloader.cifar_dataloader(
        'cifar10',
        noise_mode=args.noise_mode,
        noise_ratio=args.noise_ratio,
        batch_size=args.batch_size,
        num_workers=4,
        root_dir=args.data_set,
        model='resnet'
    )
    train_loader = loader.run('train')

    # 噪声/干净索引
    noise_mask = train_loader.dataset.noise_idx
    noise_idx = torch.where(noise_mask)[0].cpu().numpy()
    clean_idx = torch.where(~noise_mask)[0].cpu().numpy()

    noise_label = torch.tensor(train_loader.dataset.noise_label)
    clean_label = torch.tensor(train_loader.dataset.clean_label)

    print(f"Total samples: {len(train_loader.dataset)}")
    print(f"Noisy samples : {len(noise_idx)}")
    print(f"Clean samples  : {len(clean_idx)}")

    # 模型 & 优化器
    net = models.resnet18(num_classes=10).to(device)
    criterion = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov
    )

    # 损失记录
    sample_losses = defaultdict(list)           # 每个样本在每个epoch的损失
    epoch_clean_noisy_loss = defaultdict(lambda: {'clean': [], 'noisy': []})

    logger.info("Start training...")

    for epoch in range(args.epoch):
        net.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels, _, sample_indices in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            sample_indices = torch.tensor(sample_indices).to(device)

            images, labels_a, labels_b, lam = cutmix_data(images, labels, args.cutmix_alpha)

            optimizer.zero_grad()
            outputs = net(images)

            loss_a = criterion(outputs, labels_a)
            loss_b = criterion(outputs, labels_b)
            loss_per_sample = lam * loss_a + (1 - lam) * loss_b
            loss = loss_per_sample.mean()

            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clip)
            optimizer.step()

            # 记录
            total_loss += loss_per_sample.sum().item()
            total_samples += images.size(0)

            pred = outputs.argmax(dim=1)
            correct = lam * pred.eq(labels_a).sum().item() + (1-lam) * pred.eq(labels_b).sum().item()
            total_correct += correct

            # 保存每个样本的损失
            for idx, s_idx in enumerate(sample_indices):
                s_idx = s_idx.item()
                loss_val = loss_per_sample[idx].item()
                sample_losses[s_idx].append(loss_val)

                if s_idx in noise_idx:
                    epoch_clean_noisy_loss[epoch]['noisy'].append(loss_val)
                else:
                    epoch_clean_noisy_loss[epoch]['clean'].append(loss_val)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        logger.info(f"Epoch [{epoch+1:3d}/{args.epoch}]  Loss: {avg_loss:.4f}  Acc: {avg_acc:.4f}")

    # ------------------ 保存所有必要数据 ------------------
    clean_avg_losses = [np.mean(sample_losses[i]) for i in clean_idx]
    noisy_avg_losses = [np.mean(sample_losses[i]) for i in noise_idx]

    save_path = os.path.join(args.output_dir, 'losses_data.npz')
    np.savez(
        save_path,
        clean_avg_losses=clean_avg_losses,
        noisy_avg_losses=noisy_avg_losses,
        epoch_clean_loss={k: v['clean'] for k, v in epoch_clean_noisy_loss.items()},
        epoch_noisy_loss={k: v['noisy'] for k, v in epoch_clean_noisy_loss.items()},
        sample_losses=dict(sample_losses),           # 完整记录（可选，文件会比较大）
        clean_idx=clean_idx,
        noise_idx=noise_idx,
        clean_label=clean_label.numpy(),
        noise_label=noise_label.numpy()
    )

    print(f"\n损失数据已保存至：{save_path}")
    print("可使用第二个脚本进行可视化分析")


if __name__ == '__main__':
    main()