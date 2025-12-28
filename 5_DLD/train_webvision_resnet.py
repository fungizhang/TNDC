import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import argparse
import random

# 导入你提供的数据集类
# from your_dataset_file import WebVision, WebVision_WS  # 替换为实际存放数据集类的文件名
from utils.webvision_data_utils import *

# 设置随机种子
def set_seed(seed=111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 模型定义（ResNet50 + 分类头）
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # 加载预训练ResNet50
        self.backbone = models.resnet50(pretrained=True)
        # 替换分类头以适应50类任务
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch_data in pbar:
            # 解析批次数据（WebVision_WS返回的是(weak, strong, label, index)）
            _, _, labels, _ = batch_data  # 仅使用标签，忽略增强图像
            images = batch_data[0]  # 使用弱增强图像作为训练输入
            
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计指标
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100 * correct / total
            })
    
    train_acc = 100 * correct / total
    train_loss = total_loss / len(train_loader)
    return train_loss, train_acc

# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(), tqdm(test_loader, desc='Testing') as pbar:
        for batch_data in pbar:
            # 解析批次数据（WebVision返回的是(image, label, index)）
            images, labels, _ = batch_data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 统计指标
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100 * correct / total
            })
    
    test_acc = 100 * correct / total
    test_loss = total_loss / len(test_loader)
    return test_loss, test_acc

def main(args):
    set_seed()
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据加载（使用你提供的数据集类）
    train_dataset = WebVision_WS(
        data_root=args.data_dir,
        split='train',
        balance=True,  # 平衡采样
        cls_size=500,  # 每类采样数量
        randomize=True,
        transform='train'  # 使用训练变换
    )
    test_dataset = WebVision(
        data_root=args.data_dir,
        split='val',
        balance=False,
        transform='val'  # 使用验证变换
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 模型初始化
    model = ResNet50Classifier(num_classes=1000).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 学习率衰减

    # 训练循环
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        # 学习率调度
        scheduler.step()
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model (acc: {best_acc:.2f}%)")
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/mnt/home/zfj", help="WebVision数据目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载线程数")
    parser.add_argument("--save_path", default="./best_resnet50_webvision.pt", help="最佳模型保存路径")
    args = parser.parse_args()
    
    main(args)