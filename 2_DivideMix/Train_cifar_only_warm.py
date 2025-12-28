from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np

# 模型和数据加载器
from PreResNet import ResNet18
import dataloader_cifar as dataloader

import json
import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR Warmup Only')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--r', default=0, type=float, help='noise ratio')
parser.add_argument('--id', default='dino_idn80')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--data_path', default='/mnt/zfj/dataset/cifar-100-python', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar100', type=str)
args = parser.parse_args()

# 设置设备和随机种子
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True

# 数据加载器
# import sys
# sys.path.append('/mnt/zfj/projects/phd/projects_phd/CWU')
# from dataloader import dataloader_cifar as dataloader
loader = dataloader.cifar_dataloader(
    args.dataset,
    r=args.r,
    noise_mode=args.noise_mode,
    batch_size=args.batch_size,
    num_workers=8,
    root_dir=args.data_path,
    log=None,
    noise_file='/mnt/zfj/dataset/cifar-100-python/noise_file/cifar100_sym_0'
)

print('| Building net')
net1 = ResNet18(num_classes=args.num_class).cuda()
net2 = ResNet18(num_classes=args.num_class).cuda()  # 仅创建，不训练

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# 损失函数
CEloss = nn.CrossEntropyLoss()

def conf_penalty(logits):
    probs = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))

# 测试函数
def test(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    return acc

# 训练函数（预热）
def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = len(dataloader)
    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        penalty = conf_penalty(outputs)
        L = loss + penalty
        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Warmup Epoch [%3d/500] Iter[%3d/%3d] CE: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, batch_idx + 1, num_iter, loss.item()))
        sys.stdout.flush()

# 准备测试数据加载器
test_loader = loader.run('test')
warmup_trainloader = loader.run('warmup')

# 保存路径
save_dir = '/mnt/zfj/projects/phd/projects_phd/DivideMix-master/checkpoint/warmup_results'
os.makedirs(save_dir, exist_ok=True)
json_path = f'{save_dir}/warmup_{args.dataset}_{args.noise_mode}_{args.r}_{args.id}.json'

# 用于保存每轮结果
results = {
    "args": vars(args),
    "epoch_accuracies": []
}

print(f"Starting warmup for 500 epochs (only training net1), testing every epoch...")
print(f"Results will be saved to: {json_path}")

for epoch in range(1, 501):
    # 学习率调整
    lr = args.lr
    if epoch >= 150:
        lr *= 0.1
    if epoch >= 300:
        lr *= 0.1
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr

    # 训练 net1
    warmup(epoch, net1, optimizer1, warmup_trainloader)

    # 每轮测试
    test_acc = test(net1, test_loader)
    results["epoch_accuracies"].append(round(test_acc, 4))  # 保留4位小数

    # 实时输出
    print(f" | Test Acc: {test_acc:.2f}%")

    # 可选：每10轮保存一次，防止中断
    if epoch % 10 == 0 or epoch == 500:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

print("Warmup completed. Results saved to JSON.")