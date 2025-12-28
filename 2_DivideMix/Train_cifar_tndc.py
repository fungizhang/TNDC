from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import json

# ==================== 参数解析（新增 noise_ratio 和 label_path）====================
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym', choices=['sym', 'asym', 'idn'])
parser.add_argument('--noise_ratio', default=0.2, type=float, choices=[0.2, 0.4, 0.6, 0.8], help='noise ratio (e.g., 0.8)')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--id', default='idn_80_bs_256')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=5, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--data_path', default='/mnt/zfj/dataset', type=str, help='path to dataset root')
parser.add_argument('--label_path', type=str, default=None,
                    help='Path to custom label JSON file')
args = parser.parse_args()

# ==================== 自动 label_path 设置（同代码1）====================
if args.label_path is None:
    if args.noise_mode is not None and args.noise_ratio is not None:
        if args.dataset == 'cifar10':
            base_dir = '/mnt/zfj/dataset/dino_mod/cifar10'
        elif args.dataset == 'cifar100':
            base_dir = '/mnt/zfj/dataset/dino_mod/cifar100'
        else:
            raise ValueError("Unsupported dataset")
        args.label_path = os.path.join(
            base_dir,
            f"dino_mod_labels_{args.noise_mode}_{args.noise_ratio}.json"  # 注意：加 .json
        )
    elif args.noise_mode is not None or args.noise_ratio is not None:
        raise ValueError("Both --noise_mode and --noise_ratio must be specified together.")
else:
    if args.noise_mode is not None or args.noise_ratio is not None:
        print("Warning: --label_path is explicitly provided; --noise_mode and --noise_ratio are ignored.")

# ==================== 设备 & 随机种子 ====================
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ==================== 数据增强（两个视图）====================
if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
else:  # cifar100
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

transform_weak = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ==================== 自定义 Dataset（返回双增强 + index）====================
class CustomCIFARDataset(Dataset):
    def __init__(self, dataset, transform1, transform2=None):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2 or transform1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img1 = self.transform1(img)
        img2 = self.transform2(img)
        return img1, img2, label, idx

# ==================== 加载原始 CIFAR 数据集 ====================
if args.dataset == 'cifar10':
    train_dataset_raw = datasets.CIFAR10(root=args.data_path, train=True, download=False)
    test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=False, transform=transform_test)
    args.num_class = 10
    warm_up = 10
else:
    train_dataset_raw = datasets.CIFAR100(root=args.data_path, train=True, download=False)
    test_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=False, transform=transform_test)
    args.num_class = 100
    warm_up = 30

# ==================== 替换标签（关键！）====================
if args.label_path is not None:
    if not os.path.exists(args.label_path):
        raise FileNotFoundError(f"Label file not found: {args.label_path}")
    with open(args.label_path, 'r') as f:
        custom_labels = json.load(f)
    custom_labels = np.array(custom_labels, dtype=np.int64)
    if len(custom_labels) != len(train_dataset_raw):
        raise ValueError(f"Label length mismatch: {len(custom_labels)} vs {len(train_dataset_raw)}")
    train_dataset_raw.targets = custom_labels.tolist()
    print(f"Loaded noisy labels from {args.label_path}")

# ==================== 构造不同用途的 DataLoader ====================
def create_warmup_loader():
    dataset = CustomCIFARDataset(train_dataset_raw, transform_weak)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)

def create_eval_loader():
    # eval 只需单增强 + index
    class EvalDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self): return len(self.dataset)
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            img = transform_test(img)
            return img, label, idx
    dataset = EvalDataset(train_dataset_raw)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=5, pin_memory=True)

def create_train_loader(pred, prob):
    # 根据 pred mask 过滤样本
    indices = np.where(pred)[0]
    selected_targets = np.array(train_dataset_raw.targets)[indices]
    selected_prob = prob[indices]

    class SelectedDataset(Dataset):
        def __init__(self, raw_dataset, indices, probs, transform1, transform2):
            self.raw = raw_dataset
            self.indices = indices
            self.probs = probs
            self.t1 = transform1
            self.t2 = transform2
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            idx = self.indices[i]
            img, _ = self.raw[idx]
            label = self.raw.targets[idx]
            w = self.probs[i]
            return self.t1(img), self.t2(img), label, w

    labeled_set = SelectedDataset(train_dataset_raw, indices, selected_prob, transform_weak, transform_weak)
    unlabeled_set = CustomCIFARDataset(train_dataset_raw, transform_weak, transform_strong)

    labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
    return labeled_loader, unlabeled_loader

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5, pin_memory=True)

# ==================== 原有训练逻辑（保持不变）====================
# ... [Warmup, train, test, eval_train, SemiLoss, NegEntropy, create_model 等函数完全保留] ...
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, _, _ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, _, _ = next(unlabeled_train_iter)                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.noise_ratio, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, _, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode in ['asym', 'idn']:  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.noise_ratio, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)            
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))  
    return acc  # ←←← 返回准确率

def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.noise_ratio==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

# ==================== 模型 & 优化器 ====================
print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode in ['asym', 'idn']:
    conf_penalty = NegEntropy()

all_loss = [[], []]

# ==================== 训练循环 ====================
acc_history = {}
for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    eval_loader = create_eval_loader()

    if epoch < warm_up:
        warmup_loader = create_warmup_loader()
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_loader)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_loader)
    else:
        prob1, all_loss[0] = eval_train(net1, all_loss[0])
        prob2, all_loss[1] = eval_train(net2, all_loss[1])

        pred1 = (prob1 > args.p_threshold)
        pred2 = (prob2 > args.p_threshold)

        print('Train Net1')
        labeled_loader, unlabeled_loader = create_train_loader(pred2, prob2)
        train(epoch, net1, net2, optimizer1, labeled_loader, unlabeled_loader)

        print('\nTrain Net2')
        labeled_loader, unlabeled_loader = create_train_loader(pred1, prob1)
        train(epoch, net2, net1, optimizer2, labeled_loader, unlabeled_loader)

    acc = test(epoch, net1, net2)
    acc_history[epoch] = round(acc, 2)  # 保留两位小数，与示例一致

    # 保存准确率历史到 JSON
    os.makedirs('/mnt/zfj/exp_results/dividemix', exist_ok=True)
    result_path = f"/mnt/zfj/exp_results/dividemix/{args.dataset}_{args.noise_mode}_{args.noise_ratio}_tndc.json"
    with open(result_path, 'w') as f:
        json.dump(acc_history, f, indent=2)
    print(f"Accuracy history saved to {result_path}")




