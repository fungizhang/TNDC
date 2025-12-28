import os
import math
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import json
import argparse

# ==================== Parse Command-Line Arguments ====================
parser = argparse.ArgumentParser(description="Train DMLP with configurable parameters")
parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], required=True, help='Dataset name')
parser.add_argument('--device_id', type=int, default=1, help='GPU device ID (e.g., 1 for cuda:1)')
parser.add_argument('--noise_class', type=str, required=True, help='Noise type, e.g., "idn", "sym"')
parser.add_argument('--noise_ratio', type=float, required=True, help='Noise ratio, e.g., 0.2')

args = parser.parse_args()

# Assign parsed arguments
epoch = args.epoch
dataset = args.dataset
device = torch.device(f"cuda:{args.device_id}")
noise_class = args.noise_class
noise_ratio = args.noise_ratio

print(f"[INFO] Using device: {device}")
print(f"[INFO] Dataset: {dataset}, Noise: {noise_class} @ {noise_ratio}, Epochs: {epoch}")

if dataset == 'cifar10':
    num_classes = 10
    train_num = 49000
    val_num = 1000
    test_num = 10000
elif dataset == 'cifar100':
    num_classes = 100
    train_num = 49000
    val_num = 1000
    test_num = 10000
else:
    raise ValueError("Unsupported dataset")

batchsize = 7000
iteration = train_num // batchsize

batchsize_fc = 500
iteration_fc = batchsize // batchsize_fc

weight_conf = 0.4
weight_prior = 2200
epoch_T = 20

# ==================== Paths and Data Loading ====================
feat_train_path = f'/mnt/zfj/projects/phd/projects_phd/CWU/saved/saved_features/{dataset}_features_no_aug.pth'
feat_test_path = f'/mnt/zfj/projects/phd/projects_phd/CWU/saved/saved_features/{dataset}_features_no_aug_test.pth'

features_train = torch.load(feat_train_path)
features_test = torch.load(feat_test_path)

feat_tensor_train = features_train['features']
feat_tensor_test = features_test['features']
feat_tensor_test_label = features_test['labels']

if feat_tensor_train.is_cuda:
    feat_tensor_train = feat_tensor_train.cpu()
if feat_tensor_test.is_cuda:
    feat_tensor_test = feat_tensor_test.cpu()

features_train_v1 = feat_tensor_train.detach().numpy().astype(np.float32)
features_test_v1 = feat_tensor_test.detach().numpy().astype(np.float32)
print("Features shape:", features_train_v1.shape, "dtype:", features_train_v1.dtype)
print("Features shape:", features_test_v1.shape, "dtype:", features_test_v1.dtype)

features_v1 = features_train_v1
features_v2 = features_train_v1
test_features = features_test_v1

# Load clean labels
if dataset == 'cifar10':
    clean_label_path = f'/mnt/zfj/dataset/cifar-10-batches-py/noise_file/{dataset}_sym_0'
elif dataset == 'cifar100':
    clean_label_path = f'/mnt/zfj/dataset/cifar-100-python/noise_file/{dataset}_sym_0'
with open(clean_label_path, "r") as f:
    clean_labels = np.array(json.load(f))
labels_correct = torch.from_numpy(clean_labels).to(device)

test_labels = feat_tensor_test_label

# Shuffle to split train/val
idx_shuffle = torch.randperm(train_num + val_num)
idx_to_train = idx_shuffle[:train_num]
idx_to_meta = idx_shuffle[train_num:]

# Validation class distribution
data_list_val = {}
for j in range(num_classes):
    data_list_val[j] = [i for i, label in enumerate(labels_correct[idx_to_meta].cpu()) if label == j]
    print(f"Val ratio class {j}: {len(data_list_val[j]) / val_num * 100:.2f}%")

# Train / Val / Test features
train_features_v1 = torch.from_numpy(features_v1[idx_to_train])  # CPU
train_features_v2 = torch.from_numpy(features_v2[idx_to_train])  # CPU
test_features_v1 = torch.from_numpy(test_features)
test_features_v2 = test_features_v1.clone()

val_features_v1 = torch.from_numpy(features_v1[idx_to_meta]).to(device)
val_features_v2 = torch.from_numpy(features_v2[idx_to_meta]).to(device)

# Noisy labels
if dataset == 'cifar10':
    noise_path = f'/mnt/zfj/dataset/dino_mod/cifar10/dino_mod_labels_{noise_class}_{noise_ratio}.json'
elif dataset == 'cifar100':
    noise_path = f'/mnt/zfj/dataset/dino_mod/cifar100/dino_mod_labels_{noise_class}_{noise_ratio}.json'
with open(noise_path, "r") as f:
    noise_data = json.load(f)
train_all = torch.from_numpy(np.array(noise_data)).long()

correct_all = labels_correct.cpu()
noise_or_not = [(train_all[i] == correct_all[i]).item() for i in range(50000)]
print("Clean label ratio:", sum(noise_or_not) / 50000)

# ==================== Label Preparation ====================
noisy_train_labels = train_all[idx_to_train].to(device)
train_labels_v1_onehot = F.one_hot(noisy_train_labels, num_classes).float()
train_labels_v1 = Variable(train_labels_v1_onehot, requires_grad=True)
optimizer_v1 = torch.optim.Adam([train_labels_v1], lr=0.01, betas=(0.5, 0.999), weight_decay=3e-4)

val_labels_v1 = labels_correct[idx_to_meta].long().to(device)
val_labels_v1_onehot = F.one_hot(val_labels_v1, num_classes).float().to(device)

train_labels_v2 = Variable(train_labels_v1_onehot.clone(), requires_grad=True)
optimizer_v2 = torch.optim.Adam([train_labels_v2], lr=0.01, betas=(0.5, 0.999), weight_decay=3e-4)

prior = torch.ones(num_classes) / num_classes
prior = prior.to(device)

# ==================== Enhanced EAC Classifier (MLP-based) ====================
class EACClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden_dims=[512, 256], dropout=0.3):
        super(EACClassifier, self).__init__()
        layers = []
        input_dim = feat_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, num_classes))  # output logits
        self.mlp = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, apply_softmax=False):
        logits = self.mlp(x)
        if apply_softmax:
            return self.softmax(logits)
        else:
            return logits

feat_dim = train_features_v1.shape[1]

# Instantiate two EAC classifiers
model_fcv1 = EACClassifier(feat_dim, num_classes, hidden_dims=[512, 256], dropout=0.3).to(device)
model_fcv2 = EACClassifier(feat_dim, num_classes, hidden_dims=[512, 256], dropout=0.3).to(device)

# Use slightly lower LR for deeper model
optimizer_fcv1 = torch.optim.Adam(model_fcv1.parameters(), lr=0.01, betas=(0.5, 0.999), weight_decay=3e-4)
optimizer_fcv2 = torch.optim.Adam(model_fcv2.parameters(), lr=0.01, betas=(0.5, 0.999), weight_decay=3e-4)

lr_schedulerv1 = lr_scheduler.CosineAnnealingLR(optimizer_fcv1, T_max=20, eta_min=0.0002)
lr_schedulerv2 = lr_scheduler.CosineAnnealingLR(optimizer_fcv2, T_max=20, eta_min=0.0002)

loss_mse = nn.MSELoss(reduction='sum')
loss_ce = nn.CrossEntropyLoss()
maxacc = 0.0

# ==================== Learning Rate Adjust (Original Logic) ====================
def adjust_learning_rate(optimizer, epoch):
    lr = 0.006
    steps = np.sum(epoch > np.array([3]))
    if steps > 0:
        lr = lr * (0.1 ** steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# ==================== Training Loop ====================
ensemble_accuracies = {}
for i in range(epoch):
    model_fcv1.train()
    model_fcv2.train()

    adjust_learning_rate(optimizer_fcv1, i)
    adjust_learning_rate(optimizer_fcv2, i)

    for j in range(iteration):
        start_idx = batchsize * j
        end_idx = min(batchsize * (j + 1), train_num)

        batchfeat_v1 = train_features_v1[start_idx:end_idx].to(device)
        batchfeat_v2 = train_features_v2[start_idx:end_idx].to(device)
        correct_batch = labels_correct[idx_to_train[start_idx:end_idx]].to(device)

        batchlabel_v1_detach = train_labels_v1[start_idx:end_idx].detach()
        batchlabel_v2_detach = train_labels_v2[start_idx:end_idx].detach()

        acc_ori_v1 = (torch.argmax(batchlabel_v1_detach, dim=1) == correct_batch).float().mean().item() * 100
        acc_ori_v2 = (torch.argmax(batchlabel_v2_detach, dim=1) == correct_batch).float().mean().item() * 100

        # ==================== EAC Update (Neural Network Only) ====================
        for p in range(iteration_fc):
            fc_start = batchsize_fc * p
            fc_end = min(batchsize_fc * (p + 1), batchsize)
            if fc_start >= batchsize:
                break

            subfeat_v1 = batchfeat_v1[fc_start:fc_end]
            subfeat_v2 = batchfeat_v2[fc_start:fc_end]
            sublabel_v1 = batchlabel_v1_detach[fc_start:fc_end]
            sublabel_v2 = batchlabel_v2_detach[fc_start:fc_end]

            # View 1
            optimizer_fcv1.zero_grad()
            logits_fcv1 = model_fcv1(subfeat_v1, apply_softmax=False)
            hard_labels_v1 = torch.argmax(sublabel_v1, dim=1)
            loss_newv1 = loss_ce(logits_fcv1, hard_labels_v1)

            soft_pred_v1 = F.softmax(logits_fcv1, dim=1)
            conf_penalty_v1 = torch.mean(torch.sum(soft_pred_v1 * torch.log(soft_pred_v1 + 1e-8), dim=1))
            prior_penalty_v1 = torch.sum(prior * torch.log(prior / (soft_pred_v1.mean(0) + 1e-8)))

            loss_newv1 += weight_prior * prior_penalty_v1 + weight_conf * conf_penalty_v1
            loss_newv1.backward()
            optimizer_fcv1.step()

            # View 2
            optimizer_fcv2.zero_grad()
            logits_fcv2 = model_fcv2(subfeat_v2, apply_softmax=False)
            hard_labels_v2 = torch.argmax(sublabel_v2, dim=1)
            loss_newv2 = loss_ce(logits_fcv2, hard_labels_v2)

            soft_pred_v2 = F.softmax(logits_fcv2, dim=1)
            conf_penalty_v2 = torch.mean(torch.sum(soft_pred_v2 * torch.log(soft_pred_v2 + 1e-8), dim=1))
            prior_penalty_v2 = torch.sum(prior * torch.log(prior / (soft_pred_v2.mean(0) + 1e-8)))

            loss_newv2 += weight_prior * prior_penalty_v2 + weight_conf * conf_penalty_v2
            loss_newv2.backward()
            optimizer_fcv2.step()

        # EAC accuracy on current batch
        with torch.no_grad():
            pred_fcv1 = model_fcv1(batchfeat_v1, apply_softmax=True)
            pred_fcv2 = model_fcv2(batchfeat_v2, apply_softmax=True)
            acc_fcv1 = (torch.argmax(pred_fcv1, dim=1) == correct_batch).float().mean().item() * 100
            acc_fcv2 = (torch.argmax(pred_fcv2, dim=1) == correct_batch).float().mean().item() * 100

        # ==================== Label Swap (Only at last epoch and every epoch_T) ====================
        if i % epoch_T == 0 and i != 0 and i == epoch - 1:
            with torch.no_grad():
                # Use softmax predictions from the other view to update soft labels
                new_soft_v2 = model_fcv2(batchfeat_v1, apply_softmax=True)
                new_soft_v1 = model_fcv1(batchfeat_v2, apply_softmax=True)
                train_labels_v1.data[start_idx:end_idx] = new_soft_v2
                train_labels_v2.data[start_idx:end_idx] = new_soft_v1

    # ==================== Test Evaluation ====================
    with torch.no_grad():
        test_feat_gpu = test_features_v1.to(device)
        test_labels_gpu = test_labels.to(device)

        model_fcv1.eval()
        model_fcv2.eval()

        pred1 = model_fcv1(test_feat_gpu, apply_softmax=True)
        pred2 = model_fcv2(test_feat_gpu, apply_softmax=True)

        acc_test_fcv1 = (torch.argmax(pred1, dim=1) == test_labels_gpu).float().mean().item() * 100
        acc_test_fcv2 = (torch.argmax(pred2, dim=1) == test_labels_gpu).float().mean().item() * 100

        ensembled = (pred1 + pred2) / 2.0
        ensem_acc = (torch.argmax(ensembled, dim=1) == test_labels_gpu).float().mean().item() * 100

        maxacc = max(maxacc, ensem_acc)
        ensemble_accuracies[str(i)] = round(ensem_acc, 2)

        print("=" * 60)
        print(f">>> Epoch {i+1} Test Results <<<")
        print(f"Max Acc: {maxacc:.2f}% | Current Ensemble Acc: {ensem_acc:.2f}%")
        print(f"EAC MLP v1: {acc_test_fcv1:.2f}% | v2: {acc_test_fcv2:.2f}%")
        print("=" * 60)

print("Training completed!")

# ==================== Save Results ====================
save_dir = "/mnt/zfj/exp_results/DMLP"
os.makedirs(save_dir, exist_ok=True)
filename = f"{dataset}_{noise_class}_{noise_ratio}_tndc_mlp.json"
save_path = os.path.join(save_dir, filename)

with open(save_path, "w") as f:
    json.dump(ensemble_accuracies, f, indent=4)

print(f"Ensemble accuracies saved to {save_path}")