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

# ==================== 指定使用 GPU 1 ====================
device = torch.device("cuda:1")  # 之后所有 .cuda() 都会自动使用 cuda:1

# ==================== Hyperparameters ====================
num_classes = 10
train_num = 49000      # Training set size
val_num = 1000        # Validation set size
test_num = 10000
batchsize = 7000       # Batch size for IPC (closed-form update)
iteration = train_num // batchsize

batchsize_fc = 500     # Batch size for EAC (neural network update)
iteration_fc = batchsize // batchsize_fc

epoch = 100
weight_conf = 0.4      # Confidence penalty weight
weight_prior = 2200    # Class prior balance weight
epoch_T = 20           # Period for EAC label swap (only at last epoch in original logic)

noise_class = 'sym'
noise_ratio = 0.4


# ==================== Paths and Data Loading ====================
# path = '/mnt/zfj/projects/DMLP-main/symmetric20/'

# Load pre-extracted features
features_train = torch.load('/mnt/zfj/projects/phd/projects_phd/CWU/saved/saved_features/cifar10_features_no_aug.pth')
features_test = torch.load('/mnt/zfj/projects/phd/projects_phd/CWU/saved/saved_features/cifar10_features_no_aug_test.pth')
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

# Use same features for both views
features_v1 = features_train_v1
features_v2 = features_train_v1
test_features = features_test_v1
# test_features = features_train_v1[0:10000]

# Labels
# labels_correct = torch.from_numpy(np.load(path + 'correctlabels.npy')).to(device)  # Clean labels
labels_correct = torch.from_numpy(np.array(json.load(open('/mnt/zfj/dataset/cifar-10-batches-py/noise_file/cifar10_sym_0')))).to(device)  # Clean labels
# test_labels = torch.from_numpy(np.load(path + 'testlabels.npy'))
test_labels = feat_tensor_test_label

# Shuffle to split train/val
idx_shuffle = torch.randperm(train_num + val_num)
idx_to_train = idx_shuffle[:train_num]
idx_to_meta = idx_shuffle[train_num:]

# Validation class distribution (for monitoring)
data_list_val = {}
for j in range(num_classes):
    data_list_val[j] = [i for i, label in enumerate(labels_correct[idx_to_meta].cpu()) if label == j]
    print(f"Val ratio class {j}: {len(data_list_val[j]) / val_num * 100:.2f}%")

# Train / Val / Test features (keep train on CPU to save GPU memory, move batch-wise)
train_features_v1 = torch.from_numpy(features_v1[idx_to_train])          # CPU
train_features_v2 = torch.from_numpy(features_v2[idx_to_train])          # CPU
test_features_v1 = torch.from_numpy(test_features)                       # CPU -> later to device
test_features_v2 = test_features_v1.clone()

val_features_v1 = torch.from_numpy(features_v1[idx_to_meta]).to(device)
val_features_v2 = torch.from_numpy(features_v2[idx_to_meta]).to(device)

# Noise ratio
# train_all = torch.from_numpy(np.load(path + 'trainlabels.npy')).long()
noise_path = f'/mnt/zfj/dataset/cifar-10-batches-py/noise_file/cifar10_{noise_class}_{noise_ratio}'
with open(noise_path, "r") as f:
    noise_data = json.load(f)
train_all = torch.from_numpy(np.array(noise_data)).long()

correct_all = labels_correct.cpu()
noise_or_not = [(train_all[i] == correct_all[i]).item() for i in range(50000)]
print("Clean label ratio:", sum(noise_or_not) / 50000)

# ==================== Label Preparation ====================
# View 1: noisy labels -> trainable soft labels
noisy_train_labels = train_all[idx_to_train].to(device)
train_labels_v1_onehot = F.one_hot(noisy_train_labels, num_classes).float()
train_labels_v1 = Variable(train_labels_v1_onehot, requires_grad=True)
optimizer_v1 = torch.optim.Adam([train_labels_v1], lr=0.01, betas=(0.5, 0.999), weight_decay=3e-4)

# Validation labels (clean)
val_labels_v1 = labels_correct[idx_to_meta].long().to(device)
val_labels_argv1 = val_labels_v1
val_labels_v1_onehot = F.one_hot(val_labels_v1, num_classes).float().to(device)

# View 2: start from same noisy labels
train_labels_v2 = Variable(train_labels_v1_onehot.clone(), requires_grad=True)
optimizer_v2 = torch.optim.Adam([train_labels_v2], lr=0.01, betas=(0.5, 0.999), weight_decay=3e-4)

vval_labels_maxv2 = val_labels_v1  # for CE loss

# Prior for balance loss
prior = torch.ones(num_classes) / num_classes
prior = prior.to(device)

# ==================== Models (EAC linear classifiers) ====================
feat_dim = train_features_v1.shape[1]

model_fcv1 = nn.Sequential(
    nn.Linear(feat_dim, num_classes),
    nn.Softmax(dim=1)
).to(device)

model_fcv2 = nn.Sequential(
    nn.Linear(feat_dim, num_classes),
    nn.Softmax(dim=1)
).to(device)

optimizer_fcv1 = torch.optim.Adam(model_fcv1.parameters(), lr=0.03, betas=(0.5, 0.999), weight_decay=3e-4)
optimizer_fcv2 = torch.optim.Adam(model_fcv2.parameters(), lr=0.03, betas=(0.5, 0.999), weight_decay=3e-4)

lr_schedulerv1 = lr_scheduler.CosineAnnealingLR(optimizer_fcv1, T_max=20, eta_min=0.0002)
lr_schedulerv2 = lr_scheduler.CosineAnnealingLR(optimizer_fcv2, T_max=20, eta_min=0.0002)

# Losses
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
for i in range(epoch):
    # Ensure models are on correct device and in train mode
    model_fcv1.to(device)
    model_fcv2.to(device)
    model_fcv1.train()
    model_fcv2.train()

    adjust_learning_rate(optimizer_fcv1, i)
    adjust_learning_rate(optimizer_fcv2, i)

    for j in range(iteration):
        start_idx = batchsize * j
        end_idx = min(batchsize * (j + 1), train_num)

        # Move batch to GPU
        batchfeat_v1 = train_features_v1[start_idx:end_idx].to(device)
        batchfeat_v2 = train_features_v2[start_idx:end_idx].to(device)
        correct_batch = labels_correct[idx_to_train[start_idx:end_idx]]

        batchlabel_v1_detach = train_labels_v1[start_idx:end_idx].detach()
        batchlabel_v2_detach = train_labels_v2[start_idx:end_idx].detach()

        # ==================== IPC Update (Closed-form) ====================
        # View 1
        optimizer_v1.zero_grad()
        XtX_inv = torch.inverse(batchfeat_v1.t() @ batchfeat_v1)
        XtX_inv_Xt = XtX_inv @ batchfeat_v1.t()
        W1_v1 = XtX_inv_Xt @ train_labels_v1[start_idx:end_idx]

        pred_val_v1 = val_features_v1 @ W1_v1
        loss_v1 = loss_ce(pred_val_v1, val_labels_argv1) + loss_mse(pred_val_v1, val_labels_v1_onehot)
        pred_mean_v1 = torch.softmax(batchfeat_v1 @ W1_v1, dim=1).mean(0)
        penalty_wv1 = torch.sum(prior * torch.log(prior / pred_mean_v1 + 1e-8))
        loss_v1 += penalty_wv1
        loss_v1.backward()
        optimizer_v1.step()

        # View 2
        optimizer_v2.zero_grad()
        XtX_inv_v2 = torch.inverse(batchfeat_v2.t() @ batchfeat_v2)
        XtX_inv_Xt_v2 = XtX_inv_v2 @ batchfeat_v2.t()
        W1_v2 = XtX_inv_Xt_v2 @ train_labels_v2[start_idx:end_idx]

        pred_val_v2 = val_features_v2 @ W1_v2
        loss_v2 = loss_ce(pred_val_v2, vval_labels_maxv2) + loss_mse(pred_val_v2, val_labels_v1_onehot)
        pred_mean_v2 = torch.softmax(batchfeat_v2 @ W1_v2, dim=1).mean(0)
        penalty_wv2 = torch.sum(prior * torch.log(prior / pred_mean_v2 + 1e-8))
        loss_v2 += penalty_wv2
        loss_v2.backward()
        optimizer_v2.step()

        # Label accuracy before EAC
        acc_ori_v1 = (torch.argmax(batchlabel_v1_detach, dim=1) == correct_batch).float().mean().item() * 100
        acc_ori_v2 = (torch.argmax(batchlabel_v2_detach, dim=1) == correct_batch).float().mean().item() * 100

        # ==================== EAC Update (Neural Network) ====================
        newlabel_v1 = train_labels_v1[start_idx:end_idx].detach()
        newlabel_v2 = train_labels_v2[start_idx:end_idx].detach()

        for p in range(iteration_fc):
            fc_start = batchsize_fc * p
            fc_end = min(batchsize_fc * (p + 1), batchsize)

            if fc_start >= batchsize:
                break

            subfeat_v1 = batchfeat_v1[fc_start:fc_end]
            subfeat_v2 = batchfeat_v2[fc_start:fc_end]
            sublabel_v1 = newlabel_v1[fc_start:fc_end]
            sublabel_v2 = newlabel_v2[fc_start:fc_end]

            # View 1
            optimizer_fcv1.zero_grad()
            pred_fcv1 = model_fcv1(subfeat_v1)
            hard_labels_v1 = torch.argmax(sublabel_v1, dim=1)
            loss_newv1 = loss_ce(pred_fcv1, hard_labels_v1)

            soft_pred_v1 = F.softmax(pred_fcv1, dim=1)
            conf_penalty_v1 = torch.mean(torch.sum(soft_pred_v1 * torch.log(soft_pred_v1 + 1e-8), dim=1))
            prior_penalty_v1 = torch.sum(prior * torch.log(prior / soft_pred_v1.mean(0) + 1e-8))

            loss_newv1 += weight_prior * prior_penalty_v1 + weight_conf * conf_penalty_v1
            loss_newv1.backward()
            optimizer_fcv1.step()

            # View 2
            optimizer_fcv2.zero_grad()
            pred_fcv2 = model_fcv2(subfeat_v2)
            hard_labels_v2 = torch.argmax(sublabel_v2, dim=1)
            loss_newv2 = loss_ce(pred_fcv2, hard_labels_v2)

            soft_pred_v2 = F.softmax(pred_fcv2, dim=1)
            conf_penalty_v2 = torch.mean(torch.sum(soft_pred_v2 * torch.log(soft_pred_v2 + 1e-8), dim=1))
            prior_penalty_v2 = torch.sum(prior * torch.log(prior / soft_pred_v2.mean(0) + 1e-8))

            loss_newv2 += weight_prior * prior_penalty_v2 + weight_conf * conf_penalty_v2
            loss_newv2.backward()
            optimizer_fcv2.step()

        # EAC accuracy on current batch
        predict_newfcv1 = model_fcv1(batchfeat_v1)
        predict_newfcv2 = model_fcv2(batchfeat_v2)
        acc_fcv1 = (torch.argmax(predict_newfcv1, dim=1) == correct_batch).float().mean().item() * 100
        acc_fcv2 = (torch.argmax(predict_newfcv2, dim=1) == correct_batch).float().mean().item() * 100

        # ==================== Label Swap (Only when i == epoch-1 and i % epoch_T == 0) ====================
        if i % epoch_T == 0 and i != 0 and i == epoch - 1:
            with torch.no_grad():
                train_labels_v1[start_idx:end_idx] = predict_newfcv2.detach()
                train_labels_v2[start_idx:end_idx] = predict_newfcv1.detach()

        # Print training info
        print(datetime.datetime.now(),
              f' | Epoch [{i+1}/{epoch}] Batch [{j+1}/{iteration}] '
              f'acc_ori_v1: {acc_ori_v1:.2f}% acc_ori_v2: {acc_ori_v2:.2f}% '
              f'acc_fcv1: {acc_fcv1:.2f}% acc_fcv2: {acc_fcv2:.2f}% '
              f'loss_v1: {loss_v1.item():.4f} loss_v2: {loss_v2.item():.4f} '
              f'loss_fcv1: {loss_newv1.item():.4f} loss_fcv2: {loss_newv2.item():.4f}')

    # ==================== Test Evaluation ====================
    with torch.no_grad():
        test_feat_gpu = test_features_v1.to(device)
        test_labels_gpu = test_labels.to(device)

        # Closed-form linear probe
        acc_test_linear_v1 = (torch.argmax(test_feat_gpu @ W1_v1, dim=1) == test_labels_gpu).float().mean().item() * 100
        acc_test_linear_v2 = (torch.argmax(test_feat_gpu @ W1_v2, dim=1) == test_labels_gpu).float().mean().item() * 100

        # EAC models
        model_fcv1.eval()
        model_fcv2.eval()
        test_pred_fcv1 = model_fcv1(test_feat_gpu)
        test_pred_fcv2 = model_fcv2(test_feat_gpu)

        acc_test_fcv1 = (torch.argmax(test_pred_fcv1, dim=1) == test_labels_gpu).float().mean().item() * 100
        acc_test_fcv2 = (torch.argmax(test_pred_fcv2, dim=1) == test_labels_gpu).float().mean().item() * 100

        # Ensemble
        ensembled = (test_pred_fcv1 + test_pred_fcv2) / 2.0
        ensembled = F.softmax(ensembled, dim=1)
        ensem_acc = (torch.argmax(ensembled, dim=1) == test_labels_gpu).float().mean().item() * 100

        maxacc = max(maxacc, ensem_acc)

        print("=" * 60)
        print(f">>> Epoch {i+1} Test Results <<<")
        print(f"Max Acc: {maxacc:.2f}% | Current Ensemble Acc: {ensem_acc:.2f}%")
        print(f"Linear Probe v1: {acc_test_linear_v1:.2f}% | v2: {acc_test_linear_v2:.2f}%")
        print(f"EAC Linear v1: {acc_test_fcv1:.2f}% | v2: {acc_test_fcv2:.2f}%")
        print("=" * 60)

print("Training completed!")