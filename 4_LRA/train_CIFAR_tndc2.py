import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils.ema import EMA
import numpy as np
import random
import time
from utils.clip_wrapper import clip_img_wrap
import torch
import torchvision
import torchvision.transforms as transforms
from utils.data_utils import Custom_dataset
from utils.model_SimCLR import SimCLR_encoder
import torch.optim as optim
from utils.learning import *
from model_diffusion import Diffusion
import utils.ResNet_for_32 as resnet_s
from utils.knn_utils import sample_knn_labels
import argparse
import os
import json
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)


# def train(diffusion_model, train_dataset, val_dataset, test_dataset, model_path, args, real_fp):
#     device = diffusion_model.device
#     n_class = diffusion_model.n_class
#     n_epochs = args.nepoch
#     k = args.k
#     warmup_epochs = args.warmup_epochs

#     # # pre-compute for fp embeddings on training data
#     print('pre-computing fp embeddings for training data')
#     train_embed = prepare_fp_x(diffusion_model.fp_encoder, train_dataset, save_dir=None, device=device,
#                                fp_dim=fp_dim).to(device)

#     train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#     optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
#     diffusion_loss = nn.MSELoss(reduction='none')

#     ema_helper = EMA(mu=0.9999)
#     ema_helper.register(diffusion_model.model)

#     max_accuracy = 0.0
#     print('Diffusion training start')
#     for epoch in range(n_epochs):
#         diffusion_model.model.train()

#         with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
#             for i, data_batch in pbar:
#                 [x_batch, y_batch, data_indices] = data_batch[:3]

#                 if real_fp:
#                     # compute embeddings for augmented images for better performance
#                     fp_embd = diffusion_model.fp_encoder(x_batch.to(device))
#                 else:
#                     # use pre-compute embedding for efficiency
#                     fp_embd = train_embed[data_indices, :]

#                 # sample a knn labels and compute weight for the sample
#                 y_labels_batch, sample_weight = sample_knn_labels(fp_embd, y_batch.to(device), train_embed,
#                                                                   torch.tensor(train_dataset.targets).to(device),
#                                                                   k=k, n_class=n_class, weighted=True)

#                 # convert label to one-hot vector
#                 y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch.to(torch.int64),
#                                                                                       n_class=n_class)
#                 y_0_batch = y_one_hot_batch.to(device)

#                 # adjust_learning_rate
#                 adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=1000, lr_input=0.001)
#                 n = x_batch.size(0)

#                 # sampling t
#                 t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1,)).to(device)
#                 t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

#                 # train with and without prior
#                 output, e = diffusion_model.forward_t(y_0_batch, x_batch, t, fp_embd)

#                 # compute loss
#                 mse_loss = diffusion_loss(e, output)
#                 weighted_mse_loss = torch.matmul(sample_weight, mse_loss)
#                 loss = torch.mean(weighted_mse_loss)
#                 pbar.set_postfix({'loss': loss.item()})
#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
#                 optimizer.step()
#                 ema_helper.update(diffusion_model.model)

#         # if epoch % 5 == 0 and epoch >= warmup_epochs:
#         #     val_acc = test(diffusion_model, val_loader)
#         #     print(f"epoch: {epoch}, validation accuracy: {val_acc:.2f}%")
#         #     if val_acc > max_accuracy:
#         #         # save diffusion model
#         #         print('Improved! evaluate on testing set...')
#         #         test_acc = test(diffusion_model, test_loader)
#         #         states = [diffusion_model.model.state_dict(),
#         #                   diffusion_model.diffusion_encoder.state_dict(),
#         #                   diffusion_model.fp_encoder.state_dict()]
#         #         torch.save(states, model_path)
#         #         print(f"Model saved, update best accuracy at Epoch {epoch}, val acc: {val_acc}, test acc: {test_acc}")
#         #         max_accuracy = max(max_accuracy, val_acc)

#         # if epoch % 1 == 0 and epoch >= warmup_epochs:
#         # save diffusion model
#         print('Improved! evaluate on testing set...')
#         test_acc = test(diffusion_model, test_loader)
#         states = [diffusion_model.model.state_dict(),
#                     diffusion_model.diffusion_encoder.state_dict(),
#                     diffusion_model.fp_encoder.state_dict()]
#         torch.save(states, model_path)
#         print(f"Model saved, update best accuracy at Epoch {epoch}, test acc: {test_acc}")



def train(diffusion_model, train_dataset, val_dataset, test_dataset, model_path, args, real_fp, clean_targets):
    device = diffusion_model.device
    n_class = diffusion_model.n_class
    n_epochs = args.nepoch
    k = args.k
    warmup_epochs = args.warmup_epochs
    batch_size = args.batch_size


    # >>> 新增：设置结果保存路径 <<<
    result_dir = "/mnt/zfj/exp_results/LRA"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{args.noise_type}_tndc2.json")
    # 初始化记录字典（如果文件已存在则加载）
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            acc_dict = json.load(f)
    else:
        acc_dict = {}


    print('pre-computing fp embeddings for training data')
    train_embed = prepare_fp_x(diffusion_model.fp_encoder, train_dataset, save_dir=None, device=device,
                               fp_dim=fp_dim).to(device)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')
    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)

    max_accuracy = 0.0
    print('Diffusion training start')

    for epoch in range(n_epochs):
        diffusion_model.model.train()
        knn_correct_total = 0
        knn_total_samples = 0

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                [x_batch, y_batch, data_indices] = data_batch[:3]

                if real_fp:
                    fp_embd = diffusion_model.fp_encoder(x_batch.to(device))
                else:
                    fp_embd = train_embed[data_indices, :]

                y_labels_batch, sample_weight = sample_knn_labels(
                    fp_embd, y_batch.to(device), train_embed,
                    torch.tensor(train_dataset.targets).to(device),
                    k=k, n_class=n_class, weighted=True
                )

                # >>> 新增：计算 KNN 修正标签 vs 干净标签准确率 <<<
                clean_labels_batch = torch.tensor([clean_targets[idx] for idx in data_indices], device=device)
                knn_correct = (y_labels_batch == clean_labels_batch).sum().item()
                knn_correct_total += knn_correct
                knn_total_samples += y_labels_batch.size(0)

                y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(
                    y_labels_batch.to(torch.int64), n_class=n_class
                )
                y_0_batch = y_one_hot_batch.to(device)

                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=1000, lr_input=0.001)
                n = x_batch.size(0)
                t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1,)).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                output, e = diffusion_model.forward_t(y_0_batch, x_batch, t, fp_embd)
                mse_loss = diffusion_loss(e, output)
                weighted_mse_loss = torch.matmul(sample_weight, mse_loss)
                loss = torch.mean(weighted_mse_loss)
                pbar.set_postfix({'loss': loss.item(), 'knn_acc': f"{100*knn_correct/knn_total_samples:.2f}%"})

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model.model)

        # 打印整个 epoch 的 KNN 修正准确率
        epoch_knn_acc = 100.0 * knn_correct_total / knn_total_samples
        print(f"Epoch {epoch}: KNN pseudo-label accuracy vs clean labels: {epoch_knn_acc:.2f}%")

        # 保存模型（按你原来的逻辑）
        print('Improved! evaluate on testing set...')
        test_acc = test(diffusion_model, test_loader)
        states = [
            diffusion_model.model.state_dict(),
            diffusion_model.diffusion_encoder.state_dict(),
            diffusion_model.fp_encoder.state_dict()
        ]
        torch.save(states, model_path)
        print(f"Model saved at Epoch {epoch}, test acc: {test_acc}")


        # >>> 新增：保存测试准确率到 JSON <<<
        acc_dict[str(epoch)] = float(test_acc)  # 转为 float 确保 JSON 可序列化
        with open(result_file, 'w') as f:
            json.dump(acc_dict, f, indent=4)
        print(f"Test accuracy saved to {result_file}")




def test(diffusion_model, test_loader):
    start = time.time()
    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        correct_cnt = 0
        all_cnt = 0
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Doing DDIM...', ncols=100):
            [images, target, _] = data_batch[:3]
            target = target.to(device)

            label_t_0 = diffusion_model.reverse_ddim(images, stochastic=False, fq_x=None).detach().cpu()
            # correct = cnt_agree(label_t_0.detach().cpu(), target.cpu())[0].item()
            correct = cnt_agree(label_t_0.detach().cpu(), target.cpu())
            correct_cnt += correct
            all_cnt += images.shape[0]

    print(f'time cost for CLR: {time.time() - start}')

    acc = 100 * correct_cnt / all_cnt
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_type', default='cifar100-1-0.35', help='noise label file', type=str)
    parser.add_argument("--nepoch", default=200, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=128, help="batch_size", type=int)
    parser.add_argument("--device", default='cuda:1', help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=8, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    parser.add_argument("--feature_dim", default=512, help="feature_dim", type=int)
    parser.add_argument("--k", default=10, help="k neighbors for knn", type=int)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--fp_encoder", default='CLIP', help="which encoder for fp (SimCLR or CLIP)", type=str)
    parser.add_argument("--CLIP_type", default='ViT-L/14', help="which encoder for CLIP", type=str)
    parser.add_argument("--diff_encoder", default='resnet18', help="which encoder for diffusion (linear, resnet18, 34, 50...)", type=str)
    args = parser.parse_args()

    # set device
    device = args.device
    print('Using device:', device)

    dataset = args.noise_type.split('-')[0]
    noisy_mode = args.noise_type.split('-')[1]
    noisy_ratio = args.noise_type.split('-')[2]

    # load datasets
    if dataset == 'cifar10':
        n_class = 10
        train_dataset_cifar = torchvision.datasets.CIFAR10(root='/mnt/zfj/dataset', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR10(root='/mnt/zfj/dataset', train=False, download=True)

    elif dataset == 'cifar100':
        n_class = 100
        train_dataset_cifar = torchvision.datasets.CIFAR100(root='/mnt/zfj/dataset', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR100(root='/mnt/zfj/dataset', train=False, download=True)
    else:
        raise Exception("Date should be cifar10 or cifar100")

    # load fp encoder
    if args.fp_encoder == 'SimCLR':
        fp_dim = 2048
        real_fp = True
        state_dict = torch.load(f'./model/SimCLR_128_{dataset}.pt', map_location=torch.device(args.device))
        fp_encoder = SimCLR_encoder(feature_dim=128).to(args.device)
        fp_encoder.load_state_dict(state_dict, strict=False)
    elif args.fp_encoder == 'CLIP':
        real_fp = False
        fp_encoder = clip_img_wrap(args.CLIP_type, args.device, center=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        fp_dim = fp_encoder.dim
    else:
        raise Exception("fp_encoder should be SimCLR or CLIP")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    # 保存干净标签（用于后续评估KNN修正质量）
    clean_targets = train_dataset_cifar.targets[:50000]  # 原始干净标签


    train_dataset = Custom_dataset(train_dataset_cifar.data[:50000], train_dataset_cifar.targets[:50000],
                                   transform=transform_train)
    val_dataset = Custom_dataset(train_dataset_cifar.data[45000:], train_dataset_cifar.targets[45000:])
    test_dataset = Custom_dataset(test_dataset_cifar.data, test_dataset_cifar.targets)

    batch_size = args.batch_size

    # load noisy label
    # noise_label = np.load('./noise_label/' + args.noise_type + '.npy')
    # import json
    # if dataset == 'cifar10':
    #     noise_label = json.load(open(f'/mnt/zfj/dataset/cifar-10-batches-py/noise_file/cifar10_{noisy_mode}_{noisy_ratio}',"r"))
    # elif dataset == 'cifar100':
    #     noise_label = json.load(open(f'/mnt/zfj/dataset/cifar-100-python/noise_file/cifar100_{noisy_mode}_{noisy_ratio}',"r"))
    
    ######  tndc
    if dataset == 'cifar10':
        noise_label = json.load(open(f'/mnt/zfj/dataset/dino_mod/cifar10/dino_mod_labels_{noisy_mode}_{noisy_ratio}.json',"r"))
    elif dataset == 'cifar100':
        noise_label = json.load(open(f'/mnt/zfj/dataset/dino_mod/cifar100/dino_mod_labels_{noisy_mode}_{noisy_ratio}.json',"r"))
    # noise_label = json.load(open('/mnt/zfj/dataset/cifar-10-batches-py/noise_file/cifar10_{noisy_mode}_{noisy_ratio}',"r"))
    # noise_label = json.load(open(f'/mnt/zfj/dataset/cifar-100-python/noise_file/cifar100_{noisy_mode}_{noisy_ratio}',"r"))
    noise_label = np.array(noise_label)
    train_dataset.update_label(noise_label)
    print('Training on noise label:', args.noise_type)

    # initialize diffusion model
    model_path = f'./model/LRA-diffusion_{args.fp_encoder}_{args.noise_type}.pt'
    diffusion_model = Diffusion(fp_encoder=fp_encoder, n_class=n_class, fp_dim=fp_dim, feature_dim=args.feature_dim,
                                device=device, encoder_type=args.diff_encoder, ddim_num_steps=args.ddim_n_step)
    # state_dict = torch.load(model_path, map_location=torch.device(device))
    # diffusion_model.load_diffusion_net(state_dict)
    diffusion_model.fp_encoder.eval()

    # train the diffusion model
    print(f'training LRA-diffusion using fp encoder: {args.fp_encoder} on: {args.noise_type}.')
    print(f'model saving dir: {model_path}')
    # train(diffusion_model, train_dataset, val_dataset, test_dataset, model_path, args, real_fp=real_fp)
    train(diffusion_model, train_dataset, val_dataset, test_dataset, model_path, args, real_fp=real_fp, clean_targets=clean_targets)



