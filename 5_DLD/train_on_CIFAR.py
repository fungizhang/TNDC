import tqdm
import json
import random
import torch
import argparse
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from torch.utils.data import RandomSampler
import torchvision
from utils.ema import EMA
from utils.vit_wrapper import vit_img_wrap
from utils.cifar_data_utils import Custom_dataset, Double_dataset
from utils.model_ResNet import ResNet_encoder
from utils.model_SimCLR import SimCLR_encoder
import torch.optim as optim
from utils.learning import *
from utils.pre_correction import *
from utils.ws_augmentation import *
from utils.directional_diffusion_model import * 
from utils.add_ccn_noise import *
from utils.log_config import setup_logger

# Main training function (diffusion model, training, validation, test sets, model save path, command line arguments, encoder)
def train(diffusion_model, train_dataset, test_dataset, model_path, args, vit_fp, fp_dim):
    """
    Train the diffusion model with the given datasets and arguments.

    Parameters:
    - diffusion_model: The diffusion model to be trained.
    - train_dataset: The dataset used for training.
    - test_dataset: The dataset used for testing.
    - val_dataset: The dataset used for validation.
    - model_path: Path to save the trained model.
    - args: Command line arguments containing training parameters.
    - vit_fp: Whether to use precomputed feature embeddings.
    - fp_dim: Dimension of the feature embeddings.
    """
    print(f'Use loss weights: {args.loss_w}, Use Single label: {args.to_single_label}, Use One view: {args.one_view}')
    # Extract configurations from the model and command line arguments, including training device, number of classes, total training epochs, k value in KNN, and warmup epochs.
    device = diffusion_model.device
    n_class = diffusion_model.n_class
    num_models = diffusion_model.num_models
    n_epochs = args.nepoch
    k = args.k
    warmup_epochs = args.warmup_epochs
    dataset = args.noise_type.split('-')[0]
    noise_class = args.noise_type.split('-')[1]
    noise_ratio = float(args.noise_type.split('-')[2])


    clean_labels = torch.tensor(train_dataset.targets).squeeze().to(device)


    if dataset == 'cifar10':
            noisy_targets = json.load(open(f'/mnt/zfj/dataset/cifar-10-batches-py/noise_file/cifar10_{noise_class}_{noise_ratio}',"r"))
    elif  dataset == 'cifar100':
        noisy_targets = json.load(open(f'/mnt/zfj/dataset/cifar-100-python/noise_file/cifar100_{noise_class}_{noise_ratio}',"r"))
    noisy_targets = np.array(noisy_targets)

    # /mnt/zfj/dataset/cifar-10-batches-py/noise_file/cifar10_sym_0.2
    # noisy_targets = add_noise(train_dataset.targets, noise_ratio, n_class, seed=None, symmetric_noise=True)
    train_dataset.update_label(noisy_targets)
    noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)
    print(f'Training on {args.noise_type} label noise:')


    # if noise_class == 'idn':
    #     # # Add noise, first pass in the noisy label set, and then update the labels of the training set.
    #     # if args.noise_type == 'cifar10-idn-0.0':
    #     #     print('Training on pure label:', args.noise_type)
    #     # else:
    #     #     noise_label = np.load(f'./noise_label_IDN/{args.noise_type}.npy')
    #     #     train_dataset.update_label(noise_label[:])
    #     #     print(f'Training on {args.noise_type} label noise:')
    #     # noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)

    #     # Add noise, first pass in the noisy label set, and then update the labels of the training set.
    #     if args.noise_type == 'cifar10-idn-0.0':
    #         print('Training on pure label:', args.noise_type)
    #         noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)
    #     else:
    #         # Map args.noise_type to the new JSON file path
    #         # e.g., 'cifar10-idn-0.2' -> 'cifar10_idn_0.2.json'
    #         json_filename = args.noise_type.replace('-', '_')
    #         json_path = os.path.join('/mnt/zfj/dataset/cifar-10-batches-py/noise_file', json_filename)

    #         # Load noise label from JSON
    #         with open(json_path, 'r') as f:
    #             noise_label = json.load(f)

    #         # Ensure it's a list or array of integers
    #         noise_label = np.array(noise_label, dtype=np.int64)

    #         train_dataset.update_label(noise_label[:])
    #         print(f'Training on {args.noise_type} label noise:')
    #         noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)
        
    # elif noise_class == 'sym':
    #     if dataset_name == 'cifar10':
    #         noisy_targets = json.load(open(f'/mnt/zfj/dataset/cifar-10-batches-py/noise_file/cifar10_{noise_class}_{noise_ratio}',"r"))
    #     elif  dataset_name == 'cifar100':
    #         noisy_targets = json.load(open(f'/mnt/zfj/dataset/cifar-100-python/noise_file/cifar100_{noise_class}_{noise_ratio}',"r"))
    #     noisy_targets = np.array(noisy_targets)

    #     # /mnt/zfj/dataset/cifar-10-batches-py/noise_file/cifar10_sym_0.2
    #     # noisy_targets = add_noise(train_dataset.targets, noise_ratio, n_class, seed=None, symmetric_noise=True)
    #     train_dataset.update_label(noisy_targets)
    #     noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)
    #     print(f'Training on {args.noise_type} label noise:')
    # elif noise_class == 'asym':
    #     noisy_targets = add_noise(train_dataset.targets, noise_ratio, n_class, seed=None, symmetric_noise=False)
    #     train_dataset.update_label(noisy_targets)
    #     noisy_labels = torch.tensor(train_dataset.targets).squeeze().to(device)
    #     print(f'Training on {args.noise_type} label noise:')
    # elif noise_class == 'law_mod':
    #     nr = int(noise_ratio * 100)
    #     noisy_labels = json.load(open(f'/mnt/zfj/dataset/dino_mod/cifar10/dino_mod_labels_asym_{nr}.json',"r"))
    
    #     train_dataset.update_label(noisy_labels)
    #     noisy_labels = torch.tensor(noisy_labels).squeeze().to(device)
    # else:
    #     print("Check your noise type carefully!")



    

    # Compute embedding fp(x) for ws_dataset
    
    if dataset == 'cifar10':
        data_dir  = os.path.join(os.getcwd(), './data/cifar-10-batches-py')
    else:
        data_dir  = os.path.join(os.getcwd(), './data/cifar-100-python')
    train_embed_dir = os.path.join(data_dir, f'fp_embed_train_cifar')
    # Compute embedding fp(x) for ws_dataset
    print('Doing pre-computing fp embeddings for weak and strong dataset')
    weak_embed, strong_embed = prepare_2_fp_x(fp_encoder, train_dataset, save_dir=train_embed_dir, device=device, fp_dim=fp_dim)
    weak_embed = weak_embed.to(device)
    strong_embed = strong_embed.to(device)


    sampler = RandomSampler(train_dataset)  # 生成与shuffle=True相同的随机索引

    # 获取所有样本的shuffle后索引（可提前查看全局顺序）
    all_shuffled_indices = list(sampler)


    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=4)

    # Optimizer settings for dual networks
    if diffusion_model.num_models == 1:
        optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
        ema_helper = EMA(mu=0.999)
        ema_helper.register(diffusion_model.model)
    elif diffusion_model.num_models == 2:
        optimizer_res = optim.Adam(diffusion_model.model0.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
        optimizer_noise = optim.Adam(diffusion_model.model1.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
        # Initialize EMA helper and register model parameters to smooth model parameter updates during training to improve model stability and performance.
        ema_helper_res = EMA(mu=0.999)
        ema_helper_noise = EMA(mu=0.999)
        ema_helper_res.register(diffusion_model.model0)
        ema_helper_noise.register(diffusion_model.model1)

    diffusion_loss = nn.MSELoss(reduction='none')

    # Train in a loop and record the highest accuracy to save the model
    max_accuracy = 0.0
    print('Directional Diffusion training start')
    for epoch in range(n_epochs):
        if diffusion_model.num_models == 1:
            diffusion_model.model.train()
        else:
            diffusion_model.model0.train()
            diffusion_model.model1.train()
        total_loss = 0.0
        total_batches = 0

        ### zfj 
        total_correct_w = 0  # 新增：累计y_label_batch_w的正确数量
        total_correct_s = 0  # 新增：累计y_label_batch_s的正确数量
        total_samples = 0    # 新增：累计样本总数

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                [x_batch_w, x_batch_s, y_batch, data_indices] = data_batch[:4]
                x_batch_w = x_batch_w.to(device)
                x_batch_s = x_batch_s.to(device)
                y_noisy = y_batch.to(device)

                if vit_fp:
                    # Use precomputed feature embeddings
                    fp_embd_w = weak_embed[data_indices, :].to(device)
                    fp_embd_s = strong_embed[data_indices, :].to(device)
                else:
                    # Compute feature embeddings in real-time
                    fp_embd_w = diffusion_model.fp_encoder(x_batch_w.to(device))
                    fp_embd_s = diffusion_model.fp_encoder(x_batch_s.to(device))

                if noise_class != 'law_mod' or dataset == 'cifar10':
                    # pre-correct labels based on two views
                    y_label_batch_w, y_label_batch_s, loss_weights_w, loss_weights_s, y_label_batch_n, gamma_batch = \
                    precorrect_labels_in_two_view(fp_embd_w=fp_embd_w, fp_embd_s=fp_embd_s, y_noisy=y_noisy, 
                                                weak_embed=weak_embed, strong_embed=strong_embed, noisy_labels=noisy_labels, 
                                                k=k, n_class=n_class, use_cosine_similarity=args.use_cos, 
                                                to_single_label=args.to_single_label)
                else:
                    # Skip pre-correction: use original noisy labels directly
                    
                    # Convert to one-hot if not using single label
                    y_label_batch_w = F.one_hot(y_noisy, num_classes=n_class).float()
                    y_label_batch_s = F.one_hot(y_noisy, num_classes=n_class).float()

                    # Optional: set uniform loss weights (or all 1.0)
                    loss_weights_w = torch.ones_like(y_noisy, dtype=torch.float32)

                    # Dummy values for unused variables (if needed later)
                    y_label_batch_n = y_label_batch_w  # or whatever placeholder
                    gamma_batch = torch.zeros_like(y_noisy, dtype=torch.float32)  # or ones, doesn't matter


                ### zfj
                # 统计y_label_batch_w和y_label_batch_s的准确性
                # 获取当前batch样本在原始数据集中的索引（基于shuffle后的全局索引）
                # batch_original_indices = [all_shuffled_indices[idx] for idx in data_indices]
                batch_original_indices = data_indices

                # 获取当前batch的干净标签（根据原始索引从clean_labels中提取）
                batch_clean_labels = clean_labels[batch_original_indices].cpu()

                # 从y_label_batch_w中获取预测标签（概率最高的类别）
                y_pred_w = torch.argmax(y_label_batch_w, dim=1).cpu()
                # 计算y_label_batch_w的准确率
                correct_w = (y_pred_w == batch_clean_labels).sum().item()
                acc_w = correct_w / len(batch_clean_labels)

                # 从y_label_batch_s中获取预测标签
                y_pred_s = torch.argmax(y_label_batch_s, dim=1).cpu()
                # 计算y_label_batch_s的准确率
                correct_s = (y_pred_s == batch_clean_labels).sum().item()
                acc_s = correct_s / len(batch_clean_labels)

                # 打印当前batch的准确率（可选，也可累计到epoch结束后打印平均）
                # pbar.set_postfix({
                #     'res_loss': l_res_loss.item() if 'l_res_loss' in locals() else None,
                #     'noise_loss': l_noise_loss.item() if 'l_noise_loss' in locals() else None,
                #     'loss': loss.item() if 'loss' in locals() else None,
                #     'acc_w': acc_w,
                #     'acc_s': acc_s
                # })

                # 累计到epoch级别的统计量
                total_correct_w += correct_w
                total_correct_s += correct_s
                total_samples += len(batch_clean_labels)








                if args.one_view:
                    x_batch = x_batch_w
                else:
                    x_batch = (1 - gamma_batch.view(-1, 1, 1, 1)) * x_batch_w + gamma_batch.view(-1, 1, 1, 1) * x_batch_s
                
                # Check if the labels are one-hot encoded, if not, convert them to vectors
                if len(y_label_batch_w.shape) == 1:
                    y_one_hot_batch_w = cast_label_to_one_hot_and_prototype(y_label_batch_w.to(torch.int64), n_class=n_class)
                    y_one_hot_batch_s = cast_label_to_one_hot_and_prototype(y_label_batch_s.to(torch.int64), n_class=n_class)
                else:
                    y_one_hot_batch_w = y_label_batch_w
                    y_one_hot_batch_s = y_label_batch_s

                y_0_batch_w = y_one_hot_batch_w.to(device)
                y_0_batch_s = y_one_hot_batch_s.to(device)
                y_zeros = torch.zeros_like(y_0_batch_w)
                y_n_batch = y_label_batch_n.to(device)
                
                #Adjust learning rate
                if diffusion_model.num_models == 1:
                    adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=n_epochs,
                    lr_input=1e-3)
                else:
                    adjust_learning_rate(optimizer_res, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=n_epochs, lr_input=1e-3)
                    adjust_learning_rate(optimizer_noise, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=n_epochs, lr_input=1e-3)

                # Sampling t for symmetric
                n = x_batch.size(0)

                t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1, )).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                ## Sampling t for random
                # t = torch.randint(0, diffusion_model.num_timesteps, (n,), device=device).long()

                # Single view training, only train weakly augmented images with precise label sampling
                
                output, e = diffusion_model.forward_t(y_zeros, y_0_batch_w, x_batch, t, fp_embd_w)

                if diffusion_model.objective == 'pred_res_noise':
                    # Calculate L_res and L_noise losses
                    L_res = diffusion_loss(output[0], e[0])
                    L_noise = diffusion_loss(output[1], e[1])

                    # Apply optional loss weighting
                    weighted_L_res = torch.matmul(loss_weights_w, L_res) if args.loss_w else L_res
                    weighted_L_noise = torch.matmul(loss_weights_w, L_noise) if args.loss_w else L_noise

                    # Mean loss calculation
                    l_res_loss = torch.mean(weighted_L_res)
                    l_noise_loss = torch.mean(weighted_L_noise)

                    if diffusion_model.num_models == 2:
                        # Update model0 using l_res_loss
                        optimizer_res.zero_grad()
                        l_res_loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(diffusion_model.model0.parameters(), 1.0)
                        optimizer_res.step()
                        ema_helper_res.update(diffusion_model.model0)

                        # Update model1 using l_noise_loss
                        optimizer_noise.zero_grad()
                        l_noise_loss.backward()
                        torch.nn.utils.clip_grad_norm_(diffusion_model.model1.parameters(), 1.0)
                        optimizer_noise.step()
                        ema_helper_noise.update(diffusion_model.model1)

                        # Update progress bar
                        pbar.set_postfix({'res_loss': l_res_loss.item(), 'noise_loss': l_noise_loss.item()})

                    elif diffusion_model.num_models == 1:  # For single model, combine losses
                        loss = 0.1 * l_res_loss + 0.9 * l_noise_loss
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                        optimizer.step()
                        ema_helper.update(diffusion_model.model)
                        pbar.set_postfix({'loss': loss.item()})

                else:  # For objectives other than 'pred_res_noise'
                    # Calculate loss without separation
                    L_noise = diffusion_loss(output, e)
                    weighted_L_noise = torch.matmul(loss_weights_w, L_noise) if args.loss_w else L_noise
                    loss = torch.mean(weighted_L_noise)

                    # Perform single optimizer step for combined objective
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                    optimizer.step()
                    ema_helper.update(diffusion_model.model)
                    pbar.set_postfix({'loss': loss.item()})

        # Every epoch, perform validation, if the validation accuracy of the current epoch exceeds the previous highest accuracy, evaluate the model on the test set, and save the current best model parameters.
        if epoch >= warmup_epochs:
            test_acc = test(diffusion_model, test_loader)
            logger.info(f"epoch: {epoch}, test accuracy: {test_acc:.2f}%")
            if test_acc > max_accuracy:
                # Save diffusion model
                print('Improved! Evaluate on testing set...')
                if diffusion_model.num_models == 1:
                    states = {
                        'model': diffusion_model.model.state_dict(),  
                        'fp_encoder': diffusion_model.fp_encoder.state_dict(),  
                    }
                else:
                    states = {
                        'model0': diffusion_model.model0.state_dict(),  
                        'model1': diffusion_model.model1.state_dict(),  
                        'fp_encoder': diffusion_model.fp_encoder.state_dict(),  
                    }
                # torch.save(states, model_path)
                message = (f"Model saved, update best accuracy at Epoch {epoch}, test acc: {test_acc}")
                logger.info(message)
                max_accuracy = max(max_accuracy, test_acc)

        ### zfj
        # 在epoch结束时打印平均准确率
        epoch_acc_w = total_correct_w / total_samples if total_samples > 0 else 0
        epoch_acc_s = total_correct_s / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch} - y_label_batch_w 准确率: {epoch_acc_w:.4f}, y_label_batch_s 准确率: {epoch_acc_s:.4f}")
        # logger.info(f"Epoch {epoch} - y_label_batch_w 准确率: {epoch_acc_w:.4f}, y_label_batch_s 准确率: {epoch_acc_s:.4f}")

def test(diffusion_model, test_loader):
    """
    Test the diffusion model with the given test loader.

    Parameters:
    - diffusion_model: The diffusion model to be tested.
    - test_loader: DataLoader for the test set.

    Returns:
    - acc: The accuracy of the model on the test set.
    """
    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        correct_cnt = 0
        all_cnt = 0
        for idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Doing DDIM...', ncols=100):
            [images, target, _] = data_batch[:3]
            target = target.to(device)
            label_t_0 = diffusion_model.ddim_sample(x_batch = images, y_input = 0, fp_x=None, last=True, stochastic=False).detach().cpu()
            correct = cnt_agree(label_t_0.detach().cpu(), target.cpu())
            correct_cnt += correct
            all_cnt += images.shape[0]

    acc = 100 * correct_cnt / all_cnt
    return acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    # Training parameters
    parser.add_argument('--noise_type', default='cifar10-idn-0.4', help='noise label file', type=str)
    parser.add_argument("--nepoch", default=200, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=200, help="batch_size", type=int)
    parser.add_argument("--device", default='cuda:5', help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=16, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    # Diffusion model hyperparameters
    parser.add_argument("--num_models", default=2, help="number of models", type=int)
    parser.add_argument("--feature_dim", default=512, help="feature_dim", type=int)
    parser.add_argument("--k", default=50, help="k neighbors for knn or cos", type=int)
    parser.add_argument("--loss_w", default=True, help="use weights for loss", action='store_false')
    parser.add_argument("--to_single_label", default=False, help="use single_label for label sampling", action='store_true')
    parser.add_argument("--one_view", default=False, help="use single view", action='store_true')
    parser.add_argument("--use_cos", default=True, help="use cos", action='store_false')
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--diff_encoder", default='resnet34', help="which encoder for diffusion (linear, resnet18, 34, 50...)", type=str)
    parser.add_argument("--objective", default='pred_res_noise', help="which type for diffusion (pred_res, pred_noise, pred_res_noise...)", type=str)

    # Large model hyperparameters
    parser.add_argument("--fp_encoder", default='ViT', help="which encoder for fp (SimCLR, Vit or ResNet)", type=str)
    parser.add_argument("--ViT_type", default='ViT-L/14', help="which encoder for Vit", type=str)
    parser.add_argument("--ResNet_type", default='resnet34', help="which encoder for ResNet", type=str)
    # Storage path
    parser.add_argument("--log_name", default='cifar100-idn-0.2aaaa.log', help="create your logs name", type=str)
    args = parser.parse_args()
    logger = setup_logger(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set GPU or CPU for training
    device = args.device
    print('Using device:', device)

    dataset = args.noise_type.split('-')[0]

    # Load dataset
    if dataset == 'cifar10':
        n_class = 10
        train_dataset_cifar = torchvision.datasets.CIFAR10(root='/mnt/zfj/dataset', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR10(root='/mnt/zfj/dataset', train=False, download=True)
        # Data normalization parameters
        CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR_STD = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        n_class = 100
        train_dataset_cifar = torchvision.datasets.CIFAR100(root='/mnt/zfj/dataset', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR100(root='/mnt/zfj/dataset', train=False, download=True)
        CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR_STD = (0.2023, 0.1994, 0.2010)
    else:
        raise Exception("Dataset should be cifar10 or cifar100")

    # Load fp feature extractor
    if args.fp_encoder == 'SimCLR':
        fp_dim = 2048
        state_dict = torch.load(f'./model/SimCLR_128_{dataset}.pt', map_location=torch.device(args.device))
        fp_encoder = SimCLR_encoder(feature_dim=128).to(args.device)
        fp_encoder.load_state_dict(state_dict, strict=False)
    elif args.fp_encoder == 'ViT':
        fp_encoder = vit_img_wrap(args.ViT_type, args.device, center=CIFAR_MEAN, std=CIFAR_STD)
        fp_dim = fp_encoder.dim
    elif args.fp_encoder == 'ResNet':
        if args.ResNet_type == 'resnet34':
            fp_dim = 512
        else:
            fp_dim = 2048
        fp_encoder = ResNet_encoder(feature_dim=fp_dim, base_model=args.ResNet_type).to(args.device)
    else:
        raise Exception("fp_encoder should be SimCLR, Vit or ResNet")

    # Create training and test set instances using custom dataset class
    transform_fixmatch = TransformFixMatch_CIFAR10(CIFAR_MEAN, CIFAR_STD, 2, 10)
    train_dataset = Double_dataset(data=train_dataset_cifar.data[:], targets=train_dataset_cifar.targets[:], transform_fixmatch=transform_fixmatch)
    test_dataset = Custom_dataset(test_dataset_cifar.data, test_dataset_cifar.targets)

    # Initialize the diffusion model
    model_path = f'./model/DLD_{args.fp_encoder}_{args.noise_type}.pt'
    base_model = DirectionalConditionalModel(n_steps = 1000, y_dim = n_class, fp_dim  = fp_dim, feature_dim = args.feature_dim, guidance = True, num_models = args.num_models, objective = args.objective, encoder_type=args.diff_encoder).to(args.device)

    diffusion_model = DirectionalDiffusion(model = base_model, fp_encoder=fp_encoder, num_models=args.num_models, num_timesteps=1000, n_class=n_class, fp_dim=fp_dim,  device=args.device, feature_dim=args.feature_dim, encoder_type=args.diff_encoder, objective= args.objective, sampling_timesteps = args.ddim_n_step, condition = True, convert_to_ddim = False, sum_scale = 1., ddim_sampling_eta=0., beta_schedule = 'cosine')

    diffusion_model.fp_encoder.eval()
    # state_dict = torch.load(model_path, map_location=torch.device(device))
    # diffusion_model.load_diffusion_net(state_dict)

    # Train the diffusion model
    print(f'Training DLD using fp encoder: {args.fp_encoder} on: {args.noise_type}.')
    print(f'Model saving dir: {model_path}')
    train(diffusion_model, train_dataset=train_dataset, test_dataset=test_dataset, model_path=model_path, args=args, vit_fp=True, fp_dim=fp_dim)

