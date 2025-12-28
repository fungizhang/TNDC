import sys
# sys.path.append('/mnt/zfj/projects/phd/projects_phd/CWU')
import torch
from torchvision import datasets, transforms
# from datasets import load_dataset
from tqdm import tqdm
import os

# from load_parquet import *

# 参数
dataset_name = 'cifar100'  # stanford_cars
# save_path = '/home/zhangfeng/zhangfangjiao/projects/CWU/saved/saved_features/web-car-del_train_l_features.pth'
# save_path = '/home/zhangfeng/zhangfangjiao/projects/CWU/saved/saved_features/clothing1m_train_l_features.pth'
# save_path = '/home/zhangfeng/zhangfangjiao/projects/CWU/saved/saved_features/web-car-del15_train_l_features.pth'
save_path = f'./saved_features/{dataset_name}_features_no_aug_test.pth'
# save_path = '/mnt/zfj/projects/phd/projects_phd/CWU/saved/saved_features/webvision_features_no_aug.pth'


root_path = './saved_features'
os.makedirs(root_path, exist_ok=True)


if dataset_name == 'cifar10':
    dataset_path = './datasets/cifar-10-batches-py' 
elif dataset_name == 'cifar100':
    dataset_path = './datasets/cifar-100-python'
elif dataset_name == 'web-aircraft':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-aircraft'
elif dataset_name == 'web-bird':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-bird'
elif dataset_name == 'web-car':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-car'
elif dataset_name == 'web-car-del15':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-car-del15'
elif dataset_name == 'web-bird-del23':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-bird-del23'
elif dataset_name == 'web-aircraft-del21':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/NPN/web-aircraft-del21'
elif dataset_name == 'clothing1m':
    dataset_path = '/home/zhangfeng/zhangfangjiao/datasets/clothing1m'
elif dataset_name == 'webvision':
    dataset_path = '/mnt/home/zfj/'
# 检查是否有可用的 GPU，并使用 cuda:1
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



########################################################################################### 数据处理

if dataset_name == 'cifar10' or dataset_name == 'cifar100':
    from dataloader import dataloader_cifar as dataloader
    loader = dataloader.cifar_dataloader(dataset_name, noise_mode='sym', noise_ratio=0, batch_size=64, num_workers=8, 
                                        root_dir=dataset_path, model='dino')
    train_loader = loader.run('eval')
    test_loader = loader.run('test')

    noise_idx = train_loader.dataset.noise_idx
    noise_label = torch.tensor(train_loader.dataset.noise_label).to(device)
    clean_label = torch.tensor(train_loader.dataset.clean_label).to(device)
    if dataset_name == 'cifar10':
        num_class = 10
    elif dataset_name == 'cifar100':
        num_class = 100


elif dataset_name == "tiny_imagenet":
    print("Loading Tiny-ImageNet...")
    import sys
    sys.path.append('/mnt/zfj/projects/phd/projects_phd/DeFT-main')
    from utils.config import _C as cfg
    from dataloader import dataloader_tiny_imagenet as dataloader

    cfg.defrost()
    cfg.merge_from_file('/mnt/zfj/projects/phd/projects_phd/DeFT-main/config/PEFT/tiny_imagenet.yaml')
    train_loader, eval_loader, test_loader = dataloader.build_loader(cfg)

    # 输出训练集样本数量
    print(f"Number of training samples: {len(eval_loader.dataset)}")

    noise_idx = eval_loader.dataset.noise_idx
    noise_label = torch.tensor(eval_loader.dataset.noise_label).to(device)
    clean_label = torch.tensor(eval_loader.dataset.clean_label).to(device)
    num_class = 200

        
elif dataset_name == 'clothing1m':
    import sys
    sys.path.append('/home/zhangfeng/zhangfangjiao/projects/DeFT-main/')
    from dataloader import dataloader_clothing1M as dataloader
    from utils.config import _C as cfg
    cfg.defrost()
    cfg.merge_from_file('/home/zhangfeng/zhangfangjiao/projects/DeFT-main/config/PEFT/clothing1m.yaml')
    train_loader, _, test_loader = dataloader.build_loader(cfg)

    ###### train_loader
    list_train_imgs = train_loader.dataset.train_imgs
    dict_noisy_label = train_loader.dataset.train_labels
    dict_clean_label = train_loader.dataset.test_labels

    noise_label = [dict_noisy_label[p] for p in list_train_imgs]
    ### 没有完整的干净标签，所以这里也用dict_noisy_label，准确率就不用看了
    clean_label = [dict_noisy_label[p] for p in list_train_imgs]

    noise_label = torch.tensor(noise_label).to(device)
    clean_label = torch.tensor(clean_label).to(device)
    num_class = 14


    ###### test_loader
    # list_test_imgs = test_loader.dataset.test_imgs
    # dict_noisy_label = test_loader.dataset.test_labels
    # dict_clean_label = test_loader.dataset.test_labels

    # noise_label = [dict_noisy_label[p] for p in list_test_imgs]
    # ### 没有完整的干净标签，所以这里也用dict_noisy_label，准确率就不用看了
    # clean_label = [dict_noisy_label[p] for p in list_test_imgs]

    # noise_label = torch.tensor(noise_label).to(device)
    # clean_label = torch.tensor(clean_label).to(device)
    # num_class = 14

elif dataset_name == 'stanford_cars':
    print("Loading Stanford Cars...")
    import sys
    sys.path.append('/home/zhangfeng/zhangfangjiao/projects/DeFT-main/')
    from dataloader import dataloader_stanford_cars as dataloader
    from utils.config import _C as cfg
    cfg.defrost()
    cfg.merge_from_file('/home/zhangfeng/zhangfangjiao/projects/DeFT-main/config/PEFT/stanford_cars.yaml')   
    train_loader, _, test_loader = dataloader.build_loader(cfg)
    
    noise_idx = train_loader.dataset.noise_idx
    noise_label = torch.tensor(train_loader.dataset.noise_label).to(device)
    clean_label = torch.tensor(train_loader.dataset.clean_label).to(device)
    num_class = 196

elif dataset_name.startswith('web-'):
    sys.path.append('/home/zhangfeng/zhangfangjiao/projects/NPN-main')
    from util import *
    from utils.builder import *
    from torch.utils.data import *
    class_ = {"web-aircraft": 100, "web-bird": 200, "web-car": 196, "web-aircraft-del21": 100, "web-bird-del23": 200, "web-car-del15": 196}
    num_classes = class_[dataset_name]
    transform = build_transform(rescale_size=448, crop_size=448)
    dataset = build_webfg_dataset(dataset_path,
                                    CLDataTransform(transform['train'], transform["train_strong_aug"]),
                                    transform['test'])
    train_loader = DataLoader(dataset["train"], batch_size=64, shuffle=False, num_workers=4,
                                pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=4,
                                pin_memory=False)
    num_samples = len(train_loader.dataset)
    return_dict = {'trainloader': train_loader, 'num_classes': num_classes, 'num_samples': num_samples, 'dataset': dataset_name}
    return_dict['test_loader'] = test_loader

    #### val  验证dino在测试集的效果
    # train_loader = DataLoader(dataset["test"], batch_size=64, shuffle=True, num_workers=4,
    #                             pin_memory=True)

elif dataset_name == 'webvision':
    sys.path.append('/mnt/zfj/projects/phd/projects_phd/DivideMix-master')
    import dataloader_webvision as dataloader
    stats_log=open('./checkpoint/%s'%('AAAA')+'_stats.txt','w') 
    loader = dataloader.webvision_dataloader(batch_size=32,num_workers=5,root_dir=dataset_path, log=stats_log, num_class=50)
    # web_valloader = loader.run('test')
    train_loader = loader.run('eval_train')
  



########################################################################################## 模型载入
# 加载 DINOv2 模型并移到指定 GPU
# dino_model = torch.hub.load('/mnt/zfj/projects/dinov2-main', 'dinov2_vitl14', source='local', pretrained=False)
# dino_model.load_state_dict(torch.load('/mnt/zfj/dataset/models/dinov2_vitl14_pretrain.pth'))
dino_model = torch.hub.load('/mnt/zfj/projects/dinov2-main', 'dinov2_vitl14_reg', source='local', pretrained=False)
dino_model.load_state_dict(torch.load('/mnt/zfj/dataset/models/dinov2_vitl14_reg4_pretrain.pth'))
dino_model = dino_model.to(device)  # 移动模型到 cuda:1

# print(dino_model.keys())





################################################################################################### 提取并保存特征


all_features = []
all_labels = []

dino_model.eval()
with torch.no_grad():
    if dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'stanford_cars' or dataset_name == 'clothing1m':
        # for images, labels, _, _ in tqdm(train_loader, desc="Extracting features", total=len(train_loader)):
        for images, labels in tqdm(test_loader, desc="Extracting features", total=len(test_loader)):
            # 将数据移动到 cuda:1
            images = images.to(device)
            labels = labels.to(device)

            features = dino_model(images)  
            all_features.append(features.cpu())  # 可选：移回 CPU 节省内存
            all_labels.append(labels.cpu())
            # break

    elif dataset_name == 'tiny_imagenet':
        for images, labels, _, _ in tqdm(train_loader, desc="Extracting features", total=len(train_loader)):
        # for images, labels in tqdm(test_loader, desc="Extracting features", total=len(test_loader)):
            # 将数据移动到 cuda:1
            images = images.to(device)
            labels = labels.to(device)

            features = dino_model(images)  
            all_features.append(features.cpu())  # 可选：移回 CPU 节省内存
            all_labels.append(labels.cpu())

    elif dataset_name.startswith('web-'):
        
        for datas in tqdm(train_loader, desc="Extracting features", total=len(train_loader)):
            # 将数据移动到 cuda:1
            images = datas['data'][0].to(device)
            # images = datas['data'].to(device)
            labels = datas['label'].to(device)

            features = dino_model(images)  
            all_features.append(features.cpu())  # 可选：移回 CPU 节省内存
            all_labels.append(labels.cpu())

    elif dataset_name == 'webvision':
        for images, labels, index in tqdm(train_loader, desc="Extracting features", total=len(train_loader)):
        # for images, labels in tqdm(test_loader, desc="Extracting features", total=len(test_loader)):
            # 将数据移动到 cuda:1
            images = images.to(device)
            labels = labels.to(device)

            features = dino_model(images)  
            all_features.append(features.cpu())  # 可选：移回 CPU 节省内存
            all_labels.append(labels.cpu())



# 合并所有 batch 的结果（在 CPU 上）
features_tensor = torch.cat(all_features, dim=0) 
labels_tensor = torch.cat(all_labels, dim=0)



########################################################################################################### 保存到文件
# 保存 图片路径、特征、标签

# 下述代码少了 图片路径
torch.save({
    'features': features_tensor,
    'labels': labels_tensor
}, save_path)

print(f"Features and labels saved to {save_path}")





# #################################################################################################### 加载之前保存的特征，计算knn acc

checkpoint = torch.load(save_path)
features_tensor = checkpoint['features']  #
labels_tensor = checkpoint['labels']      # shape: (50000, )

print("Features and labels loaded successfully.")



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm_notebook  # 更适合脚本/终端使用的 tqdm

# 如果你在 GPU 上没有 .cpu()，请先移动到 CPU 并转换为 numpy 数组
features_np = features_tensor.numpy()
labels_np = labels_tensor.numpy()

# 初始化 KNN 分类器（例如 k=5）
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')

# 训练 KNN（实际是存储特征和标签）
print("Fitting KNN...")
knn.fit(features_np, labels_np)

# 使用 tqdm 包裹 predict，手动分批次预测以显示进度条
batch_size = 1024
preds = []
for i in tqdm(range(0, len(features_np), batch_size), desc="KNN Predicting"):
    batch = features_np[i:i+batch_size]
    preds.extend(knn.predict(batch))

acc = accuracy_score(labels_np, preds)

print(f"KNN Accuracy on {dataset_name} train set: {acc * 100:.2f}%")




################################################################## 可视化
# from sklearn.manifold import TSNE

# # 使用 t-SNE 将特征降到 2 维，便于可视化
# print("Fitting t-SNE...")
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
# features_2d = tsne.fit_transform(features_np)  # features_np 是 (50000, 384)

# print("t-SNE completed. Reduced to 2D.")

# import matplotlib.pyplot as plt

# # 可视化设置
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
#                       c=labels_np, cmap='tab10', s=2, alpha=0.6)

# # 添加颜色条（对应类别）
# plt.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in range(10)],
#            title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.title("t-SNE Visualization of DINOv2 Features on CIFAR-10")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('/home/zhangfeng/zhangfangjiao/figures/a2.png')