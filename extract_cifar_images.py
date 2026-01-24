import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm  # 可选，用于显示进度条

def extract_cifar_to_images(dataset_name, root_dir, output_dir, is_train=True):
    """
    将CIFAR数据集从pickle文件提取并保存为PNG图片。
    
    参数:
    - dataset_name: 'cifar10' 或 'cifar100'
    - root_dir: pickle文件的根目录，例如 './datasets/cifar-10-batches-py'
    - output_dir: 输出图片目录，例如 './datasets/cifar10_images/train'
    - is_train: True 为训练集，False 为测试集
    
    图片将按顺序保存为 00000.png, 00001.png, ..., 以确保索引对应数据集的样本顺序。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    labels = []  # 可选保存标签，但这里主要用于提取图片
    
    if dataset_name == 'cifar10':
        if is_train:
            # 加载5个训练batch
            for i in range(1, 6):
                file_path = os.path.join(root_dir, f'data_batch_{i}')
                with open(file_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    data.append(batch[b'data'])
                    labels.extend(batch[b'labels'])
            data = np.vstack(data)
        else:
            # 加载测试batch
            file_path = os.path.join(root_dir, 'test_batch')
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data = batch[b'data']
                labels = batch[b'labels']
    
    elif dataset_name == 'cifar100':
        if is_train:
            file_path = os.path.join(root_dir, 'train')
        else:
            file_path = os.path.join(root_dir, 'test')
        
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            data = batch['data']
            labels = batch['fine_labels'] if is_train else batch['fine_labels']
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 重塑数据为 (N, 32, 32, 3) 格式
    data = data.reshape((len(data), 3, 32, 32)).transpose((0, 2, 3, 1))
    
    # 保存为PNG图片，按顺序命名
    for i in tqdm(range(len(data)), desc="Saving images"):
        img = Image.fromarray(data[i])
        img_path = os.path.join(output_dir, f'{i:05d}.png')
        img.save(img_path)
    
    print(f"已保存 {len(data)} 张图片到 {output_dir}")
    print("图片文件名示例: 00000.png (对应数据集第0个样本), 00001.png (第1个样本), ...")
    print("注意: 确保你的dataloader加载数据的顺序与pickle文件一致，这样索引idx就能直接对应图片。")

# 示例用法（根据你的参数调整）
if __name__ == "__main__":
    dataset_name = 'cifar100'  # 'cifar10' 或 'cifar100'
    root_dir = './datasets/cifar-100-python'  # 从你的路径映射获取
    output_dir = f"./datasets/{dataset_name}_images/train"  # 训练集输出目录
    extract_cifar_to_images(dataset_name, root_dir, output_dir, is_train=True)
    
    # 如果需要测试集
    # test_output_dir = f"./datasets/{dataset_name}_images/test"
    # extract_cifar_to_images(dataset_name, root_dir, test_output_dir, is_train=False)