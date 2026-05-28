"""
Extract DINOv2 features for mini-WebVision (first 50 ImageNet classes).

CRITICAL: ordering. The features MUST be saved in exactly the same order as
`info/train_filelist_google.txt` (filtered to class_id < num_class), because
TNDC_mod_labels_mllm.py looks up `features[idx]` and `image_paths[idx]` for
the same sample idx.

This script uses the file list directly (NOT a torchvision-style ImageFolder)
to guarantee that ordering. The CIFAR features were extracted via
`extract_features.py`; for WebVision use THIS script instead.

Output:
    ./saved_features/webvision_mini_features_no_aug.pth
        {'features': Tensor [N, D], 'labels': Tensor [N]}
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class WebVisionFromList(Dataset):
    """Reads images directly from `info/train_filelist_google.txt` so that
    sample order is fully determined by the file list."""

    def __init__(self, root: str, transform, num_class: int = 50):
        self.root = root
        self.transform = transform
        self.num_class = num_class

        list_path = os.path.join(root, 'info', 'train_filelist_google.txt')
        if not os.path.isfile(list_path):
            raise FileNotFoundError(
                f"WebVision train list not found: {list_path}\n"
                f"Pass --root pointing at the WebVision directory containing "
                f"'info/' and 'google/'."
            )

        self.image_paths = []
        self.labels = []
        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                rel, target = parts[0], int(parts[1])
                if target < num_class:
                    self.image_paths.append(os.path.join(root, rel))
                    self.labels.append(target)
        print(f"[WebVisionFromList] {len(self.labels)} samples, "
              f"first {num_class} classes")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, required=True,
                   help='WebVision root (contains info/, google/, val_images_256/...)')
    p.add_argument('--num_class', type=int, default=50)
    p.add_argument('--save_path', type=str,
                   default='./saved_features/webvision_mini_features_no_aug.pth')
    p.add_argument('--dino_repo',  type=str,
                   default='/mnt/zfj/projects/dinov2-main',
                   help='Local DINOv2 repo (torch.hub source=local).')
    p.add_argument('--dino_weight', type=str,
                   default='/mnt/zfj/dataset/models/dinov2_vitl14_reg4_pretrain.pth',
                   help='DINOv2 ViT-L/14 (with registers) checkpoint.')
    p.add_argument('--dino_arch', type=str, default='dinov2_vitl14_reg',
                   help='Hub model name. Use the same as your CIFAR features '
                        '(default: dinov2_vitl14_reg).')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--image_size', type=int, default=224,
                   help='Input image size (DINOv2 expects 224 standard).')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[env] device={device}, save_path={args.save_path}")

    # Same normalization as DINOv2 default (ImageNet mean/std).
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    dataset = WebVisionFromList(args.root, transform=tfm,
                                num_class=args.num_class)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    print(f"[model] loading DINOv2 {args.dino_arch} from {args.dino_repo}")
    model = torch.hub.load(args.dino_repo, args.dino_arch,
                           source='local', pretrained=False)
    state = torch.load(args.dino_weight, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(device).eval()

    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device, non_blocking=True)
            feats = model(imgs)
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({'features': features, 'labels': labels}, args.save_path)
    print(f"[save] features={tuple(features.shape)}, "
          f"labels={tuple(labels.shape)} → {args.save_path}")


if __name__ == "__main__":
    main()
