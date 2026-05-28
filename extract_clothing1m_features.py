"""
Extract DINOv2-ViT-L/14 features for the Clothing1M 64K class-balanced subset.

DivideMix's standard practice on Clothing1M is to train on a 64,000-sample
class-balanced subset (4,571 per class * 14 classes ~= 64,000), drawn from
~1M noisy training images, rather than the full corpus. We follow the same
convention so that TNDC's preprocessing matches what downstream training
actually consumes.

This script:
  1. Loads the same 64K subset that DivideMix's clothing_dataloader produces
     (mode='all', num_samples=64000, class-balanced quota = 64000/14 per class).
  2. Runs each image through DINOv2-ViT-L/14 (no augmentation; resize 256 -> center crop 224 -> normalize).
  3. Saves features and the aligned image-path list and noisy-label list to a
     single .pth file consumable by TNDC_mod_labels_mllm.py.

Output:
  saved_features/clothing1m_features_no_aug.pth
    {
      'features':     Tensor[N=64000, D=1024],
      'image_paths':  list[str],   # aligned with `features`
      'noisy_labels': Tensor[N],   # aligned with `features`
    }

Usage:
    python extract_clothing1m_features.py \
        --root /home/zfj/dataset/clothing1M \
        --num_samples 64000 \
        --output saved_features/clothing1m_features_no_aug.pth \
        --batch_size 64 \
        --seed 42

Approx wall-clock on a single RTX 3090: ~25 minutes.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------------
# Subset construction (mirrors dataloader_clothing1M.py mode='all' behaviour)
# ----------------------------------------------------------------------------
def build_balanced_subset(root: str, num_samples: int, num_class: int = 14):
    """Return (image_paths, noisy_labels) — class-balanced subset, in the same
    order as DivideMix's `clothing_dataset(mode='all', num_samples=...)`.

    The original dataloader iterates `noisy_train_key_list.txt` once and keeps
    the first `num_samples/14` images of each class encountered. We replicate
    this exactly (no shuffle).
    """
    # noisy labels for ALL ~1M training images
    train_labels = {}
    with open(os.path.join(root, 'noisy_label_kv.txt'), 'r') as f:
        for l in f.read().splitlines():
            entry = l.split()
            img_path = os.path.join(root, entry[0][7:])  # strip "images/" prefix
            train_labels[img_path] = int(entry[1])

    # the noisy train key list (~1M entries)
    train_imgs = []
    with open(os.path.join(root, 'noisy_train_key_list.txt'), 'r') as f:
        for l in f.read().splitlines():
            train_imgs.append(os.path.join(root, l[7:]))

    # apply per-class quota
    per_class_quota = num_samples / num_class
    class_count = torch.zeros(num_class)
    selected_paths, selected_labels = [], []
    for impath in train_imgs:
        if impath not in train_labels:
            continue
        label = train_labels[impath]
        if class_count[label] < per_class_quota and len(selected_paths) < num_samples:
            selected_paths.append(impath)
            selected_labels.append(label)
            class_count[label] += 1

    print(f"[subset] loaded {len(selected_paths)} samples")
    print(f"[subset] per-class counts: {[int(c) for c in class_count]}")
    return selected_paths, selected_labels


# ----------------------------------------------------------------------------
# Feature dataset (no augmentation; matches the test-time transform of
# dataloader_clothing1M's `transform_test`)
# ----------------------------------------------------------------------------
class FeatureDataset(Dataset):
    """Yields (image_tensor, index) so that we can write features back in
    order. We use the no-augmentation, deterministic preprocessing pipeline
    so that the saved features are reproducible across runs."""
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img), idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, required=True,
                   help='Clothing1M root directory.')
    p.add_argument('--num_samples', type=int, default=64000,
                   help='Total subset size (must be divisible by 14).')
    p.add_argument('--num_class', type=int, default=14)
    p.add_argument('--output', type=str,
                   default='saved_features/clothing1m_features_no_aug.pth')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dinov2_repo', type=str,
                   default='/mnt/zfj/projects/dinov2-main',
                   help='Local DINOv2 repo (torch.hub source=local).')
    p.add_argument('--dinov2_model', type=str,
                   default='dinov2_vitl14_reg',
                   help='DINOv2 model name (default: dinov2_vitl14_reg, '
                        'matches the rest of the project).')
    p.add_argument('--dinov2_weights', type=str,
                   default='/mnt/zfj/dataset/models/dinov2_vitl14_reg4_pretrain.pth',
                   help='Path to the .pth weights file for the model above.')
    args = p.parse_args()

    # --- reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"[setup] device: {device}")

    # --- 1. Build 64K subset ---
    print(f"\n[step 1] building {args.num_samples}-sample class-balanced subset ...")
    image_paths, noisy_labels = build_balanced_subset(
        args.root, args.num_samples, args.num_class
    )
    if len(image_paths) != args.num_samples:
        print(f"[WARN] requested {args.num_samples} but got {len(image_paths)}. "
              f"This can happen if the per-class quota is not satisfiable — "
              f"check that the corpus contains enough images per class.")

    # --- 2. Load DINOv2 ---
    print(f"\n[step 2] loading DINOv2 from local repo: {args.dinov2_repo}")
    if not os.path.isdir(args.dinov2_repo):
        raise FileNotFoundError(
            f"DINOv2 repo not found: {args.dinov2_repo}\n"
            f"Pass --dinov2_repo <path> to point to your local clone."
        )
    if not os.path.isfile(args.dinov2_weights):
        raise FileNotFoundError(
            f"DINOv2 weights not found: {args.dinov2_weights}\n"
            f"Pass --dinov2_weights <path> to point to the .pth file."
        )
    t_load = time.perf_counter()
    model = torch.hub.load(
        args.dinov2_repo, args.dinov2_model,
        source='local', pretrained=False,
    )
    state = torch.load(args.dinov2_weights, map_location='cpu')
    model.load_state_dict(state)
    model = model.eval().to(device)
    print(f"[step 2] {args.dinov2_model} loaded in {time.perf_counter()-t_load:.1f} s")

    # --- 3. Build dataloader ---
    # No augmentation: deterministic preprocessing matching dataloader_clothing1M.transform_test
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214),
        ),
    ])
    dataset = FeatureDataset(image_paths, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # --- 4. Extract features ---
    N = len(image_paths)
    print(f"\n[step 3] extracting features for {N} images "
          f"in {len(loader)} batches of {args.batch_size} ...")

    # Pre-allocate feature tensor; D inferred from a single forward pass.
    feature_dim = None
    features = None

    t_start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (imgs, indices) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                feats = model(imgs)            # [B, D]
            feats = feats.float()

            if features is None:
                feature_dim = feats.shape[1]
                features = torch.zeros(N, feature_dim, dtype=torch.float32)
                print(f"[step 3] feature dim: {feature_dim}")

            features[indices] = feats.cpu()

            if (batch_idx + 1) % max(1, len(loader) // 20) == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (batch_idx + 1) * (len(loader) - batch_idx - 1)
                print(f"  batch {batch_idx+1:>5d} / {len(loader):>5d}  "
                      f"({100*(batch_idx+1)/len(loader):5.1f}%)   "
                      f"elapsed = {elapsed:6.1f} s   eta = {eta:6.1f} s")

    t_total = time.perf_counter() - t_start
    print(f"\n[step 3] feature extraction: {t_total:.1f} s ({t_total/60:.2f} min)")

    # --- 5. Save ---
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'features':     features,
        'image_paths':  image_paths,
        'noisy_labels': torch.tensor(noisy_labels, dtype=torch.long),
        'meta': {
            'dataset':     'clothing1m',
            'subset_size': len(image_paths),
            'num_class':   args.num_class,
            'feature_dim': feature_dim,
            'seed':        args.seed,
            'extract_seconds': float(t_total),
        },
    }, args.output)
    print(f"\n[save] -> {args.output}")
    print(f"[save] size: {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
