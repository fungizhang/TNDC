"""
Identify Noise-Dominant Classes (NDC) in a noisy dataset.

A class c is NDC iff (Definition 1 in the paper):
    n_{c,c}  <  max_{k != c}  n_{c,k}
where n_{c,k} = number of samples with noisy_label=c and true_label=k.

Output: a JSON file with the list of NDC class indices, plus per-class
statistics for inspection.

Usage:
    python identify_ndc_classes.py \
        --dataset_name cifar100 --noise_mode idn --noise_ratio 0.6 \
        --output ndc_classes_c100_idn06.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_name', type=str, default='cifar100',
                   choices=['cifar10', 'cifar100'])
    p.add_argument('--noise_mode', type=str, default='idn',
                   choices=['sym', 'asym', 'idn', 'asym_var'])
    p.add_argument('--noise_ratio', type=float, default=0.6)
    p.add_argument('--output', type=str, required=True,
                   help='Output JSON file path.')
    p.add_argument('--seed', type=int, default=42,
                   help='Used to ensure same noisy labels as TNDC pipeline.')
    args = p.parse_args()

    # --- random seed for reproducible noise ---
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- load CIFAR with noise via the same dataloader TNDC uses ---
    from dataloader import dataloader_cifar as dataloader

    path_map = {
        'cifar10':  '../../../dataset/cifar-10-batches-py',
        'cifar100': '../../../dataset/cifar-100-python',
    }
    loader = dataloader.cifar_dataloader(
        args.dataset_name,
        noise_mode=args.noise_mode,
        noise_ratio=args.noise_ratio,
        batch_size=64, num_workers=4,
        root_dir=path_map[args.dataset_name],
        model='dino',
    )
    train_loader = loader.run('train')
    noise_label = np.array(train_loader.dataset.noise_label)
    clean_label = np.array(train_loader.dataset.clean_label)
    num_class = 10 if args.dataset_name == 'cifar10' else 100

    print(f"[data] {args.dataset_name} {args.noise_mode}-{args.noise_ratio} | "
          f"{len(noise_label)} samples | {num_class} classes")

    # --- compute per-noisy-class true-label distribution ---
    per_class_dist = {c: defaultdict(int) for c in range(num_class)}
    for n, t in zip(noise_label, clean_label):
        per_class_dist[int(n)][int(t)] += 1

    ndc_classes = []
    ndc_details = []
    for c in range(num_class):
        dist = per_class_dist[c]
        if not dist:
            continue
        n_cc = dist.get(c, 0)              # samples whose true label == noisy label
        # max_{k != c} n_{c,k}
        other = {k: v for k, v in dist.items() if k != c}
        if not other:
            continue
        max_k = max(other, key=other.get)
        n_max = other[max_k]
        is_ndc = n_cc < n_max
        if is_ndc:
            ndc_classes.append(c)
        ndc_details.append({
            'noisy_class': c,
            'total_samples': sum(dist.values()),
            'n_correct': n_cc,
            'dominant_other_class': max_k,
            'dominant_other_count': n_max,
            'is_ndc': is_ndc,
        })

    # --- compute the NDC-rich subset indices ---
    ndc_indices = [int(i) for i, n in enumerate(noise_label) if int(n) in ndc_classes]

    # --- summary ---
    print(f"\n=== NDC classes ({len(ndc_classes)} / {num_class}) ===")
    for d in ndc_details:
        if d['is_ndc']:
            print(f"  class {d['noisy_class']:3d}: "
                  f"correct={d['n_correct']:4d}/{d['total_samples']:4d}, "
                  f"dominant other class={d['dominant_other_class']:3d} "
                  f"(count={d['dominant_other_count']:4d})")
    print(f"\nNDC subset size: {len(ndc_indices)} samples "
          f"({100*len(ndc_indices)/len(noise_label):.1f}% of total)")

    # --- save ---
    out = {
        'dataset': args.dataset_name,
        'noise_mode': args.noise_mode,
        'noise_ratio': args.noise_ratio,
        'seed': args.seed,
        'num_total_classes': num_class,
        'num_ndc_classes': len(ndc_classes),
        'ndc_classes': ndc_classes,
        'ndc_subset_indices': ndc_indices,
        'ndc_subset_size': len(ndc_indices),
        'total_samples': int(len(noise_label)),
        'per_class_details': ndc_details,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] -> {args.output}")


if __name__ == '__main__':
    main()
