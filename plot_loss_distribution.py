import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def parse_args():
    parser = argparse.ArgumentParser(description='Plot loss distributions')
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to .npz file saved by train_and_save_losses.py')
    parser.add_argument('--output_dir', type=str, default='./exp_results/loss_plots',
                        help='where to save the figures')
    parser.add_argument('--num_classes', type=int, default=10)
    return parser.parse_args()


def plot_overall_distribution(clean_avg, noisy_avg, save_dir):
    """整体干净 vs 噪声样本的平均损失分布"""
    plt.figure(figsize=(10, 6))
    plt.hist(clean_avg, bins=50, alpha=0.6, density=True, label='Clean', color='#2E86AB')
    plt.hist(noisy_avg, bins=50, alpha=0.6, density=True, label='Noisy', color='#A23B72')
    plt.axvline(np.mean(clean_avg), color='#2E86AB', ls='--', lw=2, label=f'Clean mean: {np.mean(clean_avg):.3f}')
    plt.axvline(np.mean(noisy_avg), color='#A23B72', ls='--', lw=2, label=f'Noisy mean: {np.mean(noisy_avg):.3f}')
    plt.xlabel('Average Loss per Sample', fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.title('Overall Average Loss Distribution', fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_clean_vs_noisy.png'), dpi=300)
    plt.close()


def plot_by_epoch_distribution(epoch_clean, epoch_noisy, save_dir, max_show=10):
    """选几个epoch画干净/噪声损失分布对比"""
    epochs = sorted(epoch_clean.keys())
    if len(epochs) > max_show:
        step = len(epochs) // (max_show - 1)
        selected = [epochs[i*step] for i in range(max_show-1)] + [epochs[-1]]
    else:
        selected = epochs

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Loss Distribution by Epoch (Clean vs Noisy)', fontsize=16)

    for ax, ep in zip(axes.flat, selected):
        ax.hist(epoch_clean[ep], bins=40, alpha=0.6, density=True, color='#2E86AB', label='Clean')
        ax.hist(epoch_noisy[ep], bins=40, alpha=0.6, density=True, color='#A23B72', label='Noisy')
        ax.set_title(f'Epoch {ep}', fontsize=11)
        ax.grid(alpha=0.3)
        if ax == axes.flat[0]:
            ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'epoch_by_epoch_distribution.png'), dpi=300)
    plt.close()


def plot_by_noise_class(clean_label, noise_label, sample_losses, save_dir, num_classes=10):
    """按噪声标签类别分组，画干净/噪声样本的平均损失分布"""
    clean_label = np.asarray(clean_label)
    noise_label = np.asarray(noise_label)

    class_grouped = defaultdict(lambda: {'clean': [], 'noisy': []})

    for idx, losses in sample_losses.items():
        avg_loss = np.mean(losses)
        n_label = noise_label[idx]
        c_label = clean_label[idx]

        if n_label == c_label:
            class_grouped[n_label]['clean'].append(avg_loss)
        else:
            class_grouped[n_label]['noisy'].append(avg_loss)

    for cls in range(num_classes):
        if cls not in class_grouped:
            continue

        c_losses = class_grouped[cls]['clean']
        n_losses = class_grouped[cls]['noisy']

        if not c_losses and not n_losses:
            continue

        plt.figure(figsize=(5, 4))
        if c_losses:
            plt.hist(c_losses, bins=30, alpha=0.6, density=True, color='#2E86AB', label=f'Clean (n={len(c_losses)})')
        if n_losses:
            plt.hist(n_losses, bins=30, alpha=0.6, density=True, color='#A23B72', label=f'Noisy (n={len(n_losses)})')

        plt.title(f'Noise Label Class = {cls}', fontsize=14)
        plt.xlabel('Average Loss', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'noise_class_{cls}_loss_dist.png'), dpi=300)
        plt.close()


def plot_loss_by_noise_class_and_true_class(
    clean_label,
    noise_label,
    sample_losses,
    save_dir,
    num_classes=10,
    top_k=3
    ):
    """
    为每个噪声标签类别，绘制其中样本数量占比前 top_k 个真实类别的损失分布，
    剩余类别合并为 "Others"，使用适合学术论文的配色方案
    """
    clean_label = np.asarray(clean_label)
    noise_label = np.asarray(noise_label)

    # 1. 统计每个噪声类中各个真实类的样本数量
    noise_class_stats = defaultdict(lambda: Counter())
    for idx, (n_lbl, c_lbl) in enumerate(zip(noise_label, clean_label)):
        noise_class_stats[n_lbl][c_lbl] += 1

    # 2. 按噪声类分组收集每个真实类的平均损失列表
    noise_class_grouped_losses = defaultdict(lambda: defaultdict(list))
    for idx, losses in sample_losses.items():
        if len(losses) == 0:
            continue
        avg_loss = np.mean(losses)
        n_lbl = noise_label[idx]
        c_lbl = clean_label[idx]
        noise_class_grouped_losses[n_lbl][c_lbl].append(avg_loss)

    # 学术论文常用、色盲友好的配色方案（基于 ColorBrewer / Okabe-Ito）
    colors = {
        0: '#2166ac',    # 深蓝
        1: '#fdae61',    # 橙色
        2: '#67a9cf',    # 绿色
        'others': '#d73027'  # 紫色（偏深），用于 Others
    }



    for noise_cls in range(num_classes):
        if noise_cls not in noise_class_grouped_losses:
            continue

        true_class_losses = noise_class_grouped_losses[noise_cls]
        total_samples = sum(noise_class_stats[noise_cls].values())

        if total_samples == 0:
            continue

        # 按样本数量从大到小排序所有真实类
        sorted_true_classes = sorted(
            noise_class_stats[noise_cls].items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 取前 top_k 个真实类
        top_classes = [cls for cls, cnt in sorted_true_classes[:top_k]]
        # 其余的类合并为 Others
        other_classes = [cls for cls, cnt in sorted_true_classes[top_k:]]

        # 准备绘图数据：top_k 类 + Others
        plot_data = []
        labels = []
        used_colors = []

        # 先加 top_k 类
        for i, true_cls in enumerate(top_classes):
            losses_list = true_class_losses[true_cls]
            if not losses_list:
                continue
            count = noise_class_stats[noise_cls][true_cls]
            percentage = count / total_samples * 100
            label_str = f'True Label {true_cls}: {percentage:.1f}%'

            plot_data.append(losses_list)
            labels.append(label_str)
            used_colors.append(colors.get(i, colors[0]))  # 循环使用前3色

        # 再加 Others（如果有剩余类）
        other_losses = []
        other_count = 0
        for cls in other_classes:
            other_losses.extend(true_class_losses[cls])
            other_count += noise_class_stats[noise_cls][cls]

        if other_losses and other_count > 0:
            percentage = other_count / total_samples * 100
            label_str = f'Others: {percentage:.1f}%'
            plot_data.append(other_losses)
            labels.append(label_str)
            used_colors.append(colors['others'])

        # 如果没有任何数据可画，跳过
        if not plot_data:
            continue

        # 开始绘图
        plt.figure(figsize=(5, 3))
        # plt.title(f'Loss Distribution in Noisy Class: {noise_cls}',
                #   fontsize=14, pad=12)

        # 绘制所有组的直方图
        for data, label, color in zip(plot_data, labels, used_colors):
            plt.hist(
                data,
                bins=20,
                alpha=0.75,
                density=True,
                label=label,
                color=color,
                edgecolor='none',
                linewidth=0.4
            )

        plt.xlabel('Average Loss per Sample', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)

        # 图例放在右侧外部，字体稍小一点
        plt.legend(
            fontsize=9,
            loc='upper right',          # 改成 upper right / upper left / lower right 等
            frameon=True,               # 保留边框
            framealpha=0.5,             # 稍微透明一点，更现代
            edgecolor='gray',
            facecolor='white',          # 建议显式指定背景色（防深色主题）
            borderpad=0.6,              # 内部留白
            labelspacing=0.4            # 行间距稍紧凑
        )

        

        plt.tight_layout()

        # 定义基础文件名（不带扩展名）
        base_filename = f'noise_cls_{noise_cls}_top{top_k}_true_loss_dist'

        # 保存为高分辨率 PNG
        png_path = os.path.join(save_dir, f'{base_filename}.png')
        plt.savefig(png_path, dpi=350, bbox_inches='tight')

        # 保存为矢量 PDF（推荐用于论文插入）
        pdf_path = os.path.join(save_dir, f'{base_filename}.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)  # pdf 的 dpi 通常设 300 就够

        plt.close()

        print(f"Saved PNG : {os.path.basename(png_path)}")
        print(f"Saved PDF : {os.path.basename(pdf_path)}")



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading loss data...")
    data = np.load(args.data_path, allow_pickle=True)

    clean_avg = data['clean_avg_losses']
    noisy_avg = data['noisy_avg_losses']
    epoch_clean = data['epoch_clean_loss'].item()
    epoch_noisy = data['epoch_noisy_loss'].item()
    clean_label = data['clean_label']
    noise_label = data['noise_label']
    sample_losses = data['sample_losses'].item()  # dict

    # print("Plotting overall distribution...")
    # plot_overall_distribution(clean_avg, noisy_avg, args.output_dir)

    # print("Plotting epoch-by-epoch distribution...")
    # plot_by_epoch_distribution(epoch_clean, epoch_noisy, args.output_dir)

    # print("Plotting loss distribution by noise class...")
    # plot_by_noise_class(clean_label, noise_label, sample_losses, args.output_dir, args.num_classes)

    print("Plotting loss distribution by noise class & true class...")
    plot_loss_by_noise_class_and_true_class(
        clean_label=clean_label,
        noise_label=noise_label,
        sample_losses=sample_losses,
        save_dir=args.output_dir,
        num_classes=args.num_classes
    )

    print(f"\n所有图像已保存至：{args.output_dir}")


if __name__ == '__main__':
    main()