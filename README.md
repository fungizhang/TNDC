# TNDC
A label pre-correction method

进入项目
# 第一步：训练并保存损失数据
python train_and_save_losses.py --output_dir ./exp_results/loss_analysis1 --epoch 120 --noise_ratio 0.5

# 第二步：画各种分布图
python plot_loss_distribution.py --data_path ./exp_results/loss_analysis1/losses_data.npz --output_dir ./exp_results/loss_analysis1/plots

# 第三步：标签预修正
python ./TNDC_mod_labels_mllm.py --dataset_name cifar10 --noise_mode idn --noise_ratio 0.2

# 第四步：使用TNDC插件在下游任务

python ./1_CE/train_cifar.py --data_name cifar10 --epoch 50 --noise_mode sym --noise_ratio 0.2 --gpu 5
python ./1_CE/train_cifar_tndc.py --data_name cifar10 --epoch 50 --noise_mode sym --noise_ratio 0.2 --gpu 5

python ./1_CE/train_cifar.py --data_name cifar100 --epoch 200 --noise_mode sym --noise_ratio 0.2 --gpu 5
python ./1_CE/train_cifar_tndc.py --data_name cifar100 --epoch 200 --noise_mode sym --noise_ratio 0.2 --gpu 5


python ./5_DLD/train_on_CIFAR_runable.py --noise_type cifar10-sym-0.2 --nepoch 50 --device cuda:5
python ./5_DLD/train_on_CIFAR_tndc.py --noise_type cifar10-sym-0.2 --nepoch 50 --device cuda:5

......

