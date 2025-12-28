# TNDC
A label pre-correction method


python ./1_CE/train_cifar.py --data_name cifar10 --epoch 50 --noise_mode sym --noise_ratio 0.2 --gpu 5
python ./1_CE/train_cifar_tndc.py --data_name cifar10 --epoch 50 --noise_mode sym --noise_ratio 0.2 --gpu 5

python ./1_CE/train_cifar.py --data_name cifar100 --epoch 200 --noise_mode sym --noise_ratio 0.2 --gpu 5
python ./1_CE/train_cifar_tndc.py --data_name cifar100 --epoch 200 --noise_mode sym --noise_ratio 0.2 --gpu 5


python ./5_DLD/train_on_CIFAR_runable.py --noise_type cifar10-sym-0.2 --nepoch 50 --device cuda:5
python ./5_DLD/train_on_CIFAR_tndc.py --noise_type cifar10-sym-0.2 --nepoch 50 --device cuda:5
