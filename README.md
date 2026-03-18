<p align="center">
   <img width="2444" height="572" alt="image" src="https://github.com/user-attachments/assets/d97580e0-9db7-4f6a-81ec-4f49bf51a9fd" />
</p>



<h1 align="center">TNDC: Towards Reliable Label Pre-Correction via MLLM-Assisted Semantic Prototype Construction</h1>


**Abstract**: Label pre-correction is vital for noisy label learning as it mitigates the negative impact of noise during model warm-up. However, existing pre-correction methods often overlook intra-class noise distributions, leading to biased semantic prototypes and suboptimal performance. To address this, we propose TNDC, a novel pre-correction framework that leverages Multimodal Large Language Models (MLLMs) for robust semantic prototype construction. Unlike prior arts, TNDC introduces an intra-class denoising step: samples are first clustered within each class using unsupervised features to decouple noise, after which an MLLM identifies credible subclusters to form highly reliable prototypes. Furthermore, we introduce confidence-aware label correction to avoid erroneous modification of hard noise labels, along with a minimum distance threshold to filter out open-set noise, excluding irrelevant features from training. Extensive experiments on CIFAR, WebVision, and Clothing1M demonstrate that TNDC achieves state-of-the-art pre-correction accuracy and significantly boosts downstream noise-robust learning. Remarkably, under the extreme setting of 80\% instance-dependent noise, TNDC boosts the accuracy of DLD by 91.70\% on CIFAR-10 and 76.81\% on CIFAR-100. Code is available at https://github.com/fungizhang/TNDC.

## The Table of Contents


:wink: If TNDC is helpful to you, please star this repo. Thanks! :hugs: 
- [:grimacing: Dependencies and installation](#grimacing-dependencies-and-installation)
- [:partying\_face: How to run](#partying_face-how-to-run)
- [:evergreen\_tree: Detail of our method](#evergreen_tree-detail-of-twin-sight)
- [:smiley: Citation](#smiley-citation)
- [:phone: Contact](#phone-contact)

##  :grimacing: Dependencies and installation
Before running or modifying the code, you need to:
- Make sure Anaconda or Miniconda is installed.
- Clone this repo to your machine.
  
  ```
  # git clone this repository
  git clone https://github.com/fungizhang/TNDC.git
  cd TNDC

  # create new anaconda env 
  conda create -n TNDC python=3.10 -y
  conda activate TNDC
  ```

- required packages in `requirements.txt`
  ```
  # install python dependencies
  Install whatever is missing.
  ```


## 🥳 How to Run:

We provide a complete pipeline for **TNDC (Label Pre-correction)**. Follow the steps below:

### 1. Train and Save Loss Data

Train the initial model to collect loss statistics for noise identification.

```bash
python train_and_save_losses.py --output_dir ./exp_results/loss_analysis1

```

### 2. Plot Loss Distribution

Visualize the loss distribution to analyze clean vs. noisy data.

```bash
python plot_loss_distribution.py --data_path ./exp_results/loss_analysis1/losses_data.npz --output_dir ./exp_results/loss_analysis1/plots

```

### 3. Label Pre-correction

First, extract images for MLLM (Multimodal Large Language Model) inference:

> **Note**: Update `dataset_name` (cifar10/cifar100) and `root_dir` in `extract_cifar_images.py` before running.

```bash
python ./extract_cifar_images.py

```

Then, perform MLLM-assisted label pre-correction:

```bash
python ./TNDC_mod_labels_mllm.py --dataset_name cifar10 --noise_mode idn --noise_ratio 0.2

```

### 4. Downstream Tasks with TNDC Plugin

Compare standard training with TNDC-enhanced training across different frameworks:

**CIFAR-10 (CE):**

```bash
python ./1_CE/train_cifar.py --data_name cifar10 --epoch 50 --noise_mode sym --noise_ratio 0.2 --gpu 5
python ./1_CE/train_cifar_tndc.py --data_name cifar10 --epoch 50 --noise_mode sym --noise_ratio 0.2 --gpu 5

```

**CIFAR-100 (CE):**

```bash
python ./1_CE/train_cifar.py --data_name cifar100 --epoch 200 --noise_mode sym --noise_ratio 0.2 --gpu 5
python ./1_CE/train_cifar_tndc.py --data_name cifar100 --epoch 200 --noise_mode sym --noise_ratio 0.2 --gpu 5

```

**DLD Framework:**

```bash
python ./5_DLD/train_on_CIFAR_runable.py --noise_type cifar10-sym-0.2 --nepoch 50 --device cuda:5
python ./5_DLD/train_on_CIFAR_tndc.py --noise_type cifar10-sym-0.2 --nepoch 50 --device cuda:5

```

---

## 🌲 Detail of Our Method:



---

## 😃 Citation

If our work is useful for your research, please consider citing:

```bibtex

```

---

## 📞 Contact

If you have any questions, please feel free to reach out at `fungizhang@gmail.com`.





