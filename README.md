# VAE Training for CelebA

A **Variational Autoencoder (VAE)** implementation in PyTorch for generating and reconstructing face images from the CelebA dataset.
This repository provides a clean, modular pipeline with β‑annealing, checkpointing, and sample generation.

---

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training from Python](#training-from-python)
  - [Training from YAML config](#training-from-yaml-config)
  - [Custom Dataset Path](#custom-dataset-path)
- [Configuration Parameters](#configuration-parameters)
- [Outputs](#outputs)
- [Training Results (Proof of Concept)](#training-results-proof-of-concept)
- [Model Architecture](#model-architecture)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## ✨ Features
- **β‑VAE** loss with KL annealing for better disentanglement.
- **Dynamic encoder/decoder** – automatically adapts to image size and hidden dimensions.
- **Dataset caching** – specify a custom download location for the CelebA dataset.
- **Checkpointing** – saves the best model based on validation loss.
- **Reconstruction samples** – saved every epoch for visual monitoring.
- **Final generation** – produces new images from random latent vectors.
- **YAML support** – easily change hyperparameters without touching code.

---

## 📁 Project Structure
```
.
├── Train.py                    # Main training script
├── configs/
│   └── standard_config.yaml    # Example configuration file
├── src/
│   ├── model/
│   │   ├── VAE.py              # VAE model with loss function
│   │   ├── Encoder.py          # Encoder module
│   │   └── Decoder.py          # Decoder module
│   └── preprocess/
│       └── Preprocess.py       # Dataset handling and data loaders
├── checkpoints/                # Saved model checkpoints (auto‑created during training)
├── outputs/samples/            # Reconstruction and generation samples (auto‑created during training)
└── dataset/                    # Default dataset cache directory (auto‑created during dataset download)
└── LICENSE                     # MIT license
└── README.md
```

---

## 📦 Requirements
- Python 3.8+
- PyTorch ≥ 1.9.0
- torchvision
- datasets (Hugging Face)
- Pillow
- PyYAML
- matplotlib (optional, for `show_images` utility)

---

## ⚙️ Installation

1. **Clone the repository**.
```bash
https://github.com/TryingManV2/Vanilla-VAE-CelebA-Face-Generator.git
```
or
```bash
git@github.com:TryingManV2/Vanilla-VAE-CelebA-Face-Generator.git
```
2. **Install the required packages**:
   ```bash
   pip install torch torchvision datasets pillow pyyaml matplotlib
   ```

## 🚀 Usage

### Training from Python (I do not recommend this.)
```python
from src.Train import train

train(
    dataset="flwrlabs/celeba",
    cache_dir="./dataset",
    n_epochs=100,
    batch_size=32,
    img_size=128,
    latent_dim=128,
    beta=0.0005,
    learning_rate=1e-3
)
```

### Training from YAML config (My recommendation!)
Create a YAML file (e.g., `configs/my_config.yaml`) with any of the training parameters:
```yaml
dataset: "flwrlabs/celeba"
cache_dir: ./dataset
n_epochs: 100
img_size: 128
batch_size: 32
num_workers: 4
pin_memory: True
shuffle: True
test_set: False
checkpoint_dir: "./checkpoints"
sample_dir: "./outputs/samples"
input_channels: 3
latent_dim: 128
hidden_dims: [32, 64, 128, 256, 512]
learning_rate: 1e-3
beta: 0.0005
```
Then run:
```python
from src.Train import train_from_yaml
train_from_yaml("configs/my_config.yaml")
```

### Custom Dataset Path
The dataset will be downloaded to the directory specified by `cache_dir` (default: `"./dataset"`).
If you already have the dataset cached elsewhere, point `cache_dir` to that location to reuse it.

---

## ⚙️ Configuration Parameters

| Parameter          | Type     | Default                | Description |
|--------------------|----------|------------------------|-------------|
| `dataset`          | str      | `"flwrlabs/celeba"`    | Hugging Face dataset identifier. |
| `cache_dir`        | str      | `"./dataset"`          | Local directory to store the dataset. |
| `n_epochs`         | int      | `100`                  | Number of training epochs. |
| `img_size`         | int      | `128`                  | Input image size (square). |
| `batch_size`       | int      | `32`                   | Batch size for training and validation. |
| `num_workers`      | int      | `4`                    | DataLoader worker threads. |
| `pin_memory`       | bool     | `True`                 | Enables faster GPU data transfer. |
| `shuffle`          | bool     | `True`                 | Shuffle training data. |
| `test_set`         | bool     | `False`                | Whether to load the test set (not used in training). |
| `checkpoint_dir`   | str      | `"./checkpoints"`      | Where to save the best model. |
| `sample_dir`       | str      | `"./outputs/samples"`  | Where to save reconstruction & generated images. |
| `input_channels`   | int      | `3`                    | Number of image channels (RGB). |
| `latent_dim`       | int      | `128`                  | Dimensionality of the latent space. |
| `hidden_dims`      | list     | `[32,64,128,256,512]`  | Channels in each encoder/decoder layer. |
| `learning_rate`    | float    | `1e-3`                 | Adam learning rate. |
| `beta`             | float    | `0.0005`               | KL divergence weight (β‑VAE). Annealed during first 20 epochs. |

---

## 📂 Outputs
- **Checkpoints**: `./checkpoints/best_model.pt` – the model state dict with the lowest validation loss.
- **Reconstruction grids**: `./outputs/samples/recon_epoch_XXX.png` – top row original, bottom row reconstructed.
- **Final generation**: `./outputs/samples/generated_final.png` – 16 new faces sampled from the prior.

---

## 📊 Training Results (Proof of Concept)

Due to hardware constraints, this model was trained for **only 1 epoch** on a **Google T4 GPU** as a PoC.
With this limited training, the network begins to learn basic facial structures but does not produce high‑quality reconstructions or realistic generations yet.

**Observed losses after 1 epoch**:
Train  -> Loss: 0.0508  Recon: 0.0437  KL: 282.6719
Valid  -> Loss: 0.1604  Recon: 0.0304  KL: 259.8838

If you are wondering why the reconstruction loss is lower on the validation set, this is because Dropout is turned off and BatchNorm uses its running statistics during evaluation, producing a cleaner and more stable forward pass. The raw KL loss can also be lower during validation because the encoder produces slightly different latent distribution parameters when these regularization layers are disabled. However, the total validation loss is higher because our validation β is fixed at a larger value than the early training β, placing much greater weight on the KL term.

For meaningful results, we recommend training for **at least 50–100 epochs**.
The current implementation is fully functional and ready for longer runs on more powerful hardware.

*Sample reconstructions and generated images from the 1‑epoch run are available in the `outputs/samples/` folder.*

---

## 🧠 Model Architecture
- **Encoder**: 5 convolutional layers (stride 2) with BatchNorm and LeakyReLU, followed by two linear layers for `mu` and `log_var`.
- **Decoder**: Linear projection to encoder’s final spatial size, then 5 transposed convolutional layers (stride 2) with BatchNorm and LeakyReLU, and a final Tanh activation.
- **Loss**: MSE reconstruction loss + β × KL divergence (with annealing).

---

## 🛠️ Troubleshooting

| Issue | Possible Solution |
|-------|-------------------|
| **CUDA out of memory** | Reduce `batch_size` or `img_size`. |
| **Dataset download slow** | Set `cache_dir` to a fast local drive. |
| **Device not found** | The script automatically detects CUDA, MPS, or CPU. Ensure PyTorch is installed with proper CUDA support. |
| **NaN losses** | Try lowering `learning_rate` or increasing `beta`. |

---

## 📄 License
This project is open‑source and available under the MIT License.

---