

# 🖼️ CLIP Minimal Scratch Implementation (`clip_scratch.py`)

This repository contains a self-contained, minimal implementation of the **Contrastive Language-Image Pre-training (CLIP)** model built from scratch using PyTorch. The goal is to demonstrate the core architecture—a Vision Transformer (ViT) for images and a standard Transformer for text—along with the symmetric InfoNCE contrastive learning objective.

The implementation uses **CIFAR-10** with **synthetic captions** for a runnable demonstration, allowing users to quickly see the mechanism in action.

## 💡 Overview: What is CLIP?

CLIP is a neural network trained on a wide variety of image-text pairs. It learns a **multimodal embedding space** where the embeddings of an image and its corresponding caption are close to each other, while being far from unrelated image and text embeddings.

This implementation focuses on two key components:

1.  **Image Encoder (`ViTEncoder`):** A simplified Vision Transformer.
2.  **Text Encoder (`TextEncoder`):** A simplified standard Transformer.
3.  **Contrastive Objective:** The `CLIPModel` and `clip_loss` that align the outputs of the two encoders.

-----

## 🚀 Key Features

  * **ViT Architecture:** Implements a basic **Vision Transformer** with `PatchEmbed`, **CLS token**, and **positional embeddings**.
  * **Simple Tokenizer:** Includes a minimal `SimpleTokenizer` for text processing, handling common special tokens (`<bos>`, `<eos>`, `<pad>`).
  * **Transformer Blocks:** Features a common `TransformerBlock` with multi-head attention and an MLP.
  * **Learnable Logit Scale:** Incorporates the `logit_scale` parameter, a crucial element in the original CLIP model for controlling the temperature of the contrastive loss.
  * **Symmetric InfoNCE Loss:** Implementation of the **Contrastive Loss** (`clip_loss`) that trains the model to maximize the agreement between true pairs.

-----

## 🛠️ Installation and Requirements

The project uses common PyTorch libraries.

### Prerequisites

  * Python 3.7+
  * PyTorch
  * Torchvision

### Setup

```bash
# Clone the repository (if this code is part of a larger repo)
# git clone https://github.com/YourUsername/YourRepo.git
# cd YourRepo

# Install necessary libraries
pip install torch torchvision
```

-----

## 💻 Usage: Running the Demo

The file `clip_scratch.py` contains a self-contained training function that downloads the CIFAR-10 dataset and uses simple, label-based captions (e.g., "a photo of a cat") to run a mock training loop.

### Starting the Training

To run the demo and train the model for 3 epochs:

```bash
python clip_scratch.py
```

### Expected Output

The script will print the training loss per epoch and save the final weights:

```
Using device: cuda (or cpu)
Epoch 1 | loss 1.2345 | time 45.3s
Epoch 2 | loss 1.0500 | time 45.1s
Epoch 3 | loss 0.9876 | time 44.9s
Saved clip_scratch.pth
```

### Key Hyperparameters (in `train_clip_demo`)

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `epochs` | 3 | Number of training iterations. |
| `batch_size` | 256 | Batch size used for data loading. |
| `embed_dim` | 256 | The dimension of the final **joint embedding space** (the projection head output). |
| `ViT/Text Dim` | 512 | The internal dimension of the Transformer blocks. |

-----

## 🧠 Code Architecture Highlights

The codebase is modular, built around the fundamental components of a Transformer-based multimodal model.

### 1\. Encoders (`ViTEncoder`, `TextEncoder`)

Both encoders share the basic `TransformerBlock` and project their features to the same dimensional space.

  * **`ViTEncoder`**
      * Takes input shape: $B \times 3 \times 224 \times 224$ (Batch, Channels, Height, Width).
      * Uses **`PatchEmbed`** to convert image to sequence of patches.
      * Output: Pooled CLS token feature vector ($B \times 512$).
  * **`TextEncoder`**
      * Takes input shape: $B \times L$ (Batch, Max Length).
      * Uses **`token_emb`** and **`pos_emb`**.
      * Output: Pooled `<bos>` token feature vector ($B \times 512$).

### 2\. CLIP Model (`CLIPModel`)

The central component that connects the two modalities.

  * It initializes two **`ProjectionHead`** modules (`img_proj` and `txt_proj`) to map the $512$ dimensional features down to the $256$ dimensional **joint embedding space**.
  * It applies **L2 normalization** (`l2norm`) to the projected image and text embeddings.
  * It computes the similarity matrix: $\text{Similarity} = e^{\tau} \cdot (\text{ImageEmbeddings} \cdot \text{TextEmbeddings}^\intercal)$, where $\tau$ is the learnable $\text{logit\_scale}$.

### 3\. Loss Function (`clip_loss`)

```python
def clip_loss(logits_per_image):
    # Target is the identity matrix (perfect match on the diagonal)
    B = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)     # Image-to-Text loss
    loss_t = F.cross_entropy(logits_per_image.t(), labels) # Text-to-Image loss
    return (loss_i + loss_t) / 2
```

This function implements the **symmetric InfoNCE loss** by treating the diagonal of the similarity matrix as the positive matches and all other entries in the row/column as negative matches for a standard cross-entropy calculation.

-----

