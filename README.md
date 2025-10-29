# ParagonSR: A High-Performance Super-Resolution Architecture

**Author:** Philip Hofmann
**Primary Development:** This architecture was developed in a collaborative process with advanced large language models, synthesizing and refining ideas from state-of-the-art research.

## 1. Introduction & Philosophy

ParagonSR is a state-of-the-art, general-purpose super-resolution architecture designed for a superior balance of **peak quality, training efficiency, and inference speed**. It represents a synergistic blend of the most effective and efficient ideas from a multitude of modern SISR models.

The core philosophy behind ParagonSR is that of an **"Optimized Hybrid CNN,"** engineered to achieve the perceptual power and deep feature understanding of a Transformer, but with the efficiency, stability, and deployment speed of a highly optimized Convolutional Neural Network.

### Strengths & Design Goals

ParagonSR was designed from the ground up to excel in three key areas:

1.  **High-Quality, Realistic Output:** By combining powerful context-gathering and feature-transformation modules, the architecture excels at learning the complex textures and structures necessary for photorealistic image restoration. It is particularly well-suited for reversing the complex, real-world degradations found in modern datasets.

2.  **Exceptional Inference Speed:** The architecture is built on a foundation of **reparameterization**. After training, its complex, multi-branch structure can be mathematically fused into a simple, ultra-fast network, making it ideal for real-world applications, ONNX export, and further acceleration with runtimes like NVIDIA's TensorRT.

3.  **Proven Training Stability:** Every component, from the normalization layers to the fusion logic, has been battle-tested and engineered to ensure a robust and stable training experience, even with advanced framework features like Exponential Moving Average (EMA).

## 2. Core Architectural Innovations

ParagonSR's performance is derived from the synergy of its core components, which are synthesized from the best ideas in recent computer vision research.

### The ParagonBlock: A Synergistic Core

The heart of the network is the `ParagonBlock`, a novel block designed to maximize performance and efficiency:

1.  **Efficient Multi-Scale Context (The "Eyes"):** Instead of a single large kernel, the ParagonBlock uses an **Inception-style Depthwise Convolution** (`InceptionDWConv2d`). This module captures features at multiple spatial scales (square, horizontal, and vertical) simultaneously, providing a rich, multi-scale understanding of the image with high parameter efficiency.

2.  **Powerful Gated Transformation (The "Brain"):** The features are then processed by a **Gated Feed-Forward Network** (`GatedFFN`). This is a powerful, non-linear feature transformer inspired by modern language models. Its gating mechanism allows the network to dynamically route and modulate information, a key technique for learning the complex, non-linear mappings required for high-fidelity restoration.

3.  **Advanced Reparameterization (The "Afterburner"):** The core convolutional unit within the Gated-FFN, `ReparamConvV3`, is inspired by the powerful design of SpanPP. It fuses three distinct and powerful convolutional branches with learnable weights, dramatically increasing the model's expressive power during training with a negligible impact on the final, fused inference speed.

### A Battle-Hardened Design: Engineered for Stability

Deep, reparameterizable architectures can be prone to instability during long training runs, especially when using advanced techniques like EMA. ParagonSR was specifically engineered to solve these challenges:

-   **`LayerScale` Integration:** Each ParagonBlock includes a `LayerScale` module, a powerful stabilization technique from modern Transformer designs. It forces the model to learn in a more controlled manner, preventing the "exploding gradient" (`NaN` loss) issues that can plague deep networks.
-   **Stateless Fusion for EMA:** The architecture uses a stateless, "on-the-fly" fusion method during training-time validations. This design is the result of rigorous testing and is **guaranteed to be compatible with EMA**, permanently fixing the state-synchronization bugs that can corrupt a model's weights over time.

## 3. The ParagonSR Family: A Model for Every Need

ParagonSR comes in a variety of sizes, allowing users to choose the perfect balance of quality and performance for their hardware and use case.

| Variant | Feature Dim | Depth (#Groups x #Blocks) | Training Target (VRAM) | Inference Target (VRAM) | Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `paragonsr_tiny`| 32 | 3 x 3 (9) | ~4-6GB | **Any GPU/CPU** | **Real-Time Video**, Previews |
| `paragonsr_xs` | 48 | 4 x 4 (16) | ~6-8GB | ~4-6GB | Low-End Hardware, Fast Images |
| `paragonsr_s` | 64 | 6 x 6 (36) | **~12GB** | ~6-8GB | **Flagship Model**, High Quality |
| `paragonsr_m` | 96 | 8 x 8 (64) | ~16-24GB | ~8-12GB | Prosumer Quality |
| `paragonsr_l` | 128 | 10 x 10 (100)| >24GB | ~12GB+ | Enthusiast/SOTA Quality |
| `paragonsr_xl` | 160 | 12 x 12 (144)| 48GB+ | >24GB | Research/Benchmark Chasing |

## 4. Installation & Setup

This project is designed for the **[traiNNer-redux](https://github.com/the-database/traiNNer-redux)** framework.

### Step 1: Place Project Files
Download the release files and place them in the following locations within your `traiNNer-redux` project folder:

```
traiNNer-redux/
│
├── scripts/
│   └── paragonsr/      <-- CREATE THIS FOLDER
│       ├── fuse_model.py
│       └── export_onnx.py
│
├── traiNNer/
│   └── archs/
│       └── ParagonSR_arch.py
│
├── options/
│   └── templates/
│       └── ParagonSR/    <-- CREATE THIS FOLDER
│           └── ParagonSR_fidelity.yml
│
└── train.py
```

### Step 2: Install Dependencies
It is highly recommended to use a Python virtual environment.
```sh
# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all required packages for training and export
pip install torch torchvision safetensors onnx onnxconverter-common onnxscript

# NOTE: For GPU support, ensure you install the correct PyTorch version for your CUDA toolkit.
# Visit https://pytorch.org/get-started/locally/ to get the specific command.
```

## 5. Training

To train a `ParagonSR` model, use the provided template.

1.  **Create a new config file** by copying the template:
    `cp options/templates/ParagonSR/ParagonSR_fidelity.yml options/train/My_ParagonSR_Training.yml`
2.  **Edit the new config file** (`My_ParagonSR_Training.yml`) to point to your datasets.
3.  **Start the training run:**
    ```sh
    python train.py -opt options/train/My_ParagonSR_Training.yml
    ```

## 6. Deployment: Fusing & ONNX Export

A key feature of ParagonSR is its ability to be permanently fused for a significant inference speed-up. This is a **required step** for creating a release model.

### Step 1: Fuse the Trained Model
This script loads a training checkpoint and saves a new, permanently fused model.

1.  **Edit `scripts/paragonsr/fuse_model.py`** to point `TRAINING_CHECKPOINT_PATH` to your desired `.safetensors` file (e.g., `net_g_ema_latest.safetensors`).
2.  Run the script from the **root** of your project:
    ```sh
    python -m scripts.paragonsr.fuse_model
    ```
This will create a new, fast model (e.g., `release_models/4x_ParagonSR_S_fused.safetensors`).

### Step 2: Export the Fused Model to ONNX
This script takes the fused model and converts it to FP32 and FP16 ONNX formats.

1.  **Edit `scripts/paragonsr/export_onnx.py`** to ensure `FUSED_MODEL_PATH` matches the output from the previous step.
2.  Run the script from the **root** of your project:
    ```sh
    python -m scripts.paragonsr.export_onnx
    ```
The resulting `.onnx` files are dynamic, portable, and ready for high-speed inference in applications like ChaiNNer.

## 7. Acknowledgements & Inspirations

This architecture stands on the shoulders of giants and would not be possible without the incredible research and open-source contributions of the community.

-   **Architectural Inspirations:** `SpanPP`, `HyperionSR`, `MoSRv2`, `RTMoSR`, `GaterV3`, `FDAT`, `HAT`.
-   **Core Techniques:**
    -   **Structural Reparameterization:** "RepVGG: Making VGG-style ConvNets Great Again" (Ding et al., 2021).
    -   **Gated Linear Units:** "GLU Variants Improve Transformer" (Shazeer, 2020).
    -   **Hierarchical Design & LayerScale:** "Swin Transformer" (Liu et al., 2021) and "Going deeper with Image Transformers" (Touvron et al., 2021).
    -   **Inception & Depthwise Convolutions:** "Going Deeper with Convolutions" (Szegedy et al., 2014) and "MobileNets" (Howard et al., 2017).
