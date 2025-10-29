# ParagonSR

**A High-Performance, Reparameterizable CNN-Hybrid for Super-Resolution and Restoration**  
*An Architecture by Philip Hofmann*

---

## 1. Introduction & Philosophy

ParagonSR is a state-of-the-art, general-purpose super-resolution architecture designed for a superior balance of **peak quality, training efficiency, and inference speed**. It represents a synergistic blend of the most effective and efficient ideas from a multitude of modern SISR models.

The core philosophy behind ParagonSR is that of an **"Optimized Hybrid CNN,"** engineered to achieve the perceptual power and deep feature understanding of a Transformer, but with the efficiency, stability, and deployment speed of a highly optimized Convolutional Neural Network.

Its design is a direct evolution of ideas first explored in a previous, unreleased prototype architecture, **HyperionSR**.

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

3.  **Advanced Reparameterization (The "Afterburner"):** The core convolutional unit, `ReparamConvV3`, is inspired by the powerful design of SpanPP. It fuses three distinct and powerful convolutional branches with learnable weights, dramatically increasing the model's expressive power during training with a negligible impact on the final, fused inference speed.

## 3. A Battle-Hardened Design: Engineered for Stability

Deep, reparameterizable architectures can be prone to instability during long training runs. ParagonSR was specifically engineered to solve these challenges through a process of real-world testing and refinement:

-   **`LayerScale` Integration:** Each ParagonBlock includes a `LayerScale` module, a powerful stabilization technique from modern Transformer designs. It forces the model to learn in a more controlled manner, preventing the "exploding gradient" (`NaN` loss) issues that can plague deep networks.
-   **Stateless Fusion for EMA:** The architecture uses a stateless, "on-the-fly" fusion method during training-time validations. This design is the result of rigorous testing and is **guaranteed to be compatible with EMA**, permanently fixing the state-synchronization bugs that can corrupt a model's weights over time.

## 4. Training Dynamics & Recommendations

ParagonSR is a **powerhouse architecture**. Its deep and non-linear blocks give it immense representational capacity, but this also means it can generate large intermediate values and is more sensitive than simpler networks. Successful training requires a deliberate and stable strategy.

### Evidence of Robustness: The "PSNR Dip"
During its initial pre-training, the model experienced a temporary dip in performance around the 96,000 iteration mark. This was not a bug, but a sign of the model navigating a complex loss landscape.

-   **What Happened:** The model encountered a difficult batch of data, and the optimizer took a single, suboptimal step.
-   **Why It's a Good Thing:** A lesser or unstable architecture might have failed catastrophically. Instead, ParagonSR's built-in stability features (`LayerScale`, `grad_clip`, and the `bf16` training format) successfully contained the instability. The model **immediately self-corrected** on subsequent batches and went on to achieve a new best performance. This dip and recovery is a testament to the architecture's resilience.

### Recommendations for Fine-Tuning
This pre-train is an excellent foundation. However, when fine-tuning with more complex and potentially unstable losses (like GAN, LPIPS, DISTS, etc.), a more conservative approach is **strongly recommended** to ensure stability:

-   **Low Learning Rate:** Start with a small learning rate, such as `1e-5` to `5e-5`.
-   **Gradient Clipping:** Keep `grad_clip: 1.0` (or a similar value) enabled. It is an essential safety net.
-   **Use a Warm-up:** A warm-up period of `warmup_iter: 5000` is highly recommended to allow the optimizer to adapt to new loss functions.
-   **Adjust Optimizer Betas:** For GAN training, consider using a lower momentum value, e.g., `betas: [0.8, 0.99]`, to make the optimizer more responsive.

## 5. The ParagonSR Family: A Model for Every Need

ParagonSR comes in a variety of sizes, allowing users to choose the perfect balance of quality and performance for their hardware and use case.

| Variant | Feature Dim | Depth (#Groups x #Blocks) | Training Target (VRAM) | Inference Target (VRAM) | Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `paragonsr_tiny`| 32 | 3 x 3 (9) | ~4-6GB | **Any GPU/CPU** | **Real-Time Video**, Previews |
| `paragonsr_xs` | 48 | 4 x 4 (16) | ~6-8GB | ~4-6GB | Low-End Hardware, Fast Images |
| `paragonsr_s` | 64 | 6 x 6 (36) | **~12GB** | ~6-8GB | **Flagship Model**, High Quality |
| `paragonsr_m` | 96 | 8 x 8 (64) | ~16-24GB | ~8-12GB | Prosumer Quality |
| `paragonsr_l` | 128 | 10 x 10 (100)| >24GB | ~12GB+ | Enthusiast/SOTA Quality |
| `paragonsr_xl` | 160 | 12 x 12 (144)| 48GB+ | >24GB | Research/Benchmark Chasing |

## 6. Installation & Setup

This project is designed for the **[traiNNer-redux](https://github.com/the-database/traiNNer-redux)** framework.

### Step 1: Place Project Files
Place the release files in the following locations within your `traiNNer-redux` project:

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
└── ...
```

### Step 2: Install Dependencies
It is highly recommended to use a Python virtual environment.
```sh
# Install all required packages for training and export
pip install torch torchvision safetensors onnx onnxconverter-common onnxscript

# NOTE: For GPU support, ensure you install the correct PyTorch version for your CUDA toolkit.
```

## 7. Training & Deployment

Please see the [release page for the pre-trained model](link-to-your-pretrain-release) for a detailed training template (`.yml`) and instructions.

A key feature of ParagonSR is its ability to be permanently fused for deployment. After training, run `fuse_model.py` and then `export_onnx.py` to create a final, high-speed, portable model for real-world applications.

## 8. Acknowledgements & Inspirations

This architecture stands on the shoulders of giants and would not be possible without the incredible research and open-source contributions of the community.

-   **Architectural Ancestor:** `HyperionSR` (unreleased predecessor)
-   **Primary Inspirations:** `SpanPP`, `MoSRv2`, `RTMoSR`, `GaterV3`, `FDAT`, `HAT`.
-   **Core Techniques:**
    -   **Structural Reparameterization:** "RepVGG: Making VGG-style ConvNets Great Again" (Ding et al., 2021).
    -   **Gated Linear Units:** "GLU Variants Improve Transformer" (Shazeer, 2020).
    -   **Hierarchical Design & LayerScale:** "Swin Transformer" (Liu et al., 2021) and "Going deeper with Image Transformers" (Touvron et al., 2021).
    -   **Inception & Depthwise Convolutions:** "Going Deeper with Convolutions" (Szegedy et al., 2014) and "MobileNets" (Howard et al., 2017).
