#!/usr/bin/env python3
"""
ParagonSR Model Fusion Utility (Safetensors Edition)
Author: Philip Hofmann

Description:
This script loads a trained ParagonSR checkpoint (in .safetensors format),
applies the permanent fusion logic, and saves the new, simplified state_dict
back into the .safetensors format for consistency.

Run with python -m scripts.fuse_model
"""

import torch

# --- CHANGE 1 of 2: Import both load_file and save_file ---
from safetensors.torch import load_file, save_file

# This import must be absolute to work when run as a module
from traiNNer.archs.paragonsr_arch import paragonsr_s

# --- Configuration ---
# Point this to the .safetensors checkpoint you want to fuse.
TRAINING_CHECKPOINT_PATH = (
    "experiments/4x_ParagonSR_S/models/net_g_ema_18077.safetensors"
)

# --- CHANGE 2 of 2: Update the output filename extension ---
# Define where to save the new, fused model, now as a .safetensors file.
FUSED_MODEL_PATH = "release_models/4x_ParagonSR_S_fused.safetensors"

# --- Main Fusion Logic ---
if __name__ == "__main__":
    print("--- Starting ParagonSR Model Fusion ---")

    # 1. Initialize the model structure
    model = paragonsr_s(scale=4)
    print(f"Initialized '{model.__class__.__name__}' architecture.")

    # 2. Load the trained weights from the .safetensors file
    print(f"Loading training weights from: {TRAINING_CHECKPOINT_PATH}")
    state_dict = load_file(TRAINING_CHECKPOINT_PATH)
    model.load_state_dict(state_dict, strict=True)

    # 3. Switch the model to evaluation mode
    model.eval()
    print("Model switched to evaluation mode.")

    # 4. Call the permanent fusion method
    model.fuse_for_release()
    print("Fusion complete. The model architecture has been permanently simplified.")

    # 5. Save the new, fused state_dict as a .safetensors file
    # This is a simple replacement for torch.save()
    save_file(model.state_dict(), FUSED_MODEL_PATH)
    print(f"\nSuccessfully saved fused model to: {FUSED_MODEL_PATH}")
