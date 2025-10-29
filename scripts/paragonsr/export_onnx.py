#!/usr/bin/env python3
"""
ParagonSR ONNX Export Utility
Author: Philip Hofmann

Description:
This script loads the PERMANENTLY FUSED ParagonSR checkpoint (from fuse_model.py)
and exports it to both FP32 and FP16 ONNX formats.
"""

import onnx
import torch
from onnxconverter_common import convert_float_to_float16
from safetensors.torch import load_file
from traiNNer.archs.paragonsr_arch import paragonsr_s

# --- Configuration ---
# This must match the output path from your fuse_model.py script
FUSED_MODEL_PATH = "release_models/4x_ParagonSR_S_fused.safetensors"

# Output paths for the final ONNX files
ONNX_FP32_PATH = "release_models/4x_ParagonSR_S_fp32.onnx"
ONNX_FP16_PATH = "release_models/4x_ParagonSR_S_fp16.onnx"

# --- Main Export Logic ---
if __name__ == "__main__":
    print("--- Starting ParagonSR ONNX Export ---")

    # 1. Initialize the model structure
    model = paragonsr_s(scale=4)
    print(f"Initialized '{model.__class__.__name__}' architecture.")

    # 2. CRITICAL: We must fuse the STRUCTURE before loading the fused WEIGHTS.
    # The state_dict we are about to load only has keys for the fused_conv,
    # so the model must match that structure.
    model.fuse_for_release()
    model.eval()
    print("Model structure has been prepared for fused weights.")

    # 3. Load the FUSED weights (.pth file, so standard torch.load is correct)
    print(f"Loading fused weights from: {FUSED_MODEL_PATH}")
    state_dict = load_file(FUSED_MODEL_PATH)
    model.load_state_dict(state_dict)

    # 4. Export to FP32 ONNX
    print(f"\n--- Exporting to FP32 ONNX: {ONNX_FP32_PATH} ---")
    # Create a dummy input. The size doesn't matter much due to dynamic axes,
    # but 64x64 is a good, fast default.
    dummy_input = torch.randn(1, 3, 64, 64)
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_FP32_PATH,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=17,  # Modern opset for best compatibility
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    print("Export successful.")

    # 5. Convert to FP16 ONNX
    print(f"\n--- Converting to FP16 ONNX: {ONNX_FP16_PATH} ---")
    try:
        fp32_model = onnx.load(ONNX_FP32_PATH)
        fp16_model = convert_float_to_float16(fp32_model)
        onnx.save(fp16_model, ONNX_FP16_PATH)
        print("Conversion successful.")
    except Exception as e:
        print(
            "\nWARNING: FP16 conversion failed. This is sometimes due to ONNX version mismatches."
        )
        print(f"Error details: {e}")
        print("You still have the valid FP32 model.")

    print("\n--- ONNX Export Complete! ---")
