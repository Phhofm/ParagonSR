#!/usr/bin/env python3
"""
ParagonSR ONNX Export Utility (Argument-Driven, Variant-Aware)

Author: Philip Hofmann

Description:
    This script loads a PERMANENTLY FUSED ParagonSR checkpoint (produced by
    scripts/paragonsr/fuse_model.py) and exports it to both FP32 and FP16 ONNX
    formats in a robust, configurable way.

Key features:
    - Accepts CLI arguments:
        * --checkpoint: fused .safetensors file (required)
        * --variant: ParagonSR variant (nano, tiny, xs, s, m, l, xl)
        * --scale: upscale factor (e.g., 2, 4)
        * --out-dir: directory for ONNX exports
        * --opset: ONNX opset version (default: 18)
    - Validates:
        * Fused checkpoint path and extension.
        * That it "looks like" a fused checkpoint (heuristic).
        * Output directory existence (creates if necessary).
    - Constructs the correct ParagonSR architecture for the given variant/scale,
      applies fuse_for_release() to ensure structure compatibility, and then
      loads the fused weights.
    - Exports:
        * FP32 ONNX with filename suffix: _op{opset}_fp32.onnx
        * FP16 ONNX with filename suffix: _op{opset}_fp16.onnx
      based on the input fused checkpoint filename.

Notes:
    - Default opset is set to 18 to align with Mish and modern operator support.
    - This script assumes the input checkpoint is already fused; do NOT pass
      raw training checkpoints here. Use fuse_model.py first.

Example:
    python -m scripts.paragonsr.export_onnx \
        --checkpoint release_models/4xParagonSR_S_pretrain_fused.safetensors \
        --variant s \
        --scale 4 \
        --out-dir release_models \
        --opset 18
"""

import argparse
import os
import sys
from typing import Tuple

import onnx
import torch
from onnxconverter_common import convert_float_to_float16
from safetensors.torch import load_file
from traiNNer.archs.paragonsr_arch import (
    paragonsr_l,
    paragonsr_m,
    paragonsr_nano,
    paragonsr_s,
    paragonsr_tiny,
    paragonsr_xl,
    paragonsr_xs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a fused ParagonSR checkpoint (.safetensors) to FP32 and FP16 ONNX.\n"
            "Use scripts/paragonsr/fuse_model.py first to create the fused checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the fused ParagonSR checkpoint (.safetensors, already fused).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="s",
        choices=["nano", "tiny", "xs", "s", "m", "l", "xl"],
        help=(
            "ParagonSR variant for this fused checkpoint. "
            "Must match how the model was constructed during training/fusion. "
            "Default: s"
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help=(
            "Upscale factor of the model (e.g., 2 or 4). "
            "Must match the training configuration of the checkpoint. "
            "Default: 4"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="release_models",
        help=(
            "Directory where ONNX files will be saved. "
            "Will be created if it does not exist. Default: release_models"
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help=(
            "ONNX opset version to use for export. "
            "Default: 18 (recommended for Mish and modern runtimes)."
        ),
    )
    return parser.parse_args()


def ensure_output_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Failed to create output directory '{path}': {e}")
        sys.exit(1)


def validate_fused_checkpoint(path: str) -> None:
    if not os.path.isfile(path):
        print(f"ERROR: Fused checkpoint does not exist: {path}")
        sys.exit(1)

    if not path.lower().endswith(".safetensors"):
        print(
            f"ERROR: Fused checkpoint must be a .safetensors file, got: {os.path.basename(path)}"
        )
        sys.exit(1)

    lower_name = os.path.basename(path).lower()
    if "_fused" not in lower_name:
        print(
            f"WARNING: The input checkpoint name does not contain '_fused': {path}\n"
            "This script is intended for fused checkpoints. "
            "Ensure you ran fuse_model.py beforehand."
        )


def appears_fused_paragonsr_state(state_dict: dict) -> bool:
    """
    Heuristic: check that the state_dict looks like it came from a fused ParagonSR.

    A fused ParagonSR (after fuse_for_release) should:
      - NOT contain branch-specific keys like '.spatial_mixer.conv3x3.weight'
      - Primarily contain standard Conv2d weights where reparam blocks were.

    If we still see obvious reparam branch weights, it likely is NOT fused.
    """
    has_reparam_branch = any(
        ".spatial_mixer.conv3x3.weight" in k or ".spatial_mixer.conv1x1.weight" in k
        for k in state_dict.keys()
    )
    return not has_reparam_branch


def get_paragonsr_model(variant: str, scale: int) -> torch.nn.Module:
    variant = variant.lower()
    if variant == "nano":
        return paragonsr_nano(scale=scale)
    if variant == "tiny":
        return paragonsr_tiny(scale=scale)
    if variant == "xs":
        return paragonsr_xs(scale=scale)
    if variant == "s":
        return paragonsr_s(scale=scale)
    if variant == "m":
        return paragonsr_m(scale=scale)
    if variant == "l":
        return paragonsr_l(scale=scale)
    if variant == "xl":
        return paragonsr_xl(scale=scale)
    raise ValueError(f"Unsupported ParagonSR variant: {variant}")


def derive_onnx_paths(
    fused_checkpoint: str, out_dir: str, opset: int
) -> tuple[str, str]:
    base = os.path.basename(fused_checkpoint)
    name, _ = os.path.splitext(base)
    # Ensure we clearly encode opset and precision in filenames.
    fp32_name = f"{name}_op{opset}_fp32.onnx"
    fp16_name = f"{name}_op{opset}_fp16.onnx"
    return os.path.join(out_dir, fp32_name), os.path.join(out_dir, fp16_name)


def export_fp32_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    opset: int,
) -> None:
    print(f"\n--- Exporting to FP32 ONNX (opset={opset}): {onnx_path} ---")

    dummy_input = torch.randn(1, 3, 64, 64)
    input_names = ["input"]
    output_names = ["output"]

    try:
        # Use traditional dynamic_axes here for compatibility; with opset 18 and
        # your current stack this is stable, and avoids the torch.export
        # dynamic_shapes validation error.
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            opset_version=opset,
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"},
            },
        )
    except Exception as e:
        print(f"ERROR: FP32 ONNX export failed: {e}")
        sys.exit(1)

    if not os.path.isfile(onnx_path):
        print("ERROR: FP32 ONNX file not found after export.")
        sys.exit(1)

    print("FP32 ONNX export successful.")


def export_fp16_onnx(onnx_fp32_path: str, onnx_fp16_path: str) -> None:
    print(f"\n--- Converting to FP16 ONNX: {onnx_fp16_path} ---")
    try:
        fp32_model = onnx.load(onnx_fp32_path)
        fp16_model = convert_float_to_float16(fp32_model)
        onnx.save(fp16_model, onnx_fp16_path)
    except Exception as e:
        print(
            "\nWARNING: FP16 conversion failed. This is sometimes due to ONNX/onnxruntime "
            "version mismatches or unsupported patterns."
        )
        print(f"Error details: {e}")
        print("You still have the valid FP32 ONNX model.")
        return

    if not os.path.isfile(onnx_fp16_path):
        print(
            "WARNING: FP16 ONNX file not found after conversion. "
            "Conversion may have silently failed."
        )
        return

    print("FP16 ONNX conversion successful.")


def main() -> None:
    args = parse_args()

    fused_ckpt_path = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    variant = args.variant
    scale = args.scale
    opset = args.opset

    print("--- Starting ParagonSR ONNX Export ---")
    print(f"Fused checkpoint: {fused_ckpt_path}")
    print(f"Output directory: {out_dir}")
    print(f"Network variant: {variant}")
    print(f"Scale: x{scale}")
    print(f"Opset version: {opset}")

    # 1. Validate paths and create output dir
    validate_fused_checkpoint(fused_ckpt_path)
    ensure_output_dir(out_dir)

    # 2. Load fused state_dict
    try:
        state_dict = load_file(fused_ckpt_path)
    except Exception as e:
        print(f"ERROR: Failed to load fused safetensors checkpoint: {e}")
        sys.exit(1)

    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        print("ERROR: Loaded fused checkpoint is empty or invalid.")
        sys.exit(1)

    # 3. Heuristic check that it's fused
    if not appears_fused_paragonsr_state(state_dict):
        print(
            "WARNING: Checkpoint appears to contain unfused reparameterization branches.\n"
            "It may not be a true fused checkpoint. Export may not reflect the intended "
            "deployment architecture."
        )

    # 4. Build model for variant/scale and prepare for fused weights
    try:
        model = get_paragonsr_model(variant=variant, scale=scale)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(
        f"Initialized '{model.__class__.__name__}' for variant='{variant}', scale=x{scale}."
    )

    # Ensure architecture is fused structurally before loading fused weights.
    # This matches the structure expected by the fused checkpoint.
    model.fuse_for_release()
    model.eval()
    print("Model structure has been fused and set to eval mode.")

    # 5. Load fused weights
    print("Loading fused weights into fused-structure model...")
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"ERROR: Failed to load fused checkpoint into model: {e}")
        sys.exit(1)

    if missing:
        print(
            "WARNING: Some expected keys were missing when loading fused weights: "
            f"{sorted(missing)[:10]}..."
        )
    if unexpected:
        print(
            "WARNING: Some unexpected keys were present in fused weights: "
            f"{sorted(unexpected)[:10]}..."
        )

    # 6. Derive ONNX output paths (with opset + precision encoded)
    onnx_fp32_path, onnx_fp16_path = derive_onnx_paths(
        fused_checkpoint=fused_ckpt_path,
        out_dir=out_dir,
        opset=opset,
    )

    # 7. Export FP32
    export_fp32_onnx(model=model, onnx_path=onnx_fp32_path, opset=opset)

    # 8. Export FP16
    export_fp16_onnx(onnx_fp32_path=onnx_fp32_path, onnx_fp16_path=onnx_fp16_path)

    print("\n--- ONNX Export Complete! ---")
    print(f"FP32 ONNX: {onnx_fp32_path}")
    print(f"FP16 ONNX: {onnx_fp16_path} (if conversion reported successful)")


if __name__ == "__main__":
    main()
