#!/usr/bin/env python3
"""
ParagonSR Model Fusion Utility (Safetensors Edition)

Author: Philip Hofmann

Description:
    This script loads a trained ParagonSR checkpoint (in .safetensors format),
    applies the permanent fusion logic defined in the ParagonSR architecture,
    and saves a new, simplified (fused) checkpoint as .safetensors.

Updated behavior:
    - Accepts CLI arguments instead of hardcoded paths.
    - Validates:
        * Input path exists.
        * Input file has `.safetensors` extension.
        * Input checkpoint appears to be from a NON-FUSED training model.
    - Accepts an output directory path where the fused file will be written.
        * The fused filename is derived from the input, appending `_fused`
          before the extension:
              input:  /path/to/4xParagonSR_S_pretrain.safetensors
              output: {out_dir}/4xParagonSR_S_pretrain_fused.safetensors
        * Creates the output directory if it does not exist.
    - Verifies:
        * The fused file is successfully written.
        * The fused model state_dict structurally matches the fused architecture.

Usage examples:

    # Basic usage: write fused file to a target directory
    python -m scripts.paragonsr.fuse_model \\
        --checkpoint models/4xParagonSR_S_pretrain.safetensors \\
        --out-dir release_models \\
        --variant s \\
        --scale 4

    # Using an absolute checkpoint path
    python -m scripts.paragonsr.fuse_model \\
        --checkpoint /abs/path/to/4xParagonSR_S_pretrain.safetensors \\
        --out-dir /abs/path/to/release_models \\
        --variant s \\
        --scale 4

Notes:

    - Supports ParagonSR variants: nano, tiny, xs, s, m, l, xl.
    - You must pass the correct --variant and --scale that match the checkpoint.
"""

import argparse
import os
import sys
from typing import Tuple

import torch
from safetensors.torch import load_file, save_file

# Absolute import so running as module works reliably
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
        description="Fuse a ParagonSR training checkpoint (.safetensors) for deployment."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the input training checkpoint (.safetensors, non-fused).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help=(
            "Directory where the fused checkpoint will be saved. "
            "Will be created if it does not exist."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="s",
        choices=["nano", "tiny", "xs", "s", "m", "l", "xl"],
        help=(
            "ParagonSR variant used for this checkpoint. "
            "Determines width/depth when reconstructing the model for fusion. "
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
    return parser.parse_args()


def validate_input_checkpoint(path: str) -> None:
    # Existence
    if not os.path.isfile(path):
        print(f"ERROR: Input checkpoint does not exist: {path}")
        sys.exit(1)

    # Extension
    if not path.lower().endswith(".safetensors"):
        print(
            f"ERROR: Input checkpoint must be a .safetensors file, got: {os.path.basename(path)}"
        )
        sys.exit(1)

    # Simple heuristic: discourage obviously fused inputs
    # If the filename already suggests fusion, warn and require explicit decision.
    lower_name = os.path.basename(path).lower()
    if "_fused" in lower_name:
        print(
            f"ERROR: The input checkpoint name suggests it is already fused: {path}\n"
            "Refusing to fuse an already-fused model. Please provide a training (non-fused) checkpoint."
        )
        sys.exit(1)


def ensure_output_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Failed to create output directory '{path}': {e}")
        sys.exit(1)


def derive_fused_path(input_checkpoint: str, out_dir: str) -> str:
    base = os.path.basename(input_checkpoint)
    name, ext = os.path.splitext(base)
    fused_name = f"{name}_fused{ext}"
    return os.path.join(out_dir, fused_name)


def load_training_state(checkpoint_path: str) -> dict:
    try:
        state_dict = load_file(checkpoint_path)
    except Exception as e:
        print(f"ERROR: Failed to load safetensors checkpoint '{checkpoint_path}': {e}")
        sys.exit(1)

    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        print(
            "ERROR: Loaded checkpoint is empty or invalid. "
            "Ensure this is a valid training .safetensors file."
        )
        sys.exit(1)

    return state_dict


def appears_unfused_paragonsr_state(state_dict: dict) -> bool:
    """
    Heuristic check: try to detect if this looks like a NON-fused ParagonSR checkpoint.

    Rationale:
      - A non-fused training checkpoint should contain parameters for ReparamConvV2
        branches (e.g., conv3x3, conv1x1, possibly dw_conv3x3) inside blocks.
      - A fully fused checkpoint (produced by fuse_for_release) would instead have
        only simple Conv2d weights where those modules were.

    Strategy:
      - Look for keys that strongly indicate presence of unfused ReparamConvV2:
          "body.0.blocks.0.transformer.spatial_mixer.conv3x3.weight"
        or similar patterns.
      - If such keys exist, we consider it "appears non-fused".
      - If we only see generic conv weights and no branch-specific keys,
        we assume it might be fused already.

    This is intentionally conservative; if unsure, we allow proceed but warn.
    """
    has_reparam_branch = any(
        ".spatial_mixer.conv3x3.weight" in k or ".spatial_mixer.conv1x1.weight" in k
        for k in state_dict.keys()
    )
    return has_reparam_branch


def get_paragonsr_model(variant: str, scale: int) -> torch.nn.Module:
    """
    Factory for ParagonSR variants.

    Variant must be one of: nano, tiny, xs, s, m, l, xl.
    """
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


def verify_fused_state(variant: str, scale: int, fused_state: dict) -> tuple[bool, str]:
    """
    Verify that:
      - The fused state_dict can be loaded into a freshly constructed fused model
        of the same variant/scale, after applying fuse_for_release().
      - Shape compatibility holds.

    This ensures we didn't produce a broken checkpoint.
    """
    test_model = get_paragonsr_model(variant=variant, scale=scale)
    # Apply fusion on the fresh model so its structure matches the fused checkpoint
    test_model.fuse_for_release()
    test_model.eval()

    try:
        missing, unexpected = test_model.load_state_dict(fused_state, strict=False)
    except Exception as e:
        return (
            False,
            f"Loading fused state into fused ParagonSR-{variant.upper()} (x{scale}) failed: {e}",
        )

    if missing:
        return (
            False,
            f"Fused checkpoint missing keys for fused architecture: {sorted(missing)[:10]}...",
        )

    # Unexpected keys are allowed only if they are strictly non-parameter metadata.
    unexpected_param_keys = [
        k for k in unexpected if not k.endswith((".num_batches_tracked",))
    ]
    if unexpected_param_keys:
        return (
            False,
            f"Fused checkpoint has unexpected parameter keys: {sorted(unexpected_param_keys)[:10]}...",
        )

    return (
        True,
        f"Fused state matches fused ParagonSR-{variant.upper()} (x{scale}) architecture.",
    )


def main() -> None:
    args = parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir)
    variant = args.variant
    scale = args.scale

    print("--- Starting ParagonSR Model Fusion ---")
    print(f"Input checkpoint: {checkpoint_path}")
    print(f"Output directory: {out_dir}")
    print(f"Network variant: {variant}")
    print(f"Scale: x{scale}")

    # 1. Validate and prepare paths
    validate_input_checkpoint(checkpoint_path)
    ensure_output_dir(out_dir)
    fused_path = derive_fused_path(checkpoint_path, out_dir)
    print(f"Fused checkpoint will be saved as: {fused_path}")

    # 2. Load the training (non-fused) state_dict
    state_dict = load_training_state(checkpoint_path)

    # 3. Sanity-check that it appears to be non-fused
    if not appears_unfused_paragonsr_state(state_dict):
        print(
            "WARNING: Checkpoint does not clearly appear to be a non-fused ParagonSR training state.\n"
            "It may already be fused or use an unexpected structure.\n"
            "Proceeding anyway, but resulting fused checkpoint may be invalid."
        )

    # 4. Initialize the model structure for the requested variant/scale
    try:
        model = get_paragonsr_model(variant=variant, scale=scale)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(
        f"Initialized '{model.__class__.__name__}' architecture "
        f"for variant='{variant}', scale=x{scale}."
    )

    # 5. Load the training weights
    print("Loading training weights into model...")
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint into ParagonSR-S model: {e}")
        sys.exit(1)

    # 6. Switch to evaluation mode
    model.eval()
    print("Model switched to evaluation mode.")

    # 7. Apply permanent fusion
    print("Fusing model for release...")
    model.fuse_for_release()
    print("Fusion complete. The model architecture has been permanently simplified.")

    # 8. Extract fused state_dict and verify against fused architecture
    fused_state = model.state_dict()
    ok, msg = verify_fused_state(variant=variant, scale=scale, fused_state=fused_state)
    if ok:
        print(f"Verification success: {msg}")
    else:
        print(f"ERROR during verification: {msg}")
        sys.exit(1)

    # 9. Save fused state_dict as .safetensors
    try:
        save_file(fused_state, fused_path)
    except Exception as e:
        print(f"ERROR: Failed to save fused checkpoint to '{fused_path}': {e}")
        sys.exit(1)

    # 10. Final existence check
    if not os.path.isfile(fused_path):
        print(
            "ERROR: Fused checkpoint file was not found after saving. "
            "Please check filesystem permissions."
        )
        sys.exit(1)

    print(f"\nSuccessfully saved fused model to: {fused_path}")
    print("--- ParagonSR Model Fusion Completed Successfully ---")


if __name__ == "__main__":
    main()
