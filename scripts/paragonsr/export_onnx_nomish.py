#!/usr/bin/env python3
"""
ParagonSR ONNX "NoMish" FP16 Compatibility Export

Author: Philip Hofmann

Description:
    Create a Mish-free, FP16, opset-17-targeted ONNX model from an existing
    fused FP32 ONNX, for maximum compatibility (TensorRT, DirectML, older ORT).

Key behavior:
    - Input:
        * A fused FP32 ONNX file (e.g. produced by scripts/paragonsr/export_onnx.py)
    - Transform:
        * Replace any explicit Mish nodes with:
              mish(x) = x * tanh(softplus(x))
          using only standard ONNX ops.
        * Convert weights to FP16 (keeping IO as FP32).
        * Mark opset as 17 (best-effort, via opset_import edit only).
    - Output:
        * {basename}_op17_fp16_nomish.onnx

Usage example:
    python -m scripts.paragonsr.export_onnx_nomish \
        --onnx-fp32 release_models/4xParagonSR_S_pretrain_fused_op18_fp32.onnx \
        --out-dir release_models
"""

import argparse
import os
import sys
from typing import List

import onnx
from onnx import helper
from onnxconverter_common import float16

# Target opset for maximum compatibility
TARGET_OPSET = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Mish-free FP16 ONNX (NoMish) from a fused FP32 ONNX model, "
            "targeting opset 17 for maximum compatibility."
        )
    )
    parser.add_argument(
        "--onnx-fp32",
        type=str,
        required=True,
        help=(
            "Path to the base FP32 ONNX model (e.g. *_op18_fp32.onnx) to convert. "
            "This file will NOT be modified."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="release_models",
        help=(
            "Directory where the *_op17_fp16_nomish.onnx file will be saved. "
            "Will be created if it does not exist. Default: release_models"
        ),
    )
    parser.add_argument(
        "--target-opset",
        type=int,
        default=TARGET_OPSET,
        help=("Target opset to mark in the NoMish model. Default: 17."),
    )
    return parser.parse_args()


def get_model_opset(model: onnx.ModelProto) -> int:
    if model.opset_import:
        # Use the default (empty domain) opset if present; otherwise pick the first.
        for imp in model.opset_import:
            if imp.domain == "" or imp.domain is None:
                return imp.version
        return model.opset_import[0].version
    # Reasonable default if missing
    return 17


def set_model_opset(model: onnx.ModelProto, opset: int) -> None:
    # Replace (or set) the default-domain opset_import to TARGET_OPSET
    found_default = False
    for imp in model.opset_import:
        if imp.domain == "" or imp.domain is None:
            imp.version = opset
            found_default = True
    if not found_default:
        model.opset_import.extend([helper.make_opsetid("", opset)])


def replace_mish_nodes_with_nomish(model: onnx.ModelProto) -> bool:
    """
    Replace any explicit Mish nodes with:
        y = x * tanh(softplus(x))

    Assumes:
        - Mish is represented as a Node with op_type == "Mish"
        - Or via a known custom domain (we treat all such nodes by op_type only).

    Returns:
        True if any Mish-like node was replaced, False otherwise.

    NOTE:
    - If PyTorch already exported Mish as primitive ops, there will be no Mish
      nodes to replace and this function will be a no-op.
    """
    graph = model.graph
    nodes: list[onnx.NodeProto] = list(graph.node)

    # Build lookup of existing value names to avoid accidental collisions
    existing_names = set()
    for n in nodes:
        existing_names.update(n.input)
        existing_names.update(n.output)
    for v in list(graph.input) + list(graph.output) + list(graph.value_info):
        existing_names.add(v.name)
    for init in graph.initializer:
        existing_names.add(init.name)

    def make_unique(base: str) -> str:
        idx = 0
        name = base
        while name in existing_names:
            idx += 1
            name = f"{base}_{idx}"
        existing_names.add(name)
        return name

    new_nodes: list[onnx.NodeProto] = []
    replaced_any = False

    for node in nodes:
        if node.op_type == "Mish":
            if len(node.input) != 1 or len(node.output) != 1:
                # Unexpected shape, keep as-is for safety
                new_nodes.append(node)
                continue

            x = node.input[0]
            y = node.output[0]

            # We create:
            #   softplus = Softplus(x)
            #   t = Tanh(softplus)
            #   out = Mul(x, t)

            softplus_out = make_unique(f"{y}_softplus")
            tanh_out = make_unique(f"{y}_tanh")

            softplus_node = helper.make_node(
                "Softplus",
                inputs=[x],
                outputs=[softplus_out],
                name=make_unique(f"{node.name}_Softplus") if node.name else "",
            )

            tanh_node = helper.make_node(
                "Tanh",
                inputs=[softplus_out],
                outputs=[tanh_out],
                name=make_unique(f"{node.name}_Tanh") if node.name else "",
            )

            mul_node = helper.make_node(
                "Mul",
                inputs=[x, tanh_out],
                outputs=[y],
                name=make_unique(f"{node.name}_Mul") if node.name else "",
            )

            new_nodes.extend([softplus_node, tanh_node, mul_node])
            replaced_any = True
        else:
            new_nodes.append(node)

    if replaced_any:
        # Replace graph nodes
        del graph.node[:]  # type: ignore[arg-type]
        graph.node.extend(new_nodes)

    return replaced_any


def convert_to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Convert model weights to FP16 where appropriate using onnxconverter_common.float16.
    Keep IO types as FP32 to maximize drop-in compatibility.
    """
    return float16.convert_float_to_float16(
        model,
        keep_io_types=True,
    )


def try_set_target_opset(model: onnx.ModelProto, target_opset: int) -> onnx.ModelProto:
    """
    Best-effort opset adjustment by editing opset_import only.

    IMPORTANT:
    - We do NOT use the ONNX C API version_converter (fragile for Mish/etc.).
    - After replacing Mish with primitives, most consumers are fine as long as
      the declared opset is <= what they support and ops are available.

    Strategy:
        - If current opset > target_opset:
            set default-domain opset_import to target_opset.
        - Otherwise:
            leave as is.
    """
    current = get_model_opset(model)
    if current > target_opset:
        set_model_opset(model, target_opset)
    return model


def derive_nomish_fp16_path(
    base_fp32_path: str, out_dir: str, target_opset: int
) -> str:
    base = os.path.basename(base_fp32_path)
    name, _ = os.path.splitext(base)
    return os.path.join(out_dir, f"{name}_op{target_opset}_fp16_nomish.onnx")


def main() -> None:
    args = parse_args()

    onnx_fp32_path = os.path.abspath(args.onnx_fp32)
    out_dir = os.path.abspath(args.out_dir)
    target_opset = args.target_opset

    print("--- ParagonSR ONNX NoMish FP16 Compatibility Export ---")
    print(f"Base FP32 ONNX: {onnx_fp32_path}")
    print(f"Output directory: {out_dir}")
    print(f"Target opset: {target_opset}")

    if not os.path.isfile(onnx_fp32_path):
        print(f"ERROR: Base FP32 ONNX file does not exist: {onnx_fp32_path}")
        sys.exit(1)

    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Failed to create output directory '{out_dir}': {e}")
        sys.exit(1)

    # 1) Load base FP32 ONNX
    try:
        base_model = onnx.load(onnx_fp32_path)
    except Exception as e:
        print(f"ERROR: Failed to load base FP32 ONNX: {e}")
        sys.exit(1)

    original_opset = get_model_opset(base_model)
    print(f"Base model opset version (declared): {original_opset}")

    # 2) Remove explicit Mish nodes (if any).
    print("Scanning graph for Mish nodes to replace with standard ops...")
    replaced = replace_mish_nodes_with_nomish(base_model)
    if replaced:
        print("Replaced Mish nodes with (x * tanh(softplus(x))).")
    else:
        print(
            "No explicit Mish nodes found; model is already Mish-free or Mish is decomposed."
        )

    # 3) Convert to FP16 (keeping IO FP32).
    print("Converting model to FP16 (internal weights/ops, IO kept as FP32)...")
    try:
        fp16_model = convert_to_fp16(base_model)
    except Exception as e:
        print(f"ERROR: FP16 conversion failed: {e}")
        sys.exit(1)

    # 4) Best-effort opset adjustment to target_opset.
    print(
        f"Setting declared opset (best-effort) to target_opset={target_opset} if needed..."
    )
    fp16_model = try_set_target_opset(fp16_model, target_opset)
    final_opset = get_model_opset(fp16_model)
    print(f"Final model opset version (declared): {final_opset}")

    # 5) Derive output path and save
    onnx_fp16_nomish_path = derive_nomish_fp16_path(
        base_fp32_path=onnx_fp32_path,
        out_dir=out_dir,
        target_opset=target_opset,
    )

    try:
        onnx.save(fp16_model, onnx_fp16_nomish_path)
    except Exception as e:
        print(f"ERROR: Failed to save FP16 NoMish ONNX: {e}")
        sys.exit(1)

    if not os.path.isfile(onnx_fp16_nomish_path):
        print(
            "ERROR: NoMish FP16 ONNX file not found after save. "
            "Please check filesystem permissions."
        )
        sys.exit(1)

    print(f"Successfully wrote NoMish FP16 ONNX to: {onnx_fp16_nomish_path}")
    print(
        "\nNotes:\n"
        "- Designed for stricter runtimes:\n"
        f"  * No Mish op (replaced when present)\n"
        f"  * FP16 weights with FP32 IO\n"
        f"  * Declared opset <= {target_opset} (best-effort)\n"
        "- If a runtime still cannot load this, users should fall back to the "
        "canonical fp32/fp16 ONNX exports."
    )


if __name__ == "__main__":
    main()
