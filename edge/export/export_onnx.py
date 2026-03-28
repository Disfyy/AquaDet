from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ai_core.models.hybrid_model import AquaDetHybridModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AquaDet hybrid model to ONNX")
    parser.add_argument("--weights", type=Path, default=None, help="Optional .pt state dict")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    model = AquaDetHybridModel(num_classes=4, pi_ge_enabled=True)
    if args.weights and args.weights.exists():
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state, strict=False)

    model.eval()
    dummy = torch.randn(1, 3, args.height, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic axes for TensorRT optimization and dynamic stream sizing
    dynamic_axes = {
        "image": {0: "batch_size", 2: "height", 3: "width"},
        "logits": {0: "batch_size", 2: "height", 3: "width"},
        "boxes": {0: "batch_size", 2: "height", 3: "width"},
        "masks": {0: "batch_size", 2: "height", 3: "width"},
        "depth": {0: "batch_size", 2: "height", 3: "width"},
    }

    print(f"Exporting ONNX with dynamic input axes to {args.output}...")
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["image"],
        output_names=["logits", "boxes", "masks", "depth"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print("Export successful. Ready for TensorRT (trtexec) engine compilation.")


if __name__ == "__main__":
    main()
