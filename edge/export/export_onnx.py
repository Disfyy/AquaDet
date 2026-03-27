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

    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["image"],
        output_names=["logits", "boxes", "masks", "depth"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported ONNX: {args.output}")


if __name__ == "__main__":
    main()
