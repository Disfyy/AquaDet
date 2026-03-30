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
        "obj": {0: "batch_size", 2: "height", 3: "width"},
        "depth": {0: "batch_size", 2: "height", 3: "width"},
    }

    print(f"Exporting ONNX with dynamic input axes to {args.output}...")
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["image"],
        output_names=["logits", "boxes", "masks", "obj", "depth"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print("Export successful. Testing output validity...")

    try:
        import onnx
        model_onnx = onnx.load(str(args.output))
        onnx.checker.check_model(model_onnx)
        print("ONNX model checker passed.")

        import onnxruntime as ort
        import numpy as np
        
        # Pure inference outputs from torch for verification
        with torch.no_grad():
            pt_out = model(dummy)

        sess = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
        ort_out = sess.run(None, {"image": dummy.numpy()})

        for key, ort_val in zip(["logits", "boxes", "masks", "obj", "depth"], ort_out):
            np.testing.assert_allclose(pt_out[key].numpy(), ort_val, atol=1e-4, rtol=1e-3)
            
        print("Numerical validation passed. Max absolute error is within 1e-4 tolerance.")
        print("Ready for TensorRT (trtexec) engine compilation.")
    except ImportError as e:
        print(f"Skipping validation step due to missing dependency: {e}")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    main()
