# TensorRT Deployment Notes (Jetson)

1. Export ONNX:

```bash
python edge/export/export_onnx.py --weights artifacts/hybrid_model.pt --output artifacts/hybrid_model.onnx
```

2. Build TensorRT engine on device:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=artifacts/hybrid_model.onnx --saveEngine=artifacts/hybrid_model.engine --fp16
```

3. Runtime options:
- Use `--fp16` for Orin/Xavier
- Set power mode with `sudo nvpmodel -m 0`
- Lock clocks with `sudo jetson_clocks`

4. Measure throughput and latency with:

```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=artifacts/hybrid_model.engine --dumpProfile --separateProfileRun
```
