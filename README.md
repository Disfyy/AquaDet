# AquaDet

AquaDet is an end-to-end Edge AI stack for aquatic waste monitoring:
- Detection + classification + segmentation
- Monocular depth-guided real-size estimation
- Temporal tracking to avoid duplicate counts
- Telemetry ingestion (GPS, pH, turbidity)
- Jetson-oriented export/deploy path (ONNX/TensorRT)

## Project layout

- `ai_core/` models, training, and real-time inference pipeline
- `backend/` FastAPI ingestion and metrics APIs
- `edge/` ONNX/TensorRT export + runtime scripts
- `iot/` ESP32 protocol notes + telemetry mock sender
- `configs/` YAML configs for phases
- `scripts/` helper scripts for setup/run

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
make api
```

In a second terminal:

```bash
python scripts/run_inference_demo.py --source 0 --show
```

## Roadmap mapping

- Phase 1: `ai_core/training/train_baseline.py`
- Phase 2: `ai_core/models/hybrid_model.py`
- Phase 3: `ai_core/models/pi_ge.py`, `ai_core/models/depth_head.py`
- Phase 4: `edge/export/export_onnx.py`, `edge/deploy/tensorrt_notes.md`

## Recommended dataset mix

For best efficiency/accuracy tradeoff in this architecture:

- Detection + segmentation: TrashCan 1.0 + TACO (+ optional Roboflow marine sets)
- PI-GE enhancement branch: UIEB pairs
- Monocular depth supervision: SUIM or synthetic simulator depth

Map your prepared folders in `configs/hybrid_train.yaml` under `data.image_dirs`.

## Prepare all local datasets

Aggregate all labeled data from `datasets/raw`, `datasets/external`, `datasets/processed`
into one unified YOLO-style train/val/test structure:

```bash
python scripts/prepare_hybrid_dataset.py --dataset-root datasets
```

If real data is not downloaded yet, generate a temporary synthetic dataset for end-to-end checks:

```bash
python scripts/prepare_hybrid_dataset.py --dataset-root datasets --bootstrap-synthetic-count 800
```

Quality gate (coverage/balance sanity check):

```bash
python scripts/check_dataset_quality.py --dataset datasets/processed/unified
```

## Train hybrid model

```bash
python -m ai_core.training.train_hybrid --config configs/hybrid_train.yaml --out artifacts/hybrid_model.pt
```
