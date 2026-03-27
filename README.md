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
Prefer explicit split keys:

- `data.train_image_dirs`
- `data.val_image_dirs`

## Prepare all local datasets

Aggregate all labeled data from `datasets/raw`, `datasets/external`, `datasets/processed`
into one unified YOLO-style train/val/test structure:

```bash
python scripts/prepare_hybrid_dataset.py --dataset-root datasets --class-map configs/class_map.yaml
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
python -m ai_core.training.train_hybrid --config configs/hybrid_train.yaml --out artifacts/hybrid_model.pt --workers 2 --save-every 1
```

This now saves:
- latest model: `artifacts/hybrid_model.pt`
- best model: `artifacts/hybrid_model_best.pt`
- epoch checkpoints: `artifacts/checkpoints/epoch_XXX.pt`

## Sanity check with GT metrics

```bash
python scripts/eval_inference_sanity.py \
	--images datasets/processed/unified/images/val \
	--weights artifacts/hybrid_model_best.pt \
	--max-images 50 \
	--min-avg-detections 0.3 \
	--min-avg-confidence 0.2 \
	--min-unique-classes 2 \
	--min-precision 0.05 \
	--min-recall 0.05
```

## Kaggle run (ready now)

1) Upload/import your dataset in Kaggle with structure:

- `/kaggle/input/aquadet-unified/images/train`
- `/kaggle/input/aquadet-unified/images/val`
- `/kaggle/input/aquadet-unified/labels/train`
- `/kaggle/input/aquadet-unified/labels/val`

2) In Kaggle Notebook, clone/copy this project into `/kaggle/working/AquaDet`.

3) Run training via helper:

```bash
cd /kaggle/working/AquaDet
python scripts/kaggle_train.py \
	--project-root /kaggle/working/AquaDet \
	--input-dataset /kaggle/input/aquadet-unified \
	--epochs 20 --batch 8 --imgsz 640 --workers 2
```

Output weights are saved to `/kaggle/working/artifacts/` and can be downloaded as Notebook output.
