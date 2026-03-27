PYTHON ?= python3

.PHONY: install install-train lint api mock-iot demo export-onnx prepare-data bootstrap-data check-data train-hybrid eval-sanity

install:
	$(PYTHON) -m pip install -r requirements.txt

install-train:
	$(PYTHON) -m pip install -r requirements-train.txt

lint:
	$(PYTHON) -m compileall ai_core backend edge iot scripts

api:
	$(PYTHON) -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8080 --reload

mock-iot:
	$(PYTHON) iot/mock/publish_mock_telemetry.py --url http://127.0.0.1:8080/api/v1/telemetry

demo:
	$(PYTHON) scripts/run_inference_demo.py --source 0 --show

prepare-data:
	$(PYTHON) scripts/prepare_hybrid_dataset.py --dataset-root datasets

bootstrap-data:
	$(PYTHON) scripts/prepare_hybrid_dataset.py --dataset-root datasets --bootstrap-synthetic-count 800

check-data:
	$(PYTHON) scripts/check_dataset_quality.py --dataset datasets/processed/unified

train-hybrid:
	$(PYTHON) -m ai_core.training.train_hybrid --config configs/hybrid_train.yaml --out artifacts/hybrid_model.pt

eval-sanity:
	$(PYTHON) scripts/eval_inference_sanity.py --images datasets/processed/unified/images/val --weights artifacts/hybrid_model.pt --max-images 30

train-hybrid-live:
	PYTHONUNBUFFERED=1 $(PYTHON) -u -m ai_core.training.train_hybrid --config configs/hybrid_train.yaml --out artifacts/hybrid_model.pt --log-interval 50

export-onnx:
	$(PYTHON) edge/export/export_onnx.py --output artifacts/hybrid_model.onnx
