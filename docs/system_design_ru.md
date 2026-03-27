# AquaDet: системный дизайн (MVP -> Production)

## 1) Hardware / Edge
- Основной вычислитель: Jetson Orin Nano / Xavier NX
- Камера: RGB + CPL (поляризационный фильтр)
- Телеметрия: ESP32 + GPS + pH + turbidity

## 2) AI Pipeline
1. `PI-GE` (очистка кадра, может быть отключен)
2. `HybridBackbone` (CNN + attention branch)
3. `BiFPN` (многомасштабное слияние)
4. `MultiTaskHead` (class + box + mask)
5. `DepthHead` + оценка реального размера: $size_{real}=size_{pixel}*f/Z$
6. `Tracker` (ID-последовательность между кадрами)

## 3) Backend / IoT
- API приемки детекций и телеметрии
- Aggregate summary endpoint для дашборда

## 4) Roadmap
- Phase 1: YOLO baseline + датасет
- Phase 2: кастомный hybrid
- Phase 3: PI-GE + глубина/размер
- Phase 4: ONNX/TensorRT + Jetson runtime

## 5) Подготовка данных (реализация в проекте)
- Объединение всех локальных датасетов в единый train/val/test:
	- `python scripts/prepare_hybrid_dataset.py --dataset-root datasets`
- Если реальные данные еще не загружены, быстрый bootstrap синтетики для smoke-тестов:
	- `python scripts/prepare_hybrid_dataset.py --dataset-root datasets --bootstrap-synthetic-count 800`
- Проверка качества датасета перед обучением:
	- `python scripts/check_dataset_quality.py --dataset datasets/processed/unified`
- Обучение hybrid-модели на объединенном наборе:
	- `python -m ai_core.training.train_hybrid --config configs/hybrid_train.yaml --out artifacts/hybrid_model.pt --workers 2 --save-every 1`
- Во время обучения автоматически считаются валидационные метрики (`val_loss`, `val_cls_acc`, `val_mask_iou`) и сохраняются:
	- `artifacts/hybrid_model_best.pt`
	- `artifacts/checkpoints/epoch_XXX.pt`
- Sanity-проверка с GT (precision/recall/F1):
	- `python scripts/eval_inference_sanity.py --images datasets/processed/unified/images/val --weights artifacts/hybrid_model_best.pt`
