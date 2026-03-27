import numpy as np
import torch

from ai_core.inference.pipeline import AquaDetPipeline


def test_pipeline_infer_frame_output_shape() -> None:
    torch.manual_seed(0)
    frame = np.zeros((96, 128, 3), dtype=np.uint8)

    pipeline = AquaDetPipeline(conf_threshold=0.0, max_detections=5)
    result = pipeline.infer_frame(frame, frame_index=3)

    assert result.frame_index == 3
    assert result.timestamp_ms > 0
    assert len(result.detections) <= 5

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0 <= x1 <= x2 < 128
        assert 0 <= y1 <= y2 < 96
        assert 0.0 <= det.confidence <= 1.0
        assert det.depth_m > 0
        assert det.real_size_mm >= 0
