"""Shared test fixtures for AquaDet test suite."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ai_core.models.hybrid_model import AquaDetHybridModel
from ai_core.inference.pipeline import AquaDetPipeline


@pytest.fixture
def dummy_model() -> AquaDetHybridModel:
    """Create a lightweight model for testing."""
    torch.manual_seed(42)
    model = AquaDetHybridModel(num_classes=4, pi_ge_enabled=True)
    model.eval()
    return model


@pytest.fixture
def dummy_frame() -> np.ndarray:
    """Create a dummy BGR frame for inference tests."""
    return np.random.randint(0, 255, (96, 128, 3), dtype=np.uint8)


@pytest.fixture
def dummy_pipeline() -> AquaDetPipeline:
    """Create a pipeline with low threshold for testing."""
    torch.manual_seed(42)
    return AquaDetPipeline(conf_threshold=0.0, max_detections=5)
