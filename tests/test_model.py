"""Tests for AquaDetHybridModel forward pass and architecture integrity."""
from __future__ import annotations

import pytest
import torch

from ai_core.models.hybrid_model import AquaDetHybridModel
from ai_core.models.bifpn import BiFPN, BiFPNLayer
from ai_core.models.hybrid_backbone import HybridBackbone, AttentionBranch
from ai_core.models.pi_ge import PIGE
from ai_core.models.depth_head import DepthHead
from ai_core.models.multi_task_head import MultiTaskHead


class TestModelForwardPass:
    """Verify the full model produces correct output keys and shapes."""

    def test_output_keys(self, dummy_model: AquaDetHybridModel) -> None:
        x = torch.randn(1, 3, 640, 640)
        out = dummy_model(x)
        assert set(out.keys()) == {"logits", "boxes", "masks", "obj", "depth"}

    def test_output_shapes_640(self, dummy_model: AquaDetHybridModel) -> None:
        x = torch.randn(1, 3, 640, 640)
        out = dummy_model(x)
        # Feature grid is H/2 × W/2 = 320 × 320 (due to stride-2 first layer)
        assert out["logits"].shape == (1, 4, 320, 320)
        assert out["boxes"].shape == (1, 4, 320, 320)
        assert out["masks"].shape == (1, 1, 320, 320)
        assert out["obj"].shape == (1, 1, 320, 320)
        # Depth also at P3 resolution
        assert out["depth"].shape[0] == 1
        assert out["depth"].shape[1] == 1

    def test_output_shapes_160(self) -> None:
        """Smaller input for faster CI runs."""
        model = AquaDetHybridModel(num_classes=4)
        model.eval()
        x = torch.randn(1, 3, 160, 160)
        out = model(x)
        assert out["logits"].shape == (1, 4, 80, 80)
        assert out["boxes"].shape == (1, 4, 80, 80)
        assert out["masks"].shape == (1, 1, 80, 80)
        assert out["obj"].shape == (1, 1, 80, 80)

    def test_batch_size_2(self) -> None:
        model = AquaDetHybridModel(num_classes=4)
        model.eval()
        x = torch.randn(2, 3, 160, 160)
        out = model(x)
        assert out["logits"].shape[0] == 2
        assert out["depth"].shape[0] == 2

    def test_different_num_classes(self) -> None:
        for nc in [2, 4, 8]:
            model = AquaDetHybridModel(num_classes=nc)
            model.eval()
            out = model(torch.randn(1, 3, 160, 160))
            assert out["logits"].shape[1] == nc

    def test_pi_ge_disabled(self) -> None:
        model = AquaDetHybridModel(num_classes=4, pi_ge_enabled=False)
        model.eval()
        x = torch.randn(1, 3, 160, 160)
        out = model(x)
        assert "logits" in out


class TestGradientFlow:
    """Verify gradients flow through all model components."""

    def test_all_heads_receive_gradients(self) -> None:
        model = AquaDetHybridModel(num_classes=4)
        model.train()
        x = torch.randn(1, 3, 160, 160)
        out = model(x)

        # Sum all outputs to get a scalar loss
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        # Check that key parameters have gradients
        assert model.pi_ge.t_estimator[0].weight.grad is not None, "PI-GE has no gradient"
        assert model.backbone.conv.layers[0][0].weight.grad is not None, "ConvBranch has no gradient"
        assert model.backbone.attn.stage1[0].weight.grad is not None, "AttentionBranch has no gradient"
        assert model.neck.layers[0].td_conv_p4[0].weight.grad is not None, "BiFPN has no gradient"
        assert model.head.cls_head[-1].weight.grad is not None, "Cls head has no gradient"
        assert model.head.obj_head[-1].weight.grad is not None, "Obj head has no gradient"
        assert model.depth_head.reduce[0].weight.grad is not None, "Depth head has no gradient"


class TestSubModules:
    """Test individual architecture components in isolation."""

    def test_pige_passthrough_when_disabled(self) -> None:
        pige = PIGE(enabled=False)
        x = torch.randn(1, 3, 32, 32)
        out = pige(x)
        assert torch.allclose(x, out)

    def test_pige_output_range(self) -> None:
        pige = PIGE(enabled=True)
        x = torch.rand(1, 3, 32, 32)
        out = pige(x)
        assert out.min() >= -0.1, f"PI-GE output below range: {out.min()}"
        assert out.max() <= 1.1, f"PI-GE output above range: {out.max()}"

    def test_bifpn_preserves_shapes(self) -> None:
        bifpn = BiFPN(channels=64, num_repeats=2)
        p3 = torch.randn(1, 64, 80, 80)
        p4 = torch.randn(1, 64, 40, 40)
        p5 = torch.randn(1, 64, 20, 20)
        out = bifpn([p3, p4, p5])
        assert len(out) == 3
        assert out[0].shape == p3.shape
        assert out[1].shape == p4.shape
        assert out[2].shape == p5.shape

    def test_attention_branch_produces_3_levels(self) -> None:
        branch = AttentionBranch()
        x = torch.randn(1, 3, 160, 160)
        out = branch(x)
        assert len(out) == 3
        # All should be 128 channels
        for feat in out:
            assert feat.shape[1] == 128
        # Each level should be progressively smaller
        assert out[0].shape[-1] > out[1].shape[-1] > out[2].shape[-1]

    def test_depth_head_multi_scale_input(self) -> None:
        head = DepthHead(in_channels=384)
        x = torch.randn(1, 384, 40, 40)
        out = head(x)
        assert out.shape == (1, 1, 40, 40)
        assert (out > 0).all(), "Depth should be strictly positive"

    def test_multi_task_head_objectness(self) -> None:
        head = MultiTaskHead(in_channels=64, num_classes=4)
        features = [
            torch.randn(1, 64, 80, 80),
            torch.randn(1, 64, 40, 40),
            torch.randn(1, 64, 20, 20),
        ]
        out = head(features)
        assert "obj" in out
        assert out["obj"].shape == (1, 1, 80, 80)
