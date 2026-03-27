from ai_core.utils.geometry import estimate_real_size_mm


def test_estimate_real_size_mm_positive() -> None:
    size = estimate_real_size_mm(pixel_size=100, focal_length_mm=4.0, depth_m=2.0)
    assert size > 0
