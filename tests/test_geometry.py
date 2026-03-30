from ai_core.utils.geometry import estimate_real_size_mm, focal_length_mm_to_px


def test_estimate_real_size_mm_positive() -> None:
    size = estimate_real_size_mm(pixel_size=100, focal_length_px=500.0, depth_m=2.0)
    assert size > 0


def test_estimate_real_size_dimensions() -> None:
    """Verify the formula produces correct physical units.
    
    For a 100px object at 2m depth with focal length 500px:
    size_real = 100 * 2.0 * 1000 / 500 = 400mm
    """
    size = estimate_real_size_mm(pixel_size=100, focal_length_px=500.0, depth_m=2.0)
    assert abs(size - 400.0) < 0.1


def test_focal_length_conversion() -> None:
    """4.25mm lens on 3.68mm sensor at 1280px → ~1474.5px."""
    fpx = focal_length_mm_to_px(focal_length_mm=4.25, image_width_px=1280, sensor_width_mm=3.68)
    expected = 4.25 * 1280 / 3.68  # = 1478.26...
    assert abs(fpx - expected) < 0.01


def test_zero_depth_safety() -> None:
    """Depth of zero should not cause division by zero."""
    size = estimate_real_size_mm(pixel_size=100, focal_length_px=500.0, depth_m=0.0)
    assert size > 0
