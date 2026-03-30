from __future__ import annotations


def estimate_real_size_mm(
    pixel_size: float,
    focal_length_px: float,
    depth_m: float,
) -> float:
    """Estimate real-world size in mm using the pinhole camera model.

    Formula:  size_real_mm = pixel_size * depth_m * 1000 / focal_length_px

    where focal_length_px can be computed from physical specs as:
        focal_length_px = focal_length_mm * image_width_px / sensor_width_mm

    Args:
        pixel_size: Object size in pixels (e.g., max bbox dimension).
        focal_length_px: Camera focal length in **pixels** (not mm).
        depth_m: Estimated depth to the object in metres.

    Returns:
        Estimated real-world size in millimetres.
    """
    safe_focal = max(focal_length_px, 1e-3)
    safe_depth = max(depth_m, 1e-3)
    return float(pixel_size * safe_depth * 1000.0 / safe_focal)


def focal_length_mm_to_px(
    focal_length_mm: float,
    image_width_px: int,
    sensor_width_mm: float = 3.68,
) -> float:
    """Convert focal length from mm to pixels.

    Args:
        focal_length_mm: Physical focal length in mm.
        image_width_px: Image width in pixels.
        sensor_width_mm: Physical sensor width in mm.
            Default 3.68mm is a typical 1/4" underwater camera sensor.

    Returns:
        Focal length in pixels.
    """
    return focal_length_mm * image_width_px / sensor_width_mm
