from __future__ import annotations


def estimate_real_size_mm(pixel_size: float, focal_length_mm: float, depth_m: float) -> float:
    """Estimate real-world size in mm from monocular depth approximation.

    Formula from concept: size_real = size_pixel * (f / Z)
    where f is focal length and Z is estimated depth.
    """
    safe_depth_m = max(depth_m, 1e-3)
    return float(pixel_size * (focal_length_mm / safe_depth_m))
