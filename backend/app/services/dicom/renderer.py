import io
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import pydicom
from pydicom.pixels import apply_voi_lut

def get_slice_pil_image(
    file_path: str, 
    window_center: Optional[float] = None, 
    window_width: Optional[float] = None
) -> Image.Image:
    """
    Reads a DICOM file and converts the pixel data into a standard 8-bit grayscale PIL Image.
    Applies VOI LUT / Windowing if present or min-max normalization.
    """
    ds = pydicom.dcmread(file_path)
    pixel_array = ds.pixel_array.astype(float)

    # Rescale slope / intercept if available
    rescale_slope = getattr(ds, "RescaleSlope", 1)
    rescale_intercept = getattr(ds, "RescaleIntercept", 0)
    pixel_array = pixel_array * rescale_slope + rescale_intercept

    # Try applying VOI LUT if available and no explicit window is passed
    if window_center is None or window_width is None:
        try:
            if "WindowCenter" in ds and "WindowWidth" in ds:
                pixel_array = apply_voi_lut(pixel_array, ds)
                min_val = np.min(pixel_array)
                max_val = np.max(pixel_array)
                if max_val > min_val:
                    pixel_array = (pixel_array - min_val) / (max_val - min_val) * 255.0
                else:
                    pixel_array = np.zeros_like(pixel_array)
            else:
                min_val = np.percentile(pixel_array, 1)
                max_val = np.percentile(pixel_array, 99)
                if max_val > min_val:
                    pixel_array = np.clip((pixel_array - min_val) / (max_val - min_val) * 255.0, 0, 255)
                else:
                    pixel_array = np.zeros_like(pixel_array)
        except Exception:
            min_val = np.min(pixel_array)
            max_val = np.max(pixel_array)
            if max_val > min_val:
                pixel_array = (pixel_array - min_val) / (max_val - min_val) * 255.0
            else:
                pixel_array = np.zeros_like(pixel_array)
    else:
        # Custom windowing
        lower = window_center - (window_width / 2.0)
        upper = window_center + (window_width / 2.0)
        pixel_array = np.clip((pixel_array - lower) / (upper - lower) * 255.0, 0, 255)

    img_8bit = pixel_array.astype(np.uint8)
    return Image.fromarray(img_8bit)

def get_slice_png_bytes(file_path: str) -> bytes:
    """Returns PNG image bytes for a DICOM slice file."""
    img = get_slice_pil_image(file_path)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
