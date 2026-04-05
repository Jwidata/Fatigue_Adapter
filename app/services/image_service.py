from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image

from app.models.schemas import SliceMeta
from app.services.catalog_service import CatalogService
from app.utils.dicom_utils import apply_window, get_window, load_dicom


class ImageService:
    def __init__(self, catalog: CatalogService):
        self.catalog = catalog

    def get_slice_png(self, case_id: str, slice_id: int) -> Tuple[bytes, SliceMeta]:
        path = self.catalog.get_slice_path(case_id, slice_id)
        ds = load_dicom(path)
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        pixels = pixels * slope + intercept
        window = get_window(ds)
        img_array, window_used = apply_window(pixels, window)
        image = Image.fromarray(img_array)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        meta = SliceMeta(
            case_id=case_id,
            slice_id=slice_id,
            width=img_array.shape[1],
            height=img_array.shape[0],
            window=list(window_used),
        )
        return buffer.getvalue(), meta
