from io import BytesIO
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from PIL import Image, ImageDraw

from app.services.catalog_service import CatalogService
from app.services.image_service import ImageService
from app.services.roi_service import RoiService
from app.utils.config_utils import load_config


def main():
    config = load_config()
    catalog = CatalogService(config).load_or_build()
    images = ImageService(catalog)
    roi_service = RoiService(catalog, config)

    output_dir = Path("artifacts/roi_checks")
    output_dir.mkdir(parents=True, exist_ok=True)

    case_list_path = Path("data/demo_subset_cases.txt")
    if not case_list_path.exists():
        print("No demo subset file found. Run scripts/inspect_full_imaging_dataset.py")
        return

    cases = [line.strip() for line in case_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case_id in cases:
        series_id = catalog._catalog[case_id]["default_series"]
        slice_paths = catalog.get_series(case_id, series_id)["slice_paths"]
        slice_id = 0
        found = False
        for idx in range(min(30, len(slice_paths))):
            rois = roi_service.get_rois(case_id, idx).rois
            if rois:
                slice_id = idx
                found = True
                break
        image_bytes, meta = images.get_slice_png(case_id, slice_id)
        image = Image.open(BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        draw = ImageDraw.Draw(image)
        rois = roi_service.get_rois(case_id, slice_id).rois
        for roi in rois:
            if roi.bbox:
                x = roi.bbox.x * meta.width
                y = roi.bbox.y * meta.height
                w = roi.bbox.w * meta.width
                h = roi.bbox.h * meta.height
                draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
                draw.text((x + 4, max(0, y - 14)), "Critical region", fill=(255, 255, 255))

        label = f"{case_id} slice {slice_id}"
        draw.text((10, 10), label, fill=(255, 255, 0))
        output_path = output_dir / f"{case_id}_{slice_id}.png"
        image.save(output_path)
        print(f"Saved {output_path} (roi_found={found})")


if __name__ == "__main__":
    main()
