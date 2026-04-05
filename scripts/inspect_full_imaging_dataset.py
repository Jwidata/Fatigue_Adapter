import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pydicom


def infer_case_id(path: Path) -> str:
    for parent in [path] + list(path.parents):
        name = parent.name
        if name.startswith("LIDC-IDRI"):
            return name
    return path.parent.name


def safe_get(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value)


def count_dicom_files(root: Path) -> int:
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(".dcm"):
                count += 1
    return count


def render_progress(current: int, total: int, width: int = 30):
    if total <= 0:
        return
    filled = int((current / total) * width)
    bar = "#" * filled + "-" * (width - filled)
    percent = (current / total) * 100
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({percent:5.1f}%)")
    sys.stdout.flush()


def main():
    root = Path("Data/medicalimages")
    report_path = Path("data/full_dataset_report.json")
    csv_path = Path("data/full_dataset_report.csv")
    ranked_path = Path("data/relevant_cases_ranked.csv")
    demo_subset_path = Path("data/demo_subset_cases.txt")
    eval_subset_path = Path("data/eval_subset_cases.txt")

    total_dicom = count_dicom_files(root)
    print(f"Total DICOM files: {total_dicom}")

    cases: Dict[str, Dict] = {}
    modality_counts: Dict[str, int] = {}
    total_files = 0
    dicom_files = 0
    unreadable = 0

    progress_every = 500
    processed = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            total_files += 1
            path = Path(dirpath) / filename
            if path.suffix.lower() != ".dcm":
                continue
            dicom_files += 1
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            except Exception:
                unreadable += 1
                continue
            processed += 1
            if processed % progress_every == 0:
                render_progress(processed, total_dicom)
            modality = safe_get(getattr(ds, "Modality", "unknown")) or "unknown"
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            case_id = infer_case_id(path)

            case_entry = cases.setdefault(
                case_id,
                {
                    "ct_series": {},
                    "seg_files": [],
                    "study_uids": set(),
                    "series_uids": set(),
                    "seg_references": set(),
                    "unreadable": 0,
                },
            )
            if modality == "CT":
                series_uid = safe_get(getattr(ds, "SeriesInstanceUID", "")) or "unknown"
                case_entry["study_uids"].add(safe_get(getattr(ds, "StudyInstanceUID", "")) or "unknown")
                case_entry["series_uids"].add(series_uid)
                series = case_entry["ct_series"].setdefault(
                    series_uid,
                    {"count": 0, "width": None, "height": None},
                )
                series["count"] += 1
                series["width"] = series["width"] or int(getattr(ds, "Columns", 0) or 0)
                series["height"] = series["height"] or int(getattr(ds, "Rows", 0) or 0)
            elif modality == "SEG":
                case_entry["seg_files"].append(str(path))
                ref_seq = getattr(ds, "ReferencedSeriesSequence", None)
                if ref_seq:
                    for item in ref_seq:
                        uid = safe_get(getattr(item, "SeriesInstanceUID", None))
                        if uid:
                            case_entry["seg_references"].add(uid)
    render_progress(processed, total_dicom)
    print()

    report = {
        "total_files": total_files,
        "dicom_files": dicom_files,
        "modality_counts": modality_counts,
        "unreadable": unreadable,
        "cases": {},
    }

    rows = []
    ranked_rows = []
    for case_id, entry in cases.items():
        ct_series_count = len(entry["ct_series"])
        ct_slice_count = sum(series["count"] for series in entry["ct_series"].values())
        seg_count = len(entry["seg_files"])
        seg_refs = entry["seg_references"]
        linkable = bool(seg_refs.intersection(entry["series_uids"])) if seg_refs else False
        roi_extractable = seg_count > 0 and linkable
        sample_dim = None
        for series in entry["ct_series"].values():
            if series["width"] and series["height"]:
                sample_dim = (series["width"], series["height"])
                break

        relevance_score = 0
        relevance_score += 3 if ct_slice_count > 50 else 1 if ct_slice_count > 0 else 0
        relevance_score += 3 if seg_count > 0 else 0
        relevance_score += 2 if linkable else 0
        relevance_score += 1 if sample_dim and sample_dim[0] >= 256 else 0
        relevance_score -= 2 if entry["unreadable"] > 0 else 0

        recommended_demo = "yes" if relevance_score >= 6 else "no"
        recommended_eval = "yes" if relevance_score >= 5 else "no"

        report["cases"][case_id] = {
            "ct_series_count": ct_series_count,
            "ct_slice_count": ct_slice_count,
            "seg_count": seg_count,
            "linkable_ct_seg": linkable,
            "roi_extractable": roi_extractable,
            "sample_dimensions": sample_dim,
        }

        rows.append(
            {
                "case_id": case_id,
                "ct_series_count": ct_series_count,
                "ct_slice_count": ct_slice_count,
                "seg_count": seg_count,
                "linkable_ct_seg": linkable,
                "roi_extractable": roi_extractable,
                "sample_dimensions": sample_dim,
            }
        )

        ranked_rows.append(
            {
                "case_id": case_id,
                "has_ct": ct_slice_count > 0,
                "has_seg": seg_count > 0,
                "ct_slice_count": ct_slice_count,
                "seg_count": seg_count,
                "roi_extractable": roi_extractable,
                "linkable_ct_seg": linkable,
                "relevance_score": relevance_score,
                "recommended_for_demo": recommended_demo,
                "recommended_for_model_eval": recommended_eval,
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    ranked_rows.sort(key=lambda x: x["relevance_score"], reverse=True)
    with ranked_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranked_rows[0].keys() if ranked_rows else [])
        writer.writeheader()
        writer.writerows(ranked_rows)

    demo_cases = [row["case_id"] for row in ranked_rows if row["recommended_for_demo"] == "yes"][:10]
    eval_cases = [row["case_id"] for row in ranked_rows if row["recommended_for_model_eval"] == "yes"]

    demo_subset_path.write_text("\n".join(demo_cases), encoding="utf-8")
    eval_subset_path.write_text("\n".join(eval_cases), encoding="utf-8")

    print(f"Report saved to {report_path}")
    print(f"Ranked cases saved to {ranked_path}")
    print(f"Demo subset: {len(demo_cases)} cases")
    print(f"Eval subset: {len(eval_cases)} cases")


if __name__ == "__main__":
    main()
