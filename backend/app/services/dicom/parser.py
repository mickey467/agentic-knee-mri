import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pydicom

def determine_orientation(iop: Optional[List[float]]) -> str:
    """
    Determines anatomical orientation (Sagittal, Coronal, Axial) from 
    ImageOrientationPatient DICOM tag [Xx, Xy, Xz, Yx, Yy, Yz].
    """
    if not iop or len(iop) < 6:
        return "Unknown"
    
    try:
        x = np.array(iop[:3], dtype=float)
        y = np.array(iop[3:], dtype=float)
        z = np.cross(x, y)
        z = np.abs(z)
        
        max_idx = np.argmax(z)
        if max_idx == 0:
            return "Sagittal"
        elif max_idx == 1:
            return "Coronal"
        else:
            return "Axial"
    except Exception:
        return "Unknown"

def clean_dicom_str(val: Any, default: str = "") -> str:
    """Helper to convert DICOM PersonName or string elements into clean text."""
    if val is None:
        return default
    s = str(val).strip()
    return s if s else default

def parse_dicom_file_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    """Reads basic header tags from a single DICOM file."""
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        
        iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
        ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
        spacing = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None

        # Extract demographic tags with fallback logic
        patient_name = clean_dicom_str(getattr(ds, "PatientName", None), "Anonymized")
        patient_sex = clean_dicom_str(getattr(ds, "PatientSex", None), "")
        patient_age = clean_dicom_str(getattr(ds, "PatientAge", None), "")

        return {
            "file_path": str(filepath),
            "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "unknown")),
            "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "unknown")),
            "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "unknown")),
            "patient_name": patient_name,
            "patient_sex": patient_sex,
            "patient_age": patient_age,
            "instance_number": int(getattr(ds, "InstanceNumber", 0)),
            "series_description": str(getattr(ds, "SeriesDescription", "")),
            "series_number": int(getattr(ds, "SeriesNumber", 0)),
            "modality": str(getattr(ds, "Modality", "MR")),
            "body_part": str(getattr(ds, "BodyPartExamined", "KNEE")),
            "rows": int(getattr(ds, "Rows", 0)),
            "columns": int(getattr(ds, "Columns", 0)),
            "slice_thickness": float(getattr(ds, "SliceThickness", 0.0)) if "SliceThickness" in ds else None,
            "pixel_spacing": spacing,
            "image_orientation_patient": iop,
            "image_position_patient": ipp,
            "orientation": determine_orientation(iop),
            "repetition_time": float(getattr(ds, "RepetitionTime", 0.0)) if "RepetitionTime" in ds else None,
            "echo_time": float(getattr(ds, "EchoTime", 0.0)) if "EchoTime" in ds else None,
            "magnetic_field_strength": float(getattr(ds, "MagneticFieldStrength", 0.0)) if "MagneticFieldStrength" in ds else None
        }
    except Exception as e:
        return None

def scan_study_directory(study_dir: str) -> Dict[str, Any]:
    """
    Scans an entire study folder, groups slices by series, sorts them spatially,
    and returns a structured study representation.
    """
    study_path = Path(study_dir)
    if not study_path.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")

    dcm_files = list(study_path.rglob("*.dcm"))
    if not dcm_files:
        # Check files without extension
        dcm_files = [f for f in study_path.rglob("*") if f.is_file() and not f.name.startswith(".")]

    series_map: Dict[str, Dict[str, Any]] = {}
    study_info: Dict[str, Any] = {
        "study_id": study_path.name,
        "study_path": str(study_path.resolve()),
        "total_files": len(dcm_files),
        "modality": "MR",
        "body_part": "KNEE",
        "patient_name": "Anonymized",
        "patient_sex": "",
        "patient_age": "",
        "series": []
    }

    for fpath in dcm_files:
        meta = parse_dicom_file_metadata(str(fpath))
        if not meta:
            continue

        # Extract top-level demographics if found
        if meta["patient_name"] and study_info["patient_name"] == "Anonymized":
            study_info["patient_name"] = meta["patient_name"]
        if meta["patient_sex"] and not study_info["patient_sex"]:
            study_info["patient_sex"] = meta["patient_sex"]
        if meta["patient_age"] and not study_info["patient_age"]:
            study_info["patient_age"] = meta["patient_age"]

        s_uid = meta["series_instance_uid"]
        if s_uid not in series_map:
            series_map[s_uid] = {
                "series_id": s_uid,
                "series_description": meta["series_description"] or f"Series {meta['series_number']}",
                "series_number": meta["series_number"],
                "modality": meta["modality"],
                "orientation": meta["orientation"],
                "rows": meta["rows"],
                "columns": meta["columns"],
                "slice_thickness": meta["slice_thickness"],
                "pixel_spacing": meta["pixel_spacing"],
                "slices": []
            }

        series_map[s_uid]["slices"].append({
            "sop_instance_uid": meta["sop_instance_uid"],
            "instance_number": meta["instance_number"],
            "image_position_patient": meta["image_position_patient"],
            "file_path": meta["file_path"]
        })

    # Sort slices in each series by spatial position along slice normal or instance number
    for s_uid, s_data in series_map.items():
        slices = s_data["slices"]
        
        def sort_key(item):
            ipp = item.get("image_position_patient")
            if ipp and len(ipp) == 3:
                return (ipp[2], ipp[1], ipp[0])
            return (item.get("instance_number", 0), 0, 0)
            
        slices.sort(key=sort_key)
        
        for idx, sl in enumerate(slices, 1):
            sl["slice_index"] = idx
            
        s_data["slice_count"] = len(slices)
        study_info["series"].append(s_data)

    return study_info
