from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.dicom.service import study_manager
from app.services.dicom.renderer import get_slice_png_bytes

router = APIRouter(prefix="/studies", tags=["studies"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AnalyzePayload(BaseModel):
    """Request body for the analyze endpoint. series_uids is optional;
    if omitted, all series in the study will be used."""
    series_uids: Optional[List[str]] = None


class InferenceResult(BaseModel):
    study_id: str
    probabilities: Dict[str, float]
    predicted_labels: List[str]
    raw_probs: List[float]
    labels: List[str]
    series_processed: int
    n_folds: int

@router.get("", response_model=List[Dict[str, Any]])
def list_studies():
    """List all available pre-loaded and uploaded MRI studies."""
    return study_manager.get_all_studies_summary()

@router.get("/{study_id}")
def get_study(study_id: str):
    """Get full structured information for a specific study."""
    study = study_manager.get_study(study_id)
    if not study:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
    return study

def _series_label(s: dict) -> str:
    """Human-readable series label; sample files carry placeholder headers."""
    desc = (s.get("series_description") or "").strip()
    if desc and "dummy" not in desc.lower():
        return desc
    orient = s.get("orientation") or "Series"
    return f"{orient} Series {s.get('series_number', '')}".strip()


@router.get("/{study_id}/metadata")
def get_study_metadata(study_id: str):
    """Get a condensed clinical metadata summary of the study."""
    study = study_manager.get_study(study_id)
    if not study:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")
    
    return {
        "study_id": study["study_id"],
        "modality": study["modality"],
        "body_part": study["body_part"],
        "patient_name": study.get("patient_name", "Anonymized"),
        "patient_sex": study.get("patient_sex", ""),
        "patient_age": study.get("patient_age", ""),
        "has_complete_demographics": bool(study.get("patient_sex") and study.get("patient_age")),
        "total_files": study["total_files"],
        "series": [
            {
                "series_id": s["series_id"],
                "series_description": s["series_description"],
                "display_label": _series_label(s),
                "series_number": s["series_number"],
                "orientation": s["orientation"],
                "slice_count": s["slice_count"],
                "slice_thickness": s["slice_thickness"],
                "dimensions": f"{s['rows']}x{s['columns']}"
            }
            for s in study["series"]
        ]
    }

@router.get("/{study_id}/series/{series_id}")
def get_series(study_id: str, series_id: str):
    """Get information and slice list for a specific series."""
    series = study_manager.get_series(study_id, series_id)
    if not series:
        raise HTTPException(status_code=404, detail=f"Series '{series_id}' not found in study '{study_id}'")
    return series

@router.get("/{study_id}/series/{series_id}/slices/{slice_index}/image")
def get_slice_image(study_id: str, series_id: str, slice_index: int):
    """Returns the PNG rendering of a specific DICOM slice."""
    fpath = study_manager.get_slice_path(study_id, series_id, slice_index)
    if not fpath:
        raise HTTPException(status_code=404, detail=f"Slice {slice_index} not found for series '{series_id}'")
    
    try:
        png_bytes = get_slice_png_bytes(fpath)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render slice: {str(e)}")


@router.post("/{study_id}/analyze", response_model=InferenceResult)
def analyze_study(study_id: str, payload: Optional[AnalyzePayload] = None):
    """
    Run DINOv2-LoRA + Model G ensemble inference on a study.

    Optionally restrict to specific series UIDs via the request body.
    Returns 12-class sigmoid probabilities (one per knee abnormality label).
    """
    study = study_manager.get_study(study_id)
    if not study:
        raise HTTPException(status_code=404, detail=f"Study '{study_id}' not found")

    series_uids = (payload.series_uids if payload else None)

    try:
        from app.services.model.inference import run_inference
        result = run_inference(study_id, series_uids=series_uids)
        return InferenceResult(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
