import pytest
from pathlib import Path
from app.services.dicom.parser import scan_study_directory, determine_orientation
from app.services.dicom.service import study_manager
from app.services.dicom.renderer import get_slice_pil_image

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"

def test_determine_orientation():
    # Sagittal plane normal is along X axis: (1, 0, 0)
    sag_iop = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0] # normal is [ -1, 0, 0 ]
    assert determine_orientation(sag_iop) == "Sagittal"
    
    # Coronal plane normal is along Y axis: (0, 1, 0)
    cor_iop = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0] # normal is [ 0, 1, 0 ]
    assert determine_orientation(cor_iop) == "Coronal"
    
    # Axial plane normal is along Z axis: (0, 0, 1)
    ax_iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] # normal is [ 0, 0, 1 ]
    assert determine_orientation(ax_iop) == "Axial"

def test_scan_sample_studies():
    studies = study_manager.get_all_studies_summary()
    assert len(studies) >= 3
    
    study_ids = [s["study_id"] for s in studies]
    assert "acl" in study_ids
    assert "effusion" in study_ids
    assert "medial_oa" in study_ids

def test_acl_study_details():
    study = study_manager.get_study("acl")
    assert study is not None
    assert study["total_files"] > 0
    assert len(study["series"]) > 0
    
    # Verify slices in series
    first_series = study["series"][0]
    assert first_series["slice_count"] > 0
    
    # Test slice rendering
    slice_path = study_manager.get_slice_path("acl", first_series["series_id"], 1)
    assert slice_path is not None
    assert Path(slice_path).exists()
    
    img = get_slice_pil_image(slice_path)
    assert img.size[0] > 0
    assert img.size[1] > 0
    assert img.mode == "L"
