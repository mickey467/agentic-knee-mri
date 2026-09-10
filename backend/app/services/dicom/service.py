import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from app.services.dicom.parser import scan_study_directory

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "sample_data"

DISPLAY_NAMES = {
    "acl": "Acute Knee Trauma (ACL)",
    "effusion": "Joint Effusion",
    "medial_oa": "Medial Compartment OA",
}


class StudyManager:
    """In-memory manager and cache for parsed DICOM studies."""
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or SAMPLE_DATA_DIR
        self._studies_cache: Dict[str, Dict[str, Any]] = {}

    def get_all_studies_summary(self) -> List[Dict[str, Any]]:
        """Returns lightweight summary for all available studies."""
        if not self.data_dir.exists():
            return []
            
        studies = []
        for p in self.data_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                study = self.get_study(p.name)
                if study:
                    studies.append({
                        "study_id": study["study_id"],
                        "name": DISPLAY_NAMES.get(
                            p.name, p.name.replace("_", " ").title()
                        ),
                        "patient_name": study.get("patient_name", "Anonymized"),
                        "patient_sex": study.get("patient_sex", ""),
                        "patient_age": study.get("patient_age", ""),
                        "series_count": len(study.get("series", [])),
                        "total_files": study.get("total_files", 0),
                        "modality": study.get("modality", "MR"),
                        "body_part": study.get("body_part", "KNEE"),
                        "series_descriptions": [s["series_description"] for s in study.get("series", [])]
                    })
        return studies

    def get_study(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Gets or parses full study metadata."""
        if study_id in self._studies_cache:
            return self._studies_cache[study_id]

        study_path = self.data_dir / study_id
        if not study_path.exists() or not study_path.is_dir():
            return None

        study_data = scan_study_directory(str(study_path))
        self._studies_cache[study_id] = study_data
        return study_data

    def get_series(self, study_id: str, series_id: str) -> Optional[Dict[str, Any]]:
        """Gets specific series data within a study."""
        study = self.get_study(study_id)
        if not study:
            return None
        for s in study.get("series", []):
            if s["series_id"] == series_id or str(s["series_number"]) == series_id:
                return s
        return None

    def get_slice_path(self, study_id: str, series_id: str, slice_index: int) -> Optional[str]:
        """Returns the file path for a specific slice index (1-indexed)."""
        series = self.get_series(study_id, series_id)
        if not series:
            return None
        slices = series.get("slices", [])
        if 1 <= slice_index <= len(slices):
            return slices[slice_index - 1]["file_path"]
        return None

# Global singleton instance
study_manager = StudyManager()
