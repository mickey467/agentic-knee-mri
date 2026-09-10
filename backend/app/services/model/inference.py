"""
Inference module for Knee MRI abnormality detection.

Wraps the DINOv2-336 LoRA backbone + 5-fold Model G ensemble, adapted from
the original Kaggle notebook. Runs on CPU for HF Spaces deployment.

Key differences from the notebook:
- Paths are resolved relative to this file (no Kaggle paths).
- `load_volume` takes an explicit `series_dir` instead of a global lookup dict.
- `run_inference` is the single public entry point for the FastAPI backend.
- DEVICE is forced to CPU.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
# backend/app/services/model/ → ../../.. → backend/
BACKEND_DIR = _THIS_DIR.parents[2]
CKPT_DIR = BACKEND_DIR / "model"
SAMPLE_DATA_DIR = BACKEND_DIR / "sample_data"
# DINOv2 base model — we load it from HuggingFace cache on first call;
# callers may override via the DINOV2_LOCAL_PATH env var.
DINOV2_LOCAL_PATH: Optional[str] = os.environ.get("DINOV2_LOCAL_PATH")

# ---------------------------------------------------------------------------
# Configuration (must match training exactly)
# ---------------------------------------------------------------------------
LABELS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
N_LABELS = len(LABELS)

MODEL_G_CONFIG = {
    "embed_dim": 768,
    "hidden_dim": 384,
    "view_dim": 256,
    "dropout": 0.20,
    "classifier_hidden_dim": 128,
}
METADATA_DIM = 3  # [fluid_sensitive, fat_suppression, slice_fraction]

N_FOLDS = 5
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_KEYWORDS = ["query", "key", "value", "dense"]
DINO_RESOLUTION = 336
DINO_BATCH_SIZE = 8  # Conservative for CPU inference
VOLUME_RESIZE_HW = DINO_RESOLUTION
MAX_SLICES_PER_SERIES = 48
MAX_SERIES_PER_STUDY = 8
MAX_SLICES_FOR_METADATA_NORM = 60
INTENSITY_NORMALIZATION = "baseline"
USE_LATERALITY_CANONICALIZATION = True

# Auto-detect best available device (CUDA > MPS > CPU)
# HF Spaces free tier = CPU, paid/ZeroGPU tiers = CUDA
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[inference] Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Preprocessing utilities (verbatim from training notebook)
# ---------------------------------------------------------------------------

def normalize_intensity(volume: np.ndarray, mode: str = "baseline", eps: float = 1e-6) -> np.ndarray:
    volume = volume.astype(np.float32)
    if mode == "baseline":
        return volume / 255.0 if volume.max() > 1.5 else volume
    elif mode == "minmax":
        lo, hi = volume.min(), volume.max()
        return (volume - lo) / (hi - lo + eps)
    elif mode == "percentile":
        lo, hi = np.percentile(volume, [1, 99])
        clipped = np.clip(volume, lo, hi)
        return (clipped - lo) / (hi - lo + eps)
    else:
        raise ValueError(f"Unknown INTENSITY_NORMALIZATION mode: {mode}")


def canonicalize_laterality(volume: np.ndarray, side: str):
    """volume: (D, H, W). side: 'L' | 'R' | 'unknown'."""
    if side == "unknown":
        return volume, False
    if side == "R":
        return volume[:, :, ::-1].copy(), True
    return volume, False


def _dicom_slice_sort_key(ds) -> float:
    try:
        import numpy as np
        iop = [float(v) for v in ds.ImageOrientationPatient]
        ipp = [float(v) for v in ds.ImagePositionPatient]
        row = np.array(iop[:3], dtype=np.float64)
        col = np.array(iop[3:], dtype=np.float64)
        normal = np.cross(row, col)
        return float(np.dot(np.array(ipp, dtype=np.float64), normal))
    except Exception:
        try:
            return float(ds.InstanceNumber)
        except Exception:
            return 0.0


def _frame_to_uint8(frame2d: np.ndarray, slope: float, intercept: float, size: int) -> np.ndarray:
    arr = frame2d.astype(np.float32) * slope + intercept
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    arr = np.clip(arr, lo, hi)
    denom = (hi - lo) if (hi - lo) > 1e-6 else 1.0
    arr = (arr - lo) / denom
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape != (size, size):
        try:
            import cv2
            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image
            arr = np.array(Image.fromarray(arr).resize((size, size), Image.BILINEAR))
    return arr


def _dicom_to_uint8_slices(ds, size: int = VOLUME_RESIZE_HW) -> List[np.ndarray]:
    raw = np.asarray(ds.pixel_array)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if raw.ndim == 2:
        frames2d = [raw]
    elif raw.ndim == 3:
        if raw.shape[-1] in (3, 4) and n_frames <= 1:
            frames2d = [raw[..., :3].astype(np.float32).mean(axis=-1)]
        else:
            frames2d = [raw[f] for f in range(raw.shape[0])]
    elif raw.ndim == 4:
        frames2d = [raw[f, ..., :3].astype(np.float32).mean(axis=-1) for f in range(raw.shape[0])]
    else:
        raise ValueError(f"Unsupported pixel_array shape {raw.shape}")
    return [_frame_to_uint8(f, slope, intercept, size) for f in frames2d]


def load_volume(series_dir: str, size: int = VOLUME_RESIZE_HW, verbose: bool = False) -> Optional[np.ndarray]:
    """Load a (D, H, W) uint8 volume from a DICOM series directory."""
    import pydicom

    paths = [
        entry.path for entry in os.scandir(series_dir)
        if entry.is_file() and entry.name.lower().endswith(".dcm")
    ]
    if not paths:
        return None

    datasets = []
    for p in paths:
        try:
            datasets.append(pydicom.dcmread(p, force=True))
        except Exception as e:
            if verbose:
                print(f"  [load_volume] skipping unreadable {p}: {e}")
    if not datasets:
        return None

    datasets.sort(key=_dicom_slice_sort_key)

    slices: List[np.ndarray] = []
    for ds in datasets:
        try:
            slices.extend(_dicom_to_uint8_slices(ds, size=size))
        except Exception as e:
            if verbose:
                print(f"  [load_volume] skipping undecodable slice: {e}")
    if not slices:
        return None
    return np.stack(slices, axis=0)  # (D, H, W) uint8


def build_25d_indices(n_slices: int):
    return [(max(i - 1, 0), i, min(i + 1, n_slices - 1)) for i in range(n_slices)]


def make_25d_image(volume_norm: np.ndarray, i_lo: int, i_mid: int, i_hi: int) -> np.ndarray:
    """Returns (H, W, 3) float32."""
    return np.stack([volume_norm[i_lo], volume_norm[i_mid], volume_norm[i_hi]], axis=-1)


def get_series_laterality(series_dir: str) -> str:
    """Read laterality from the first DICOM header in a series directory."""
    import pydicom
    try:
        for entry in os.scandir(series_dir):
            if entry.is_file() and entry.name.lower().endswith(".dcm"):
                ds = pydicom.dcmread(entry.path, stop_before_pixels=True, force=True)
                value = getattr(ds, "Laterality", None) or getattr(ds, "ImageLaterality", None)
                if value:
                    v = str(value).strip().upper()
                    if v.startswith("L"):
                        return "L"
                    if v.startswith("R"):
                        return "R"
                break
    except Exception:
        pass
    return "unknown"


def infer_metadata_vector(n_slices: int, max_slices: int = MAX_SLICES_FOR_METADATA_NORM) -> np.ndarray:
    """Returns [fluid_sensitive=0, fat_suppression=0, slice_fraction] as float32."""
    slice_fraction = min(n_slices / max(max_slices, 1), 1.0)
    return np.array([0.0, 0.0, slice_fraction], dtype=np.float32)


# ---------------------------------------------------------------------------
# LoRA layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = W0 x + (alpha / r) * B(A(dropout(x))). W0 stays frozen."""
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_update = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_update


def apply_lora_to_vit(model: nn.Module, rank: int = 8, alpha: int = 16,
                       dropout: float = 0.05, target_keywords=("query", "key", "value")) -> nn.Module:
    replaced = 0
    for _, module in list(model.named_modules()):
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(k in attr_name.lower() for k in target_keywords):
                setattr(module, attr_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced += 1
    print(f"LoRA applied to {replaced} attention projection layers (rank={rank}, alpha={alpha}).")
    return model


# ---------------------------------------------------------------------------
# Model G architecture (exact reproduction from training)
# ---------------------------------------------------------------------------

class ModelG(nn.Module):
    """Label-specific multi-view fusion classifier."""
    def __init__(self, cfg: dict = MODEL_G_CONFIG, metadata_dim: int = METADATA_DIM, n_labels: int = N_LABELS):
        super().__init__()
        embed_dim = cfg["embed_dim"]
        hidden_dim = cfg["hidden_dim"]
        view_dim = cfg["view_dim"]
        dropout = cfg["dropout"]
        cls_hidden = cfg["classifier_hidden_dim"]

        self.input_projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, view_dim),
            nn.LayerNorm(view_dim),
        )
        self.series_projection = nn.Sequential(
            nn.Linear(hidden_dim + view_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.to_view = nn.Linear(hidden_dim, view_dim)

        self.label_queries = nn.Parameter(torch.randn(n_labels, view_dim) * 0.02)
        self.attention_temperature = nn.Parameter(torch.ones(n_labels))

        self.label_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(view_dim),
                nn.Linear(view_dim, cls_hidden),
                nn.GELU(),
                nn.Dropout(dropout / 2.0),
                nn.Linear(cls_hidden, 1),
            )
            for _ in range(n_labels)
        ])
        self.dropout = nn.Dropout(dropout)
        self.n_labels = n_labels

    def forward(self, slice_embeddings: torch.Tensor, slice_mask: torch.Tensor,
                metadata: torch.Tensor, series_mask: torch.Tensor):
        B, S, T, E = slice_embeddings.shape
        x = self.input_projection(slice_embeddings)
        x = self.dropout(x)

        mask = slice_mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=2)
        counts = mask.sum(dim=2).clamp(min=1.0)
        series_repr = summed / counts

        meta = self.metadata_projection(metadata)
        fused = torch.cat([series_repr, meta], dim=-1)
        fused = self.series_projection(fused)
        view_repr = self.to_view(fused)

        logits_out = []
        for label_idx in range(self.n_labels):
            query = self.label_queries[label_idx]
            temperature = self.attention_temperature[label_idx].abs() + 1e-6
            scores = (view_repr * query).sum(dim=-1) / temperature
            scores = scores.masked_fill(series_mask <= 0, -1e4)
            weights = F.softmax(scores, dim=1)
            fused_label = (weights.unsqueeze(-1) * view_repr).sum(dim=1)
            logit = self.label_heads[label_idx](fused_label).squeeze(-1)
            logits_out.append(logit)

        logits = torch.stack(logits_out, dim=1)  # [B, 12]
        return logits


# ---------------------------------------------------------------------------
# Lazy singleton model loader
# ---------------------------------------------------------------------------

_backbone = None
_processor = None
_fold_models: List[nn.Module] = []
_models_loaded = False


def _load_models() -> None:
    global _backbone, _processor, _fold_models, _models_loaded
    if _models_loaded:
        return

    from transformers import AutoImageProcessor, AutoModel

    backbone_ckpt = CKPT_DIR / "dinov2_336_lora_finetuned_backbone.pt"
    if not backbone_ckpt.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_ckpt}")

    # --- Load base DINOv2 ---
    local_path = DINOV2_LOCAL_PATH
    if local_path and os.path.isdir(local_path):
        print(f"Loading DINOv2 from local path: {local_path}")
        processor = AutoImageProcessor.from_pretrained(local_path)
        model = AutoModel.from_pretrained(local_path)
    else:
        print("Downloading DINOv2 from HuggingFace Hub (first run may be slow)...")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base")

    # Set resolution to 336
    if hasattr(processor, "size"):
        if isinstance(processor.size, dict):
            processor.size = {"height": DINO_RESOLUTION, "width": DINO_RESOLUTION}
        else:
            processor.size = DINO_RESOLUTION
    model.config.image_size = DINO_RESOLUTION

    # Apply LoRA
    model = apply_lora_to_vit(
        model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        target_keywords=LORA_TARGET_KEYWORDS,
    )

    # Load fine-tuned backbone weights
    state_dict = torch.load(backbone_ckpt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # --- Load Model G folds ---
    fold_models = []
    for i in range(N_FOLDS):
        fold_ckpt = CKPT_DIR / f"dino336_finetuned_fold_{i}_best.pt"
        if not fold_ckpt.exists():
            raise FileNotFoundError(f"Fold {i} checkpoint not found: {fold_ckpt}")
        fold_model = ModelG().to(DEVICE)
        fold_state = torch.load(fold_ckpt, map_location="cpu")
        fold_model.load_state_dict(fold_state, strict=True)
        fold_model.eval()
        for p in fold_model.parameters():
            p.requires_grad_(False)
        fold_models.append(fold_model)
        print(f"Fold {i} loaded [ok]")

    _processor = processor
    _backbone = model
    _fold_models = fold_models
    _models_loaded = True
    print(f"All {N_FOLDS} Model G folds loaded. Running on {DEVICE}.")


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _extract_embeddings(series_dir: str, laterality: str = "unknown") -> Optional[Dict[str, Any]]:
    """Extracts CLS embeddings for all slices in a series directory."""
    volume = load_volume(series_dir)
    if volume is None:
        return None

    if USE_LATERALITY_CANONICALIZATION:
        volume, _ = canonicalize_laterality(volume, laterality)

    vol_norm = normalize_intensity(volume, mode=INTENSITY_NORMALIZATION)
    n_slices = vol_norm.shape[0]
    triplets = build_25d_indices(n_slices)

    cls_embeds = []
    for start in range(0, n_slices, DINO_BATCH_SIZE):
        batch_triplets = triplets[start: start + DINO_BATCH_SIZE]
        imgs = [make_25d_image(vol_norm, lo, mid, hi) for (lo, mid, hi) in batch_triplets]
        imgs_uint8 = [(im * 255).astype(np.uint8) for im in imgs]
        inputs = _processor(
            images=imgs_uint8,
            return_tensors="pt",
            size={"height": DINO_RESOLUTION, "width": DINO_RESOLUTION},
        )
        pixel_values = inputs["pixel_values"].to(DEVICE)
        try:
            out = _backbone(pixel_values=pixel_values, interpolate_pos_encoding=True)
        except TypeError:
            out = _backbone(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        cls_embeds.append(cls)

    cls_embeds = np.concatenate(cls_embeds, axis=0)  # (N_slices, 768)
    return {"cls": cls_embeds, "n_slices": n_slices}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(study_id: str, series_uids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run full ensemble inference on a study.

    Args:
        study_id: The study folder name under SAMPLE_DATA_DIR.
        series_uids: Optional list of series UIDs to use. If None, all
                     series found in the study directory are used.

    Returns:
        Dict with keys:
          - "probabilities": Dict[label, float]
          - "predicted_labels": List[str] (threshold 0.5)
          - "raw_probs": List[float] (12 values)
          - "series_processed": int
    """
    _load_models()

    study_dir = SAMPLE_DATA_DIR / study_id
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")

    # Discover series directories (recursive: sample studies nest DICOMs as
    # study/<studyUID>/<seriesUID>/*.dcm; each leaf dir holding .dcm = one series)
    leaf_dirs: Dict[str, str] = {}
    for root, _, files in os.walk(study_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            leaf_dirs[os.path.basename(root)] = root

    if series_uids:
        # Match when the caller passes a UID equal to a leaf dirname
        series_dirs = {uid: leaf_dirs[uid] for uid in series_uids if uid in leaf_dirs}
    else:
        series_dirs = dict(leaf_dirs)

    if not series_dirs:
        raise ValueError(f"No series directories found in study: {study_dir}")

    S = MAX_SERIES_PER_STUDY
    T = MAX_SLICES_PER_SERIES
    E = MODEL_G_CONFIG["embed_dim"]

    slice_embeddings = np.zeros((S, T, E), dtype=np.float32)
    slice_mask = np.zeros((S, T), dtype=np.float32)
    metadata = np.zeros((S, METADATA_DIM), dtype=np.float32)
    series_mask = np.zeros((S,), dtype=np.float32)

    n_processed = 0
    for s_i, (series_uid, series_dir) in enumerate(list(series_dirs.items())[:S]):
        laterality = get_series_laterality(series_dir)
        result = _extract_embeddings(series_dir, laterality)
        if result is None:
            continue
        emb = result["cls"][:T]
        n = emb.shape[0]
        slice_embeddings[s_i, :n] = emb
        slice_mask[s_i, :n] = 1.0
        series_mask[s_i] = 1.0
        metadata[s_i] = infer_metadata_vector(n)
        n_processed += 1

    if n_processed == 0:
        raise ValueError(f"No series could be embedded for study: {study_id}")

    # Build tensors
    se = torch.from_numpy(slice_embeddings).unsqueeze(0).to(DEVICE)
    sm = torch.from_numpy(slice_mask).unsqueeze(0).to(DEVICE)
    md = torch.from_numpy(metadata).unsqueeze(0).to(DEVICE)
    smask = torch.from_numpy(series_mask).unsqueeze(0).to(DEVICE)

    # Ensemble over folds
    fold_probs = []
    with torch.inference_mode():
        for fold_model in _fold_models:
            logits = fold_model(se, sm, md, smask)
            probs = torch.sigmoid(logits).float().cpu().numpy()[0]
            fold_probs.append(probs)

    mean_probs = np.mean(fold_probs, axis=0).tolist()
    probabilities = {label: round(float(p), 4) for label, p in zip(LABELS, mean_probs)}
    predicted_labels = [label for label, p in zip(LABELS, mean_probs) if p >= 0.5]

    return {
        "study_id": study_id,
        "probabilities": probabilities,
        "predicted_labels": predicted_labels,
        "raw_probs": [round(float(p), 4) for p in mean_probs],
        "labels": LABELS,
        "series_processed": n_processed,
        "n_folds": N_FOLDS,
    }
