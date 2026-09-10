"""
Tests for the inference service (Milestone 2).

These tests verify that:
1. The inference module's preprocessing utilities work correctly.
2. The ModelG architecture can be instantiated and forward-passes without errors.
3. The FastAPI /analyze endpoint returns valid 12-class probabilities.

NOTE: Full end-to-end model inference (backbone + folds) is tested via the
`test_analyze_endpoint_integration` test which is marked with `slow` — skip it
in CI with `pytest -m "not slow"` to avoid downloading DINOv2 weights.
"""

import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"

# ---------------------------------------------------------------------------
# Preprocessing unit tests (no model weights needed)
# ---------------------------------------------------------------------------

def test_normalize_intensity_baseline():
    from app.services.model.inference import normalize_intensity
    # Values strictly in [0, 255] — after /255 they must be in [0, 1]
    vol = np.random.randint(0, 256, (10, 64, 64), dtype=np.uint8).astype(np.float32)
    norm = normalize_intensity(vol, mode="baseline")
    assert norm.max() <= 1.0 + 1e-6, "Baseline normalization should divide by 255"
    assert norm.min() >= 0.0


def test_normalize_intensity_minmax():
    from app.services.model.inference import normalize_intensity
    vol = np.array([[[0.0, 100.0, 200.0]]], dtype=np.float32)
    norm = normalize_intensity(vol, mode="minmax")
    assert abs(norm.min()) < 1e-5
    assert abs(norm.max() - 1.0) < 1e-5


def test_build_25d_indices():
    from app.services.model.inference import build_25d_indices
    triplets = build_25d_indices(5)
    assert triplets[0] == (0, 0, 1), "First triplet should clamp lo to 0"
    assert triplets[-1] == (3, 4, 4), "Last triplet should clamp hi to n-1"
    assert len(triplets) == 5


def test_make_25d_image():
    from app.services.model.inference import make_25d_image
    vol = np.random.rand(10, 64, 64).astype(np.float32)
    img = make_25d_image(vol, 0, 1, 2)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.float32


def test_canonicalize_laterality_right():
    from app.services.model.inference import canonicalize_laterality
    vol = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    flipped, was_flipped = canonicalize_laterality(vol, "R")
    assert was_flipped
    np.testing.assert_array_equal(flipped, vol[:, :, ::-1])


def test_canonicalize_laterality_unknown():
    from app.services.model.inference import canonicalize_laterality
    vol = np.ones((3, 3, 3), dtype=np.float32)
    out, was_flipped = canonicalize_laterality(vol, "unknown")
    assert not was_flipped
    np.testing.assert_array_equal(out, vol)


def test_infer_metadata_vector():
    from app.services.model.inference import infer_metadata_vector
    meta = infer_metadata_vector(n_slices=30, max_slices=60)
    assert meta.shape == (3,)
    assert meta[0] == 0.0  # fluid_sensitive
    assert meta[1] == 0.0  # fat_suppression
    assert abs(meta[2] - 0.5) < 1e-5  # slice_fraction = 30/60


# ---------------------------------------------------------------------------
# ModelG architecture tests (no weights needed — random init)
# ---------------------------------------------------------------------------

def test_model_g_forward_pass():
    """ModelG should accept the expected tensor shapes and output (1, 12) logits."""
    import torch
    from app.services.model.inference import ModelG, MODEL_G_CONFIG, METADATA_DIM, N_LABELS, MAX_SERIES_PER_STUDY, MAX_SLICES_PER_SERIES

    model = ModelG()
    model.eval()

    B, S, T, E = 1, MAX_SERIES_PER_STUDY, MAX_SLICES_PER_SERIES, MODEL_G_CONFIG["embed_dim"]
    with torch.no_grad():
        se = torch.zeros(B, S, T, E)
        sm = torch.zeros(B, S, T)
        md = torch.zeros(B, S, METADATA_DIM)
        smask = torch.ones(B, S)
        logits = model(se, sm, md, smask)

    assert logits.shape == (B, N_LABELS), f"Expected (1, 12), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits should not contain NaN"


def test_model_g_sigmoid_range():
    """Sigmoid of ModelG logits should be in [0, 1]."""
    import torch
    from app.services.model.inference import ModelG, MODEL_G_CONFIG, METADATA_DIM, MAX_SERIES_PER_STUDY, MAX_SLICES_PER_SERIES

    model = ModelG()
    model.eval()

    B, S, T, E = 2, MAX_SERIES_PER_STUDY, MAX_SLICES_PER_SERIES, MODEL_G_CONFIG["embed_dim"]
    with torch.no_grad():
        logits = model(
            torch.randn(B, S, T, E),
            torch.ones(B, S, T),
            torch.zeros(B, S, METADATA_DIM),
            torch.ones(B, S),
        )
        probs = torch.sigmoid(logits)

    assert (probs >= 0.0).all() and (probs <= 1.0).all()


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_analyze_endpoint_exists(client):
    """The /analyze endpoint should respond (not 404/405) even if inference fails."""
    response = client.post("/api/studies/acl/analyze", json={})
    # We accept 200 (inference succeeded), 500 (inference failed — model weights
    # not loaded in test env), but NOT 404 (route missing) or 405 (method not allowed).
    assert response.status_code in (200, 500, 422), (
        f"Unexpected status {response.status_code}: {response.text}"
    )


def test_analyze_nonexistent_study(client):
    """Analyzing a non-existent study_id should return 404."""
    response = client.post("/api/studies/nonexistent_study/analyze", json={})
    assert response.status_code == 404


@pytest.mark.slow
def test_analyze_endpoint_integration(client):
    """
    Full integration test: runs real inference on the 'acl' demo study.
    Skip in CI with: pytest -m "not slow"

    Requires:
      - backend/model/dinov2_336_lora_finetuned_backbone.pt
      - backend/model/dino336_finetuned_fold_*_best.pt
      - Either DINOV2_LOCAL_PATH env var or internet access for HF Hub download.
    """
    response = client.post("/api/studies/acl/analyze", json={})
    assert response.status_code == 200, f"Inference failed: {response.text}"

    data = response.json()

    # Check structure
    assert "probabilities" in data
    assert "predicted_labels" in data
    assert "raw_probs" in data
    assert "labels" in data
    assert data["n_folds"] == 5

    # Check 12 labels present
    probs = data["probabilities"]
    assert len(probs) == 12, f"Expected 12 labels, got {len(probs)}"

    raw = data["raw_probs"]
    assert len(raw) == 12

    # All probabilities should be in [0, 1]
    for label, p in probs.items():
        assert 0.0 <= p <= 1.0, f"Probability for {label} out of range: {p}"

    # At least one series was processed
    assert data["series_processed"] >= 1


def _wait_job(client, study_id, job_id, tries=60):
    import time

    for _ in range(tries):
        resp = client.get(f"/api/studies/{study_id}/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        if resp.json()["status"] == "done":
            return resp.json()["result"]
        time.sleep(0.5)
    raise AssertionError("background job did not finish in time")


def test_analyze_background_flow(client, monkeypatch):
    """Background submit → poll → result, with stubbed inference (fast)."""
    import app.services.model.inference as inference_mod
    from app.services.model import jobs

    jobs.reset()
    monkeypatch.setattr(
        inference_mod, "run_inference",
        lambda study_id, series_uids=None: {
            "study_id": study_id, "probabilities": {"ACL": 0.9},
            "predicted_labels": ["ACL"], "raw_probs": [0.9], "labels": ["ACL"],
            "series_processed": 1, "n_folds": 5,
        },
    )
    try:
        resp = client.post("/api/studies/acl/analyze", json={"background": True})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "running" and body["job_id"]
        result = _wait_job(client, "acl", body["job_id"])
        assert result["predicted_labels"] == ["ACL"]
    finally:
        jobs.reset()


def test_analyze_background_failure_surfaces(client, monkeypatch):
    import app.services.model.inference as inference_mod
    from app.services.model import jobs

    jobs.reset()
    monkeypatch.setattr(
        inference_mod, "run_inference",
        lambda study_id, series_uids=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    try:
        resp = client.post("/api/studies/acl/analyze", json={"background": True})
        job_id = resp.json()["job_id"]
        import time

        for _ in range(60):
            poll = client.get(f"/api/studies/acl/jobs/{job_id}")
            if poll.status_code == 500:
                assert "boom" in poll.json()["detail"]
                return
            time.sleep(0.5)
        raise AssertionError("failed job never surfaced the error")
    finally:
        jobs.reset()


def test_job_unknown_id_404(client):
    assert client.get("/api/studies/acl/jobs/nope").status_code == 404
    assert client.post("/api/studies/nope/analyze", json={"background": True}).status_code == 404
