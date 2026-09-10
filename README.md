# Agentic Knee MRI Analysis & Reporting System

**Created by Micheal Ehab** · Solo-developer portfolio project · Research use only — not a medical device.

Upload a knee MRI study (DICOM), get AI abnormality predictions, and receive an
evidence-grounded radiology-style report from an LLM agent — then chat about it.
```

DICOM studies → DINOv2 + LoRA ensemble → 12 abnormality scores
                    │
        ┌───────────┼───────────┐
     DICOM MCP   Model MCP   RAG MCP (FAISS)   Visualization MCP
        └───────────┼───────────┘
              LangGraph agent (report → chat)
```

> **Live demo:** coming via GitHub Codespaces (see Deployment).

## How it works

1. **Pick a study** — 3 pre-loaded de-identified knee MRI studies (ACL, effusion, medial OA).
2. **Vision inference** — the Model-G ensemble scores 12 abnormalities (CPU, ~20s).
3. **Agent report** — a LangGraph agent gathers metadata → predictions → cited
   medical evidence and writes a 4-section report (Study Info, Findings,
   Impression, Evidence). Missing age/sex pauses the run and asks you, then
   resumes. Progress streams live over SSE.
4. **Chat** — ask follow-ups (RAG answers) or drive the viewer
   ("show me the sagittal series" jumps the viewer there via `viz_commands`).

## Model G: DINOv2 + LoRA abnormality predictor

Shared input pipeline for all experiments — DINOv2 inference runs once and
embeddings are reused:

```
3D MRI series → 2.5D slices [i-1, i, i+1] → DINOv2-B/14 → 768-D embeddings
```

**Backbone.** DINOv2 ViT-B/14 at 336px (`interpolate_pos_encoding`), adapted
with LoRA (rank 8, alpha 16, dropout 0.05) on the `query`/`key`/`value`/`dense`
projections (48 layers). Only the adapter + head weights are trained and stored
(`dinov2_336_lora_finetuned_backbone.pt`).

**Preprocessing.** Percentile windowing to uint8, baseline intensity
normalization, laterality canonicalization (right knees flipped), 2.5D triplet
stacking (batch 8, ≤48 slices/series, ≤8 series/study), plus a 3-D metadata
vector `[fluid_sensitive, fat_suppression, slice_fraction]`.

**Model-G head.** Slice embeddings 768→384, metadata 3→256, concatenated series
fusion →256-D view space; 12 label-specific query vectors with learned
attention temperatures and per-label heads (LayerNorm→128→GELU→Dropout→1).
Sigmoid outputs, 0.50 threshold, mean over a **5-fold ensemble**
(`dino336_finetuned_fold_{0..4}_best.pt`).

**12 targets:** ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA,
Lateral OA, PF OA, Effusion, Synovitis, Baker's cyst, Contusion, Fracture.

**Experiment leaderboard** (macro ROC-AUC, shared pipeline):

| Rank | Model | Macro ROC-AUC | Description |
|------|-------|---------------|-------------|
| 1 | model_G + base dino v2 | **0.7651** | Label-specific anatomical-view fusion |
| 2 | baseline_B_mean_mean | 0.7208 | Mean slice pooling + mean series pooling |
| 3 | model_C_slice_attn | 0.7130 | Slice attention + mean series pooling |
| 4 | model_D_series_attn | 0.6045 | Slice attention + series attention |
| 5 | model_E_metadata | 0.5767 | Slice attention + series attention + metadata |
| 6 | model_F_final | 0.5767 | Final generic attention architecture |

## Repo layout

```
backend/
  app/
    api/endpoints/     # studies, reports (+ SSE stream), chat
    services/
      dicom/           # pydicom parsing, series sorting, PNG rendering
      model/           # inference.py — DINOv2+LoRA + Model-G ensemble
      mcp/             # 4 MCP servers (+ `python -m app.services.mcp <name>` stdio runners)
      agent/           # provider, tool adapters, prompts, graph, failover, trace
      rag/             # FAISS retriever (MiniLM-L6-v2, cosine)
  scripts/ingest_rag.py  # PDF → chunks → FAISS store
  tests/                 # 60+ pytest tests (offline-safe; `slow` needs weights)
  requirements.txt
frontend/              # Vite + React + Tailwind (black + cyan), served statically in prod
Dockerfile             # single-container HF Spaces image (port 7860)
.devcontainer/         # GitHub Codespaces demo environment
```

Large files (`backend/model/*.pt`, `backend/sample_data/`, `backend/rag_data/*.index`)
are **not** in git — see Deployment for where the demo gets them.

## Local quickstart

```powershell
cd backend
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:OPENROUTER_API_KEY='...'          # primary LLM
$env:OPENROUTER_API_KEY_FALLBACK='...' # optional secondary key (quota relief)
.\.venv\Scripts\python -m uvicorn app.main:app --port 8000

cd ..\frontend
npm install; npm run dev -- --port 5173   # proxies /api → :8000
```

Open http://localhost:5173. First inference downloads DINOv2 weights (~350MB)
once; first RAG query loads MiniLM (~90MB) once.

Rebuild the knowledge base:
```powershell
.\.venv\Scripts\python scripts\ingest_rag.py --pdf-dir "<pdf folder>"  # → backend/rag_data/
```

Run tests: `.\.venv\Scripts\python -m pytest tests\` (`-m "not slow"` skips
weight-dependent tests).

## API

| Method & path | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /api/studies` | list demo studies |
| `GET /api/studies/{id}[/metadata]` | study detail / clinical summary |
| `GET /api/studies/{id}/series/{s}/slices/{i}/image` | slice PNG |
| `POST /api/studies/{id}/analyze` | 12-class probabilities |
| `POST /api/studies/{id}/report` | generate report (may pause for demographics; resume via `resume_payload`) |
| `POST /api/studies/{id}/report/stream` | same, as SSE (`tool`/`paused`/`report`/`error` events) |
| `GET /api/studies/{id}/report?session_id=` | fetch completed report |
| `POST /api/studies/{id}/chat` | chat (`reply`, `viz_commands`, `tool_trace`) |

Responses carry `"model": "primary"|"fallback"`.

## Deployment

- **Demo (free, no card):** GitHub Codespaces (4-core) via `.devcontainer/` —
  weights/data download from Hugging Face Hub repos on first boot, keys via
  Codespaces secrets, forward port 5173 as Public.
- **Single container:** `docker build -t knee-mri .` (see `Dockerfile`, HF
  Spaces port 7860; bake `model/`, `sample_data/`, `rag_data/` into the image).

## Limitations

- Demo/educational tool. Model outputs and LLM text are **not diagnoses**;
  independent radiologist review required.
- Chat history is in-memory (restarts wipe it); the agent never re-runs
  inference in chat — it reuses report values.
- Free-tier LLMs are slow (reports take minutes) and rate-limited; the
  secondary key + SSE progress exist for exactly that.
