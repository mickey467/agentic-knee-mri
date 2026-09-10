#!/usr/bin/env bash
# One-time Codespaces setup: CPU torch, deps, demo artifacts, frontend.
# Large files live in Hugging Face Hub repos (set HF_MODEL_REPO / HF_DATA_REPO);
# API keys come from Codespaces secrets (OPENROUTER_API_KEY[_FALLBACK]).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "--- Python deps (CPU-only torch first) ---"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend/requirements.txt

echo "--- Demo artifacts from Hugging Face Hub ---"
: "${HF_MODEL_REPO:?Set HF_MODEL_REPO (e.g. USER/knee-mri-weights)}"
: "${HF_DATA_REPO:?Set HF_DATA_REPO (e.g. USER/knee-mri-demo-data)}"
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["HF_MODEL_REPO"],
                  local_dir="backend/model",
                  allow_patterns=["*.pt", "*.ipynb"])
snapshot_download(os.environ["HF_DATA_REPO"],
                  local_dir="backend",
                  allow_patterns=["sample_data/**", "rag_data/knee_faiss.index"])
print("artifacts downloaded")
EOF

echo "--- Frontend deps ---"
cd frontend && npm install && cd ..

echo "--- Sanity check ---"
cd backend && python -m pytest tests/test_dicom.py -q && cd ..
echo "SETUP DONE — start servers with: bash backend/scripts/codespaces-start.sh"
