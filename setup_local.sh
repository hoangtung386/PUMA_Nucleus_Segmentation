#!/usr/bin/env bash
# Workstation setup for PUMA V13.2 using uv.
#
# Safe to run on a fresh copy of the project and safe to re-run. It deliberately
# rebuilds .venv from scratch, because a virtualenv carries absolute paths and cannot
# be moved between machines or directories.
#
#   bash setup_local.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

KERNEL_NAME="symbiopan"
KERNEL_LABEL="SymbioPan (uv .venv)"
CUDA_BACKEND="${CUDA_BACKEND:-cu128}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it, then re-run this script:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo '  export PATH="$HOME/.local/bin:$PATH"'
  exit 1
fi

echo "==> Project : $PROJECT_DIR"
echo "==> uv      : $(uv --version)"

# 1. Discard any virtualenv carried over from another machine: its scripts, its
#    sysconfig paths, and any Jupyter kernel pointing at it are all stale.
if [ -e .venv ]; then
  echo "==> Removing existing .venv (absolute paths do not survive a copy)"
  rm -rf .venv
fi

# 2. Fresh virtualenv. uv downloads CPython 3.11 itself, so no system Python, no apt.
#    3.11 is chosen over 3.13 for wider rasterio/timm wheel coverage.
uv venv --python 3.11 .venv
export VIRTUAL_ENV="$PROJECT_DIR/.venv"

# 3. PyTorch with CUDA. Deliberately NOT in requirements_colab.txt, because Colab
#    preinstalls torch; on a workstation it has to be installed explicitly.
#    Override the wheel build with e.g. CUDA_BACKEND=cu126 for an older driver.
uv pip install torch torchvision --torch-backend="$CUDA_BACKEND"

# 4. Project dependencies + Jupyter.
uv pip install -r requirements_colab.txt
uv pip install jupyterlab ipykernel ipywidgets

# 5. Re-register the kernel so it points at THIS machine's .venv.
if ./.venv/bin/jupyter kernelspec list 2>/dev/null | grep -qw "$KERNEL_NAME"; then
  ./.venv/bin/jupyter kernelspec uninstall -y "$KERNEL_NAME" >/dev/null 2>&1 || true
fi
./.venv/bin/python -m ipykernel install --user \
  --name "$KERNEL_NAME" --display-name "$KERNEL_LABEL"

# 6. The code looks for Dataset/ (capital D); the folder on disk is dataset/.
#    Linux paths are case-sensitive, so this symlink is required.
if [ ! -e Dataset ]; then
  ln -s dataset Dataset
  echo "==> Created symlink Dataset -> dataset"
fi

# 7. Verify the environment end to end.
./.venv/bin/python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
# Read the GPU inventory before importing torch, mirroring what the notebooks do.
from puma.gpu import query_gpu_inventory, select_cuda_device

inventory = query_gpu_inventory()
print(f"python           {sys.version.split()[0]}  ({sys.executable})")
print(f"gpus visible     {len(inventory)}")
for entry in inventory:
    memory = f"{entry['memory_mb'] / 1024:.0f} GB" if entry.get("memory_mb") else "unknown"
    print(f"  GPU {entry['index']}          {entry['name']}  {memory}")

plan = select_cuda_device(1, inventory=inventory, environ={})
if plan["selected_index"] is not None:
    print(f"notebooks will use GPU {plan['selected_index']} ({plan['selected_name']})")
    print(f"                 {plan['reason']}")

import torch

print(f"torch            {torch.__version__}")
print(f"cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(index)
        print(f"  torch cuda:{index}    {p.name}  {p.total_memory / 1024**3:.1f} GB  "
              f"sm_{p.major}{p.minor}  bf16={torch.cuda.is_bf16_supported()}")
else:
    print("WARNING: no CUDA device. Preprocessing works; Stage-1 training would be unusably slow.")

for name in ('numpy', 'pandas', 'scipy', 'tifffile', 'shapely', 'rasterio',
             'timm', 'huggingface_hub', 'safetensors', 'tqdm', 'psutil'):
    __import__(name)
print("deps             all 12 imports OK")

root = Path.cwd()
images = root / 'Dataset' / '01_training_dataset_tif_ROIs'
labels = root / 'Dataset' / '01_training_dataset_geojson_nuclei'
for label, directory, suffix in (('tif ROIs', images, '.tif'),
                                 ('geojson nuclei', labels, '.geojson')):
    count = len(list(directory.glob(f'*{suffix}'))) if directory.is_dir() else 0
    status = 'OK' if count else 'MISSING'
    print(f"{label:16s} {count} files  [{status}]  {directory}")
PY

cat <<EOF

Setup complete. Start JupyterLab from this directory:

    ./.venv/bin/jupyter lab

Run the notebooks in order on the "$KERNEL_LABEL" kernel:

    00_Preprocess.ipynb -> 01_Train_Stage1.ipynb
    02_Train_Stage2.ipynb -> 03_Evaluate_Infer.ipynb

See README_RUN_FIRST.md for the rest.

On a multi-GPU machine the notebooks train on GPU 1 by default. To override:

    CUDA_VISIBLE_DEVICES=0 ./.venv/bin/jupyter lab

Stage 2 downloads the gated MahmoodLab/UNI2-h checkpoint, so accept its terms
on Hugging Face and export a token before starting JupyterLab:

    export HF_TOKEN=hf_...
EOF
