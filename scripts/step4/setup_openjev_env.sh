#!/usr/bin/env bash
# Separate environment: never upgrades the active loopjev environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA=/root/siton-tmp/home/liuxinyu/miniconda3/bin/conda
PREFIX="$ROOT/.envs/openjev"
if [ -e "$PREFIX" ]; then
  echo "Environment already exists: $PREFIX; inspect it rather than overwrite." >&2
  exit 1
fi
"$CONDA" create -y -p "$PREFIX" python=3.11
"$PREFIX/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$PREFIX/bin/python" -m pip install 'transformers==5.10.2' 'peft==0.19.1' 'accelerate==1.13.0' safetensors numpy scipy pyarrow
"$PREFIX/bin/python" -c 'import torch,transformers,peft; print(torch.__version__,transformers.__version__,peft.__version__)'
