#!/usr/bin/env bash
# Create the `loopjev` conda env. Pins: transformers<4.56 (Ouro-1.4B modeling code requirement), torch 2.8 cu128 (matches local driver 580 / H200).
set -euo pipefail
ENV=loopjev
conda create -y -n "$ENV" python=3.11
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/$ENV/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install \
  "transformers==4.55.0" "peft>=0.17" "accelerate>=1.0" "datasets>=3.0" "huggingface_hub[cli]" \
  safetensors einops numpy pandas pyarrow scikit-learn scipy matplotlib tqdm pyyaml \
  sentencepiece tiktoken ipykernel jupyter rich
$PY -m ipykernel install --user --name "$ENV" --display-name "Python ($ENV)"
$PY - <<'PYEOF'
import torch, transformers, peft, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "peft", peft.__version__, "datasets", datasets.__version__)
PYEOF
echo "ENV_OK"
