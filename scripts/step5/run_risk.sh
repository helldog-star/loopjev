#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python}"
ROOT="${ROOT:-artifacts/step5/risk_uniform4_v1}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
"$PY" scripts/run.py step5 risk-data extract --root "$ROOT"
"$PY" scripts/run.py step5 risk-fit fit --root "$ROOT"
"$PY" scripts/run.py step5 risk-data extract --root "$ROOT" --partitions final_check
"$PY" scripts/run.py step5 risk-fit check --root "$ROOT"
