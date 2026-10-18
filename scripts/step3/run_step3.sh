#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python}"
OUT="${OUT:-artifacts/step3/uniform4}"
REF4="${REF4:-artifacts/step2/lora4}"
REF2="${REF2:-artifacts/step2/lora2}"
for ref in "$REF4" "$REF2"; do
  test -f "$ref/results/metrics.json"
  test -f "$ref/results/validation_predictions.pt"
done
# Prevent an accidental duplicate long training run.
if [ -e "$OUT" ]; then
  echo "Output already exists: $OUT. Use a new OUT; compare existing results separately." >&2
  exit 1
fi
"$PY" scripts/run.py step3 train --mode lora --loops 4 --weighting uniform \
  --model models/Ouro-1.4B --data data/Open-Jev-v1.1 \
  --cache-dir artifacts/step1_expanded/cache --out-dir "$OUT"
"$PY" scripts/run.py step3 compare --candidate "$OUT" --reference "$REF4" \
  --out "$OUT/results/vs-e1-four-loop.json"
"$PY" scripts/run.py step3 compare --candidate "$OUT" --reference "$REF2" \
  --out "$OUT/results/vs-e1-two-loop.json"
