#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python}"
OUT="${OUT:-artifacts/step4/qwen3_1p7b_qv_r8}"
# A long GPU experiment must wait for other experiments to finish.
if [ -e "$OUT" ]; then
  echo "Output already exists: $OUT; run comparisons separately or choose a new OUT." >&2
  exit 1
fi
"$PY" scripts/run.py step4 train --out-dir "$OUT"
"$PY" scripts/run.py step4 compare --candidate "$OUT" --reference artifacts/step2/lora4 \
  --out "$OUT/results/vs-e1-four-loop.json"
"$PY" scripts/run.py step4 compare --candidate "$OUT" --reference artifacts/step2/lora2 \
  --out "$OUT/results/vs-e1-two-loop.json"
if [ -f artifacts/step3/uniform4/results/metrics.json ]; then
  "$PY" scripts/run.py step4 compare --candidate "$OUT" --reference artifacts/step3/uniform4 --reference-loop 2 \
    --out "$OUT/results/vs-e2-loop2.json"
  "$PY" scripts/run.py step4 compare --candidate "$OUT" --reference artifacts/step3/uniform4 --reference-loop 4 \
    --out "$OUT/results/vs-e2-loop4.json"
else
  echo "E2 metrics are not ready; E2 comparisons remain pending. See Step4 README."
fi
