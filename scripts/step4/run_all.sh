#!/usr/bin/env bash
# Run GPU workloads serially; stop on any failed validation or comparison.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python}"
JEVPY="$PWD/.envs/openjev/bin/python"
for path in artifacts/step4/qwen3_1p7b_qv_r8 artifacts/step4/openjev2b work/openjev_gpu_smoke; do
  if [ -e "$path" ]; then
    echo "Output exists: $path. Inspect results and resume individual commands; do not overwrite." >&2
    exit 1
  fi
done
for log in logs/step4_qwen.log logs/step4_openjev_smoke.log logs/step4_openjev_eval.log; do
  if [ -e "$log" ]; then echo "Log exists: $log" >&2; exit 1; fi
done
echo "Starting Qwen formal training and paired comparisons"
"$PY" scripts/run.py step4 matrix > logs/step4_qwen.log 2>&1
echo "Qwen completed; checking isolated Open-Jev environment"
"$JEVPY" -c 'import transformers,peft; assert transformers.__version__=="5.10.2"; assert peft.__version__=="0.19.1"'
"$JEVPY" scripts/run.py step4 external-eval --smoke --benchmark-rows 1 --benchmark-repeats 1 \
  --out-dir work/openjev_gpu_smoke > logs/step4_openjev_smoke.log 2>&1
echo "Open-Jev real-weight smoke passed; starting full external evaluation"
"$JEVPY" scripts/run.py step4 external-eval --out-dir artifacts/step4/openjev2b > logs/step4_openjev_eval.log 2>&1
for spec in 'artifacts/step2/lora4:4:e1-four-loop' 'artifacts/step2/lora2:2:e1-two-loop' 'artifacts/step3/uniform4:2:e2-loop2' 'artifacts/step3/uniform4:4:e2-loop4' 'artifacts/step4/qwen3_1p7b_qv_r8:1:qwen'; do
  IFS=: read -r reference loop label <<< "$spec"
  "$PY" scripts/run.py step4 external-compare --candidate artifacts/step4/openjev2b \
    --reference "$reference" --reference-loop "$loop" --out "artifacts/step4/openjev2b/results/vs-$label.json"
done
echo "Step4 pipeline completed"
