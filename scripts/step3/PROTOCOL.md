# Step 3: E2 normalized multi-loop supervision

Run in `/root/siton-tmp/home/liuxinyu/loopjev` using the `loopjev` environment.
`step3_train.py` reuses Step2 input validation, initialization, model loading,
feature extraction, evaluation and latency measurement. Existing Step2 sources
and completed results are not modified. Keep the referenced Step2 scripts unchanged:
comparisons verify their recorded hashes against E1.

## Locked pilot

Default: fresh original Ouro, four actual loops, shared scalar head initialized
from the original Yes-minus-No rows, LoRA r8/alpha16 on q_proj and v_proj,
no bias, frozen gates. Uniform alpha=[.25,.25,.25,.25]. Each row contributes
sum(alpha[t] * (soft CE[t] + .1 Brier[t])); candidate softmax is separate for
each loop. Average across decision rows; add raw-head L2 once per update.
One backward per complete row includes all four loop losses. Head-only mode is
available as a diagnostic; the main E2 pilot is LoRA.

Same train/dev-cal/validation membership (4096/512/2048), candidate order,
seed0, shuffled epochs, 300 updates x 16 rows, head LR .001, LoRA LR .0002,
L2 .001, clip1, AdamW zero weight decay, checkpointing, candidate batch4 and
4096 padded-token budget as E1. E2 never continues an E1 checkpoint.
Equal steps do not imply equal FLOPs, especially against two-loop E1.
`--weighting late` ([.1,.2,.3,.4]) is optional and not in the default runner.

## Run

```bash
cd /root/siton-tmp/home/liuxinyu/loopjev
mkdir -p logs
bash scripts/run_step3.sh > logs/step3_uniform.log 2>&1
```

For a detached run, use `nohup bash scripts/run_step3.sh > logs/step3_uniform.log 2>&1 &`.
The runner executes one GPU condition then the two CPU comparison jobs sequentially.
It refuses an existing OUT. Override `PY`, `OUT`, `REF4`, `REF2` if necessary;
other seeds/budgets require matching fresh E1 references and manual training CLI.
The default output is `artifacts/step3/uniform4`.

```bash
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
$PY scripts/step3_train.py --mode lora --loops 4 --weighting uniform \
  --model models/Ouro-1.4B --data data/Open-Jev-v1.1 \
  --cache-dir artifacts/step1_expanded/cache --out-dir artifacts/step3/uniform4
$PY scripts/step3_compare.py --candidate artifacts/step3/uniform4 \
  --reference artifacts/step2/lora4 --out artifacts/step3/uniform4/results/vs-e1-four-loop.json
$PY scripts/step3_compare.py --candidate artifacts/step3/uniform4 \
  --reference artifacts/step2/lora2 --out artifacts/step3/uniform4/results/vs-e1-two-loop.json
```

If training completes but comparison fails, fix the pairing issue and rerun just
comparison; do not retrain or overwrite a completed experiment.

## Evaluation and interpretation

Metrics/predictions call the shared head `multi_loop_head`, not `final_only_head`.
Each loop gets raw and dev-cal temperature-scaled metrics; independent probes
are trained on freshly extracted adapted features. Official calibration/test/ood
are not read. Preserve select/check and all/group/source summaries from Step2.

Comparisons verify original model/data/cache provenance, E1 dependency hashes,
initialization, packages, GPU, training settings, schedules, candidate/token counts
and exact validation records. The three primary comparisons are:

- E2 loop2 versus E1 four-loop final: shallow quality target.
- E2 loop2 versus E1 two-loop final: fixed-depth baseline (training compute differs).
- E2 loop4 versus E1 four-loop final: deep quality retention.

Also report E2 loop2 versus E1-four-loop's loop2 for head transfer diagnostics.
Accuracy one-sided95 lower > -.01 AND NLL upper < .02 defines noninferiority.
Use group bootstrap, report raw/scaled separately; no multiplicity correction.
Single-seed pilot confidence intervals do not capture variation across training seeds.

Benchmark still executes all four loops on 32 fixed pretokenized validation rows.
Comparison latency ratio explicitly refers to full runtimes (4/4 or 4/2).
Reading loop2 predictions offline does NOT demonstrate measured 2-loop speedup
or adaptive exit. Device timeline includes host gaps; no FLOPs claim.

Keep final adapter/head/probe, predictions, metrics and protocol in results/;
reusable LoRA features in cache/. No per-step checkpoints. Logs record total
normalized loss; metrics.json additionally records per-loop losses.

## Functional smoke (not an experiment)

Append `--smoke --steps 1 --batch-rows 1 --bootstrap 20 --benchmark-rows 1
--benchmark-repeats 1` and use an isolated temporary directory under work/.
This uses eight rows per split. Matching E1 smoke runs need identical settings;
comparison requires `--allow-smoke` and remains marked SMOKE ONLY.
Remove agent-created smoke outputs after verification. Do not launch the full
300-step pilot merely to validate scripts.

## Validation performed (2026-10-18)

- Python compilation and shell syntax checks passed.
- Independent soft-label loss/gradient oracle passed for uniform and late weights;
  every loop receives gradients; synchronized candidate/target permutation preserves
  loss; final-only weights exactly reduce to E1; invalid weights are rejected.
- E2 uniform LoRA and matching E1 four/two-loop one-update smokes completed on
  eight rows/split, including adapter/head updates, extraction, probes and metrics.
  Both paired comparisons completed and retain SMOKE ONLY labels. Protocol checks
  reject changed seeds, step counts and E1 dependency hashes.
- Saved E2 adapter was reloaded and compared against extracted FP16 features using
  the existing BF16 tolerance (atol=.25, rtol=.03). It passed; maximum absolute
  difference was .3125, accepted by the combined absolute/relative criterion.
- Multi-loop checkpoint diagnostic on one complete row: BF16 ordinary repeated
  backward had relative gradient L2 difference .005232; ordinary vs checkpoint
  .005098, with identical losses. An initially stricter elementwise comparison
  failed. In FP32, ordinary vs checkpoint relative L2 difference was 2.42e-6
  (maximum absolute 2.95e-7). This is consistent with BF16 numerical/backward
  variability, not evidence of a distinct checkpoint error. Bitwise gradient
  equivalence in BF16 is not claimed. Frozen parameters received no gradients.
- No full 300-update Step3 experiment was launched. Validation outputs were
  removed after inspection; original Step2 files and experiment artifacts remain.
