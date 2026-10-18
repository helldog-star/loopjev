# Step2 final-only development protocol

Run from `/root/siton-tmp/home/liuxinyu/loopjev` with the `loopjev` environment.
Scripts are owned by LoopJev; Open-Jev supplies the already verified renderer.
JevK5 training scripts are not used.

## Conditions and fixed settings

- `--mode head --loops 4`: frozen Ouro, original four-loop cached states, final-only scalar head.
- `--mode head --loops 2`: independently load Ouro with two loops, validate its frozen prefix against Step1, train only on cached round 2.
- `--mode lora --loops 4`: independently load original Ouro; train head plus r8 LoRA on q_proj/v_proj, supervise round 4.
- `--mode lora --loops 2`: independently load original Ouro with two loops; train head plus the same LoRA, supervise round 2.

All heads start at bare Yes-minus-No output rows, without a bias. LoRA runs start
from original Ouro and a fresh head, without continuing a head-only checkpoint.
Two-loop config is set before loading; runtime loops are asserted. Frozen
two-loop states may reuse the prefix only after a fresh two-loop forward check
(`atol=.25, rtol=.03`, BF16). Actual inference timing always uses the loaded model.

Data membership, prompt hashes, token lengths, candidate permutations and targets
must match Step1: train/dev-cal/validation = 4096/512/2048. Model/tokenizer/data
hashes are checked. These memberships stay fixed across training seeds.
Official calibration/test/ood are not read. Train-derived dev-cal fits each
method/loop temperature; the existing validation-select/check split is retained.

Defaults: 300 optimizer updates, 16 decision rows/update, seed 0, head LR .001,
LoRA LR .0002, AdamW with zero weight decay, raw-head L2 .001, gradient norm clip 1.
Rows follow the same deterministic shuffled epochs in all four conditions.
Each update averages soft CE + .1 Brier across complete decision rows. LoRA
processes one row at a time with gradient accumulation and non-reentrant decoder
checkpointing. Candidate microbatches preserve order, at most 4 candidates and
4096 padded tokens. All candidates of a row contribute jointly to its softmax.
LoRA rank 8, alpha 16, dropout 0, q_proj/v_proj only. Ouro gates stay frozen.

The independent diagnostic probes use the Step1 selected settings: 300 full-batch
updates, LR .01, L2 .01, CE+.1 Brier, train-only feature standardization. They do
not update Ouro. LoRA models re-extract train/dev-cal/validation states first.

## Commands

Use a fresh output directory for each condition. Run one GPU condition at a time.
This is a reproducible development pilot, not a claim that these training
hyperparameters are optimal.

```sh
PY=/root/siton-tmp/home/liuxinyu/miniconda3/envs/loopjev/bin/python
$PY scripts/step2_train.py --mode head --loops 4 \
  --model models/Ouro-1.4B --data data/Open-Jev-v1.1 \
  --cache-dir artifacts/step1_expanded/cache --out-dir artifacts/step2/head4
```

Repeat with `head/2`, `lora/4`, `lora/2`, using `head2`, `lora4`, `lora2` output
directories. Do not launch this matrix automatically while validating scripts.
For a short functional check, append `--smoke --steps 1 --batch-rows 1
--bootstrap 20 --benchmark-rows 1 --benchmark-repeats 1`, and use a temporary
directory. Smoke takes eight rows/split and cannot be reported as an experiment.

```sh
$PY scripts/step2_compare.py --candidate artifacts/step2/lora4 \
  --reference artifacts/step2/head4 --out artifacts/step2/comparisons/lora4-head4.json
$PY scripts/step2_compare.py --candidate artifacts/step2/lora2 \
  --reference artifacts/step2/lora4 --out artifacts/step2/comparisons/lora2-lora4.json
$PY scripts/step2_compare.py --candidate artifacts/step2/head2 \
  --reference artifacts/step2/head4 --out artifacts/step2/comparisons/head2-head4.json
```

Compare LoRA vs head at the same depth for backbone adaptation. Compare two vs
four loops within the same training mode for fixed-depth quality/cost. Compare
each independent probe at the same loop before/after LoRA to interpret final-head
transfer separately from representation readability. H2 concerns changes in the
shallow-vs-final gap, not shallow degradation alone; inspect pre/post probe curves.

## Evidence and artifacts

Results include raw/scaled accuracy (hard one-hot only), NLL/Brier (all rows),
two ECE variants, AURC/risk coverage, per-source/macro and predefined task groups,
soft-label NLL/Brier separately, and paired group bootstrap. Two-vs-four quality
noninferiority needs accuracy one-sided 95% lower > -.01 and NLL upper < .02.
An inconclusive interval means insufficient evidence. Do not select a favorable
calibration variant after seeing results: raw and scaled are separate analyses.

Training cost records real row/candidate/token counts, updates, device-timeline
span, wall time, trainable/frozen parameters, peak allocated memory and actual
loops. Device timeline includes host gaps and is not active GPU kernel time.
Checkpoint recomputation and cached head training have different costs; equal
steps are not equal FLOPs. FLOPs are not measured by these scripts.
The protocol also references original frozen-cache extraction time, including
its reused-candidate counts; this is historical incremental extraction cost,
not a complete cold-start estimate for head-only training.
Inference measures full decision rows using a fixed first-32 validation subset,
one warmup pass and three timed passes, including forward and head on pretokenized
inputs. Compare actual latency; do not infer speedups from offline selected depth.

Keep only final head, adapter (LoRA only, no base weights), diagnostic probe,
predictions, protocol/metrics under `<out-dir>/results/`. LoRA-recomputed reusable
features belong in `<out-dir>/cache/`; frozen heads reuse Step1 caches. There are
no per-step checkpoints or automatic tensor dumps. Logs contain concise progress.
Delete smoke outputs after verifying save/load and comparisons.

Single-seed, source-balanced choice results remain development evidence; the
retrieval control has few groups and is not a matched causal control. Step2 does
not establish multi-loop supervision, adaptive-exit gains, OOD robustness or
RLCD replication.

## Script validation

All four conditions passed a one-update smoke on eight rows/split. The checks
covered prompt/data/model hashes, actual two/four runtime loops, frozen-prefix
agreement, head and adapter updates, adapter save/reload, metrics and paired
comparison. In the four-loop LoRA check, non-reentrant checkpoint gradients
matched ordinary backward for all 96 trainable adapter tensors (maximum absolute
difference 0). Reloaded adapter hidden states differed from saved FP16 features
by at most .125, within the prescribed BF16 tolerance. Smoke metrics are not
scientific results; temporary tensors/adapters are removed after verification.
