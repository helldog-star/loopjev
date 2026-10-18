"""E2 multi-loop shared-head training, with the locked Step2 data/budget."""

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from step1_common import fingerprint
from step2_evaluate import evaluate, write_json
from step2_train import (
    SPLITS, set_seed, synchronize, row_schedule, decision_loss, validated_caches,
    prepare_inputs, load_model, verify_frozen_prefix, candidate_batches, last_states,
    extract, benchmark,
)

PROTOCOL_VERSION = "step3-multi-loop-v1"

def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["head", "lora"], default="lora")
    parser.add_argument("--loops", type=int, choices=[4], default=4)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-rows", type=int, default=16)
    parser.add_argument("--candidate-batch", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--lora-lr", type=float, default=0.0002)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--benchmark-rows", type=int, default=32)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--weighting", choices=["uniform", "late"], default="uniform")
    return parser.parse_args()


def loop_weights(weighting):
    return [0.25] * 4 if weighting == "uniform" else [0.1, 0.2, 0.3, 0.4]


def multi_loop_loss(scores, target, alpha):
    """scores[candidate, loop]; normalize across candidates independently per loop."""
    if scores.ndim != 2 or scores.shape[1] != len(alpha):
        raise ValueError("Expected candidate-by-loop scores")
    if any(value < 0 for value in alpha) or not np.isclose(sum(alpha), 1.0):
        raise ValueError("Loop weights must be nonnegative and sum to one")
    return sum(weight * decision_loss(scores[:, loop], target)
               for loop, weight in enumerate(alpha))


def train_condition(args, payload, tokens, tokenizer, model):
    weights = payload["initial_head"].float().cuda().clone().requires_grad_()
    parameters = [weights]
    parameter_groups = [{"params": [weights], "lr": args.head_lr}]
    if args.mode == "lora":
        model.train()
        adapter_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        parameters.extend(adapter_parameters)
        parameter_groups.append({"params": adapter_parameters, "lr": args.lora_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
    schedule = list(row_schedule(len(payload["rows"]), args.batch_rows, args.seed, args.steps))
    budget = {"optimizer_steps": args.steps, "decision_rows": 0, "candidates": 0,
              "unpadded_tokens": 0, "padded_tokens": 0, "runtime_loops": args.loops,
              "backbone_forward_in_training": args.mode == "lora"}
    curve = []
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    for update, row_indices in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        per_loop = np.zeros(args.loops)
        for row_index in row_indices:
            record = payload["rows"][row_index]
            start, end = payload["row_offsets"][row_index:row_index + 2].tolist()
            if args.mode == "head":
                features = payload["hidden"][start:end, :args.loops].float().cuda()
                scores = features @ weights
            else:
                parts = []
                for indices in candidate_batches(range(start, end), tokens, args):
                    states, _, padded = last_states(model, tokenizer, [tokens[index] for index in indices])
                    parts.append(states @ weights)
                    budget["padded_tokens"] += padded
                scores = torch.cat(parts)
            loss = multi_loop_loss(scores, record["target"], loop_weights(args.weighting)) / len(row_indices)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite decision loss")
            with torch.no_grad():
                per_loop += np.array([float(decision_loss(scores[:, loop], record["target"]))
                                      for loop in range(args.loops)]) / len(row_indices)
            loss.backward()
            total += float(loss.detach())
            budget["decision_rows"] += 1
            budget["candidates"] += end - start
            budget["unpadded_tokens"] += sum(record["token_lengths"])
        penalty = args.l2 * weights.square().sum()
        penalty.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 50 == 0 or update == args.steps:
            curve.append({"update": update, "decision_loss": total, "per_loop_decision_loss": per_loop.tolist(), "head_penalty": float(penalty.detach()), "gradient_norm": float(norm)})
            print(f"update={update}/{args.steps} decision_loss={total:.4f}", flush=True)
    end_event.record()
    synchronize()
    budget.update({"wall_seconds": time.perf_counter() - started,
        "cuda_event_span_seconds": start_event.elapsed_time(end_event) / 1000,
        "cuda_event_span_definition": "elapsed device timeline, including host gaps; not a FLOP count or active-kernel time",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "head_parameters": weights.numel(), "adapter_parameters": sum(parameter.numel() for parameter in parameters[1:]),
        "row_schedule_sha256": hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        "gradient_checkpointing": args.mode == "lora", "trainable_head_bias": False})
    budget["candidate_loop_equivalents"] = budget["candidates"] * args.loops
    budget["token_loop_equivalents"] = budget["unpadded_tokens"] * args.loops
    budget["compute_proxy_limitation"] = "loop equivalents omit attention length, backward and checkpoint recomputation; not FLOPs"
    if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise ValueError("Frozen model parameter received a gradient")
    if not torch.isfinite(weights).all() or torch.equal(weights.detach().cpu(), payload["initial_head"].float()):
        raise ValueError("Head did not update correctly")
    return weights.detach().cpu(), curve, budget


def main():
    args = arguments()
    if min(args.steps, args.batch_rows, args.candidate_batch, args.token_budget, args.bootstrap,
           args.benchmark_rows, args.benchmark_repeats) < 1 or min(args.head_lr, args.lora_lr) <= 0 or args.l2 < 0 or args.seed < 0:
        raise ValueError("Invalid training arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("Step3 requires a CUDA GPU")
    if args.smoke and args.steps > 2:
        raise ValueError("Smoke must specify --steps 1 or 2")
    root = Path(args.out_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new empty out-dir")
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_seed(args.seed)
    caches = validated_caches(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_sets = prepare_inputs(args, caches, tokenizer)
    protocol = {"version": PROTOCOL_VERSION, "args": vars(args), "rows": {split: len(payload["rows"]) for split, payload in caches.items()},
        "supervision": "normalized multi-loop row-average soft CE + 0.1 Brier; L2 once on raw shared head; no head bias",
        "alpha": loop_weights(args.weighting), "head_method": "multi_loop_head",
        "initialization": "original bare Yes-minus-No output rows; independent original Ouro per condition, no head warm-start",
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "targets": ["q_proj", "v_proj"]},
        "probe": {"steps": 300, "lr": 0.01, "l2": 0.01, "train_only_standardization": True},
        "evaluation": "raw and train-derived dev-cal temperatures; official calibration/test/ood unopened",
        "limitations": ["single-seed development; previously inspected validation", "source-balanced choice only",
                        "small retrieval group is not a matched causal control", "equal updates do not imply equal FLOPs"],
        "source_cache_sha256": {split: fingerprint(Path(args.cache_dir) / f"{split}.pt") for split in SPLITS},
        "original_cache_extraction": {split: {key: caches[split]["report"].get(key) for key in
            ["elapsed_seconds", "candidates", "reused_candidates"]} for split in SPLITS},
        "model_provenance": caches["train"]["config"],
        "scripts_sha256": {name: fingerprint(Path(__file__).with_name(name)) for name in
            ["step3_train.py", "step3_compare.py", "step2_train.py", "step2_evaluate.py", "step2_compare.py", "step1_common.py", "step1_finish.py", "step1_probe.py"]},
        "packages": {package: version(package) for package in ["torch", "transformers", "peft", "numpy", "pyarrow"]}}
    protocol["gpu"] = {"name": torch.cuda.get_device_name(), "memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    write_json(results / "protocol.json", protocol)
    model = load_model(args, adapter=args.mode == "lora")
    base_parameters = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    prefix_check = verify_frozen_prefix(model, tokenizer, prompt_sets["train"], caches["train"], args.loops)
    initial_trainable = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    extraction_cost = {}
    weights, curve, training_cost = train_condition(args, caches["train"], prompt_sets["train"], tokenizer, model)
    training_cost["frozen_backbone_and_lm_head_parameters"] = base_parameters
    if args.mode == "lora":
        updated = [name for name, parameter in model.named_parameters()
                   if name in initial_trainable and not torch.equal(parameter.detach().cpu(), initial_trainable[name])]
        if not updated:
            raise ValueError("LoRA did not update")
        training_cost["changed_adapter_tensors"] = len(updated)
        model.save_pretrained(results / "adapter", safe_serialization=True, save_embedding_layers=False)
        fresh = {}
        adapter_hashes = {path.name: fingerprint(path) for path in (results / "adapter").iterdir() if path.is_file()}
        for split in SPLITS:
            fresh[split], extraction_cost[split] = extract(model, tokenizer, prompt_sets[split], caches[split], args)
            fresh[split]["config"]["adapter_sha256"] = adapter_hashes
            fresh[split]["config"]["step3_supervision"] = {"alpha": loop_weights(args.weighting), "version": PROTOCOL_VERSION}
            fresh[split]["config"]["step3_scripts_sha256"] = protocol["scripts_sha256"]
            fresh[split]["report"] = {"rows": len(fresh[split]["rows"]), **extraction_cost[split]}
        caches = fresh
    if args.mode == "lora":
        cache_root = root / "cache"
        cache_root.mkdir()
        for split, payload in caches.items():
            payload["config"]["source_cache_sha256"] = protocol["source_cache_sha256"][split]
            torch.save(payload, cache_root / f"{split}.pt")
    latency = benchmark(model, tokenizer, prompt_sets["validation"], caches["validation"], weights, args)
    if args.mode == "head":
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Head-only model must remain frozen")
    del initial_trainable, model
    torch.cuda.empty_cache()
    methods, predictions, probe = evaluate(caches, weights, args)
    methods["multi_loop_head"] = methods.pop("final_only_head")
    predictions["multi_loop_head"] = predictions.pop("final_only_head")
    torch.save({"weights": weights, "loops": args.loops, "protocol": protocol}, results / "head.pt")
    torch.save(probe, results / "probe.pt")
    torch.save(predictions, results / "validation_predictions.pt")
    write_json(results / "metrics.json", {"status": "SMOKE ONLY" if args.smoke else "completed single-seed development condition",
        "protocol_version": PROTOCOL_VERSION, "mode": args.mode, "loops": args.loops,
        "training_curve": curve, "training_cost": training_cost, "extraction_cost": extraction_cost,
        "frozen_feature_check": prefix_check, "latency": latency, "methods": methods})
    print(f"Completed {args.mode} loops={args.loops}; results={results}", flush=True)


if __name__ == "__main__":
    main()
