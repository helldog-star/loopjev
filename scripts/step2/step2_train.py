"""One reproducible final-only condition: frozen head or Ouro LoRA, two/four loops."""

import argparse
import hashlib
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from step1_common import candidate_prompts, fingerprint
from step1_probe import load_cache
from step2_evaluate import evaluate, write_json


SPLITS = ["train", "dev-cal", "validation"]
ROW_COUNTS = {"train": 4096, "dev-cal": 512, "validation": 2048}
PROTOCOL_VERSION = "step2-final-only-v1"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["head", "lora"], required=True)
    parser.add_argument("--loops", type=int, choices=[2, 4], required=True)
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
    return parser.parse_args()


def synchronize():
    torch.cuda.synchronize()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def row_schedule(row_count, batch_rows, seed, steps):
    rng = random.Random(seed)
    order, position = [], 0
    for _ in range(steps):
        batch = []
        while len(batch) < batch_rows:
            if position == len(order):
                order = list(range(row_count))
                rng.shuffle(order)
                position = 0
            amount = min(batch_rows - len(batch), len(order) - position)
            batch.extend(order[position:position + amount])
            position += amount
        yield batch


def decision_loss(scores, target):
    target = scores.new_tensor(target)
    log_probabilities = scores.log_softmax(0)
    return -(target * log_probabilities).sum() + 0.1 * (log_probabilities.exp() - target).square().sum()


def validated_caches(args):
    root = Path(args.cache_dir)
    caches = {split: load_cache(root / f"{split}.pt") for split in SPLITS}
    provenance = caches["train"]["config"]
    for key in ["model_sha256", "source_sha256"]:
        for name, expected in provenance[key].items():
            if fingerprint(Path(args.model) / name) != expected:
                raise ValueError(f"Original model/tokenizer changed: {name}")
    observed = set()
    for split, payload in caches.items():
        if payload["config"]["split"] != split or payload["config"]["total_ut_steps"] != 4:
            raise ValueError("Expected original four-loop frozen Step1 features")
        if len(payload["rows"]) != ROW_COUNTS[split]:
            raise ValueError("Unexpected locked split size")
        for key in ["model_sha256", "source_sha256", "seed", "max_length", "prompt_format"]:
            if payload["config"][key] != provenance[key]:
                raise ValueError(f"Inconsistent cache: {key}")
        groups = {record["group_id"] for record in payload["rows"]}
        if observed & groups or len(groups) != len(payload["rows"]):
            raise ValueError("Repeated/leaking group")
        observed |= groups
    if args.smoke:
        for split, payload in caches.items():
            count = min(8, len(payload["rows"]))
            end = int(payload["row_offsets"][count])
            caches[split] = {**payload, "rows": payload["rows"][:count],
                "hidden": payload["hidden"][:end], "gates": payload["gates"][:end],
                "zero_logits": payload["zero_logits"][:end], "row_offsets": payload["row_offsets"][:count + 1]}
    return caches


def prepare_inputs(args, caches, tokenizer):
    prompt_sets = {}
    for split, payload in caches.items():
        actual_split = "train" if split == "dev-cal" else split
        data_root = Path(args.data) / "data/community-hard-mix-v2-redistributable"
        paths = sorted(data_root.glob(actual_split + "-*.parquet"))
        actual_hashes = {path.name: fingerprint(path) for path in paths}
        expected_hashes = {Path(path).name: digest for path, digest in payload["config"]["data_sha256"].items()}
        if actual_hashes != expected_hashes:
            raise ValueError("Dataset changed after Step1 extraction")
        columns = ["id", "group_id", "source", "kind", "state_json", "question", "options", "target"]
        selected_ids = {record["id"] for record in payload["rows"]}
        records = {}
        for record in parquet.read_table(paths, columns=columns).to_pylist():
            if record["id"] in selected_ids:
                if record["id"] in records:
                    raise ValueError("Duplicate selected id")
                records[record["id"]] = record
        input_ids = []
        for cached in payload["rows"]:
            record = records[cached["id"]]
            if any(record[key] != cached[key] for key in ["group_id", "source", "kind"]):
                raise ValueError("Metadata mismatch")
            permutation = cached["permutation"]
            if sorted(permutation) != list(range(len(record["options"]))):
                raise ValueError("Invalid cached permutation")
            if cached["target"] != [record["target"][index] for index in permutation]:
                raise ValueError("Target mismatch")
            if cached["options"] != [record["options"][index] for index in permutation]:
                raise ValueError("Candidate mismatch")
            raw = candidate_prompts({**record, "state": json.loads(record["state_json"])})
            rendered = [tokenizer.apply_chat_template([{"role": "user", "content": raw[index]}],
                tokenize=False, add_generation_prompt=True) for index in permutation]
            if [hashlib.sha256(prompt.encode()).hexdigest() for prompt in rendered] != cached["prompt_sha256"]:
                raise ValueError("Prompt hash mismatch")
            tokens = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
            if list(map(len, tokens)) != cached["token_lengths"]:
                raise ValueError("Token lengths changed")
            input_ids.extend(tokens)
        prompt_sets[split] = input_ids
    return prompt_sets


def load_model(args, adapter=False):
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    config.total_ut_steps = args.loops
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    if model.model.total_ut_steps != args.loops:
        raise ValueError("Runtime loop count mismatch")
    model.requires_grad_(False)
    if adapter:
        names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and name.endswith((".q_proj", ".v_proj"))]
        if len(names) != 2 * model.config.num_hidden_layers:
            raise ValueError("Unexpected LoRA target modules")
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
            bias="none", target_modules=["q_proj", "v_proj"]))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if any(parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()):
            raise ValueError("Unexpected trainable backbone weight")
    return model


@torch.no_grad()
def verify_frozen_prefix(model, tokenizer, tokens, template, loops):
    model.eval()
    indices = [0, min(range(len(tokens)), key=lambda index: len(tokens[index]))]
    actual, _, _ = last_states(model, tokenizer, [tokens[index] for index in indices])
    expected = template["hidden"][indices, :loops].float().cuda()
    maximum = float((actual - expected).abs().max())
    if not torch.allclose(actual, expected, atol=0.25, rtol=0.03):
        raise ValueError(f"Original frozen feature mismatch: {maximum}")
    return {"max_absolute_difference": maximum, "atol": 0.25, "rtol": 0.03,
            "meaning": "original two-loop runtime matches prefix of the original four-loop cache within BF16 tolerance"}


def decoder(model):
    return model.get_base_model().model if hasattr(model, "peft_config") else model.model


def candidate_batches(indices, tokens, args):
    current = []
    maximum = 0
    for index in indices:
        length = len(tokens[index])
        if length > args.token_budget:
            raise ValueError("Single candidate exceeds token budget")
        if current and (len(current) >= args.candidate_batch or (len(current) + 1) * max(maximum, length) > args.token_budget):
            yield current
            current, maximum = [], 0
        current.append(index)
        maximum = max(maximum, length)
    if current:
        yield current


def last_states(model, tokenizer, input_ids):
    encoded = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").to("cuda")
    positions = encoded["attention_mask"].sum(-1) - 1
    batch = torch.arange(len(input_ids), device="cuda")
    output, states, gates = decoder(model)(**encoded, use_cache=False)
    if output.past_key_values is not None or len(states) != decoder(model).total_ut_steps:
        raise ValueError("Unexpected cache or loop count")
    selected = torch.stack([state[batch, positions].float() for state in states], dim=1)
    gate_values = torch.stack([gate[batch, positions, 0].float() for gate in gates], dim=1)
    return selected, gate_values, encoded["input_ids"].numel()


@torch.no_grad()
def extract(model, tokenizer, tokens, template, args):
    model.eval()
    states = torch.empty((len(tokens), args.loops, template["hidden"].shape[-1]), dtype=torch.float16)
    gates = torch.empty((len(tokens), args.loops), dtype=torch.float16)
    padded_tokens = 0
    synchronize()
    started = time.perf_counter()
    for indices in candidate_batches(sorted(range(len(tokens)), key=lambda index: len(tokens[index])), tokens, args):
        hidden, gate, padded = last_states(model, tokenizer, [tokens[index] for index in indices])
        states[indices], gates[indices] = hidden.cpu().half(), gate.cpu().half()
        padded_tokens += padded
    synchronize()
    if not torch.isfinite(states).all() or not torch.isfinite(gates).all():
        raise ValueError("Non-finite extracted features")
    return {**template, "hidden": states, "gates": gates,
        "zero_logits": torch.einsum("clh,h->cl", states.float(), template["initial_head"].float()),
        "config": {**template["config"], "total_ut_steps": args.loops,
                   "step2_model": "LoRA adapted" if args.mode == "lora" else "original frozen"}}, {
        "wall_seconds": time.perf_counter() - started, "padded_tokens": padded_tokens,
        "candidates": len(tokens), "loops": args.loops}


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
        for row_index in row_indices:
            record = payload["rows"][row_index]
            start, end = payload["row_offsets"][row_index:row_index + 2].tolist()
            if args.mode == "head":
                features = payload["hidden"][start:end, args.loops - 1].float().cuda()
                scores = features @ weights
            else:
                parts = []
                for indices in candidate_batches(range(start, end), tokens, args):
                    states, _, padded = last_states(model, tokenizer, [tokens[index] for index in indices])
                    parts.append(states[:, -1] @ weights)
                    budget["padded_tokens"] += padded
                scores = torch.cat(parts)
            loss = decision_loss(scores, record["target"]) / len(row_indices)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite decision loss")
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
            curve.append({"update": update, "decision_loss": total, "head_penalty": float(penalty.detach()), "gradient_norm": float(norm)})
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


@torch.inference_mode()
def benchmark(model, tokenizer, tokens, payload, weights, args):
    model.eval()
    weights = weights.cuda()
    count = min(args.benchmark_rows, len(payload["rows"]))
    samples = []
    for repeat in range(args.benchmark_repeats + 1):
        synchronize()
        started = time.perf_counter()
        for row_index in range(count):
            start, end = payload["row_offsets"][row_index:row_index + 2].tolist()
            scores = []
            for indices in candidate_batches(range(start, end), tokens, args):
                states, _, _ = last_states(model, tokenizer, [tokens[index] for index in indices])
                scores.append(states[:, -1] @ weights)
            torch.cat(scores).softmax(0)
        synchronize()
        if repeat:
            samples.append(time.perf_counter() - started)
    return {"decision_rows": count, "row_ids": [record["id"] for record in payload["rows"][:count]],
        "candidate_batch": args.candidate_batch, "repeats": args.benchmark_repeats, "warmup_passes": 1,
        "seconds_per_pass": samples, "mean_seconds_per_decision": float(np.mean(samples) / count),
        "scope": "GPU model forward + decision head; pretokenized; no autoregressive generation",
        "runtime_loops": decoder(model).total_ut_steps}


def main():
    args = arguments()
    if min(args.steps, args.batch_rows, args.candidate_batch, args.token_budget, args.bootstrap,
           args.benchmark_rows, args.benchmark_repeats) < 1 or min(args.head_lr, args.lora_lr) <= 0 or args.l2 < 0 or args.seed < 0:
        raise ValueError("Invalid training arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("Step2 requires a CUDA GPU")
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
        "supervision": "final-loop only, row-average soft CE + 0.1 Brier; L2 on raw head; no head bias",
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
            ["step2_train.py", "step2_evaluate.py", "step2_compare.py", "step1_common.py", "step1_finish.py", "step1_probe.py"]},
        "packages": {package: version(package) for package in ["torch", "transformers", "peft", "numpy", "pyarrow"]}}
    protocol["gpu"] = {"name": torch.cuda.get_device_name(), "memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    write_json(results / "protocol.json", protocol)
    model = load_model(args, adapter=args.mode == "lora")
    base_parameters = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    prefix_check = verify_frozen_prefix(model, tokenizer, prompt_sets["train"], caches["train"], args.loops)
    initial_trainable = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    extraction_cost = {}
    if args.mode == "head" and args.loops == 2:
        caches = {split: {**payload, "hidden": payload["hidden"][:, :2], "gates": payload["gates"][:, :2],
                          "zero_logits": payload["zero_logits"][:, :2],
                          "config": {**payload["config"], "total_ut_steps": 2}} for split, payload in caches.items()}
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
            fresh[split]["config"]["step2_scripts_sha256"] = protocol["scripts_sha256"]
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
