"""Extract frozen Ouro features using pinned Open-Jev candidate text."""

import argparse
import collections
import hashlib
import inspect
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from step1_common import OPENJEV_REVISION, PROMPT_VERSION, candidate_prompts, fingerprint


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rows", default="train=512,dev-cal=128,validation=256")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--verify-batching", action="store_true")
    parser.add_argument("--reuse-cache-dir")
    return parser.parse_args()


def dev_group(group_id):
    return int(hashlib.sha256(("loopjev-dev-cal-v1:" + group_id).encode()).hexdigest()[:8], 16) % 5 == 0


def select_rows(root, split, count, seed):
    actual_split = "train" if split == "dev-cal" else split
    paths = sorted((root / "data/community-hard-mix-v2-redistributable").glob(actual_split + "-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files for {actual_split}")
    columns = ["id", "group_id", "kind", "source", "question", "options", "target", "state_json"]
    records = parquet.read_table(paths, columns=columns).to_pylist()
    buckets = collections.defaultdict(list)
    for record in records:
        if record["kind"] != "choice":
            continue
        if actual_split == "train" and dev_group(record["group_id"]) != (split == "dev-cal"):
            continue
        buckets[record["source"]].append(record)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    sources = sorted(buckets)
    rng.shuffle(sources)
    selected, groups = [], set()
    while sources and len(selected) < count:
        remaining = []
        for source in sources:
            bucket = buckets[source]
            while bucket:
                record = bucket.pop()
                if record["group_id"] not in groups:
                    selected.append(record)
                    groups.add(record["group_id"])
                    break
            if bucket:
                remaining.append(source)
            if len(selected) == count:
                break
        sources = remaining
    return selected, {str(path): fingerprint(path) for path in paths}


def prepare(records, tokenizer, args):
    accepted, prompts, lengths, rejected = [], [], [], []
    for record in records:
        target = np.asarray(record["target"], dtype=float)
        if len(target) != len(record["options"]) or not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(), 1):
            raise ValueError(f"Invalid target: {record['id']}")
        raw_prompts = candidate_prompts({**record, "state": json.loads(record["state_json"])})
        permutation = list(range(len(raw_prompts)))
        row_seed = hashlib.sha256(f"{args.seed}:{record['id']}".encode()).hexdigest()
        random.Random(row_seed).shuffle(permutation)
        rendered = [tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompts[index]}], tokenize=False, add_generation_prompt=True
        ) for index in permutation]
        row_lengths = [len(ids) for ids in tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]]
        if max(row_lengths) > args.max_length:
            rejected.append({"id": record["id"], "source": record["source"], "max_tokens": max(row_lengths)})
            continue
        metadata = {key: record[key] for key in ["id", "group_id", "source", "kind"]}
        metadata.update({
            "options": [record["options"][index] for index in permutation],
            "target": [record["target"][index] for index in permutation],
            "permutation": permutation, "token_lengths": row_lengths,
            "prompt_sha256": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in rendered],
        })
        accepted.append(metadata)
        prompts.extend(rendered)
        lengths.extend(row_lengths)
    if not accepted:
        raise ValueError("No accepted rows after length filtering")
    return accepted, prompts, lengths, rejected


def batches(lengths, args):
    current = []
    for index in sorted(range(len(lengths)), key=lengths.__getitem__):
        if current and (len(current) >= args.batch_size or (len(current) + 1) * lengths[index] > args.token_budget):
            yield current
            current = []
        current.append(index)
    if current:
        yield current


@torch.inference_mode()
def forward(prompts, tokenizer, model):
    encoded = tokenizer(prompts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt").to(model.device)
    positions = encoded["attention_mask"].sum(-1) - 1
    batch_indices = torch.arange(len(prompts), device=model.device)
    output, hidden, gates = model.model(**encoded, use_cache=False)
    if len(hidden) != model.model.total_ut_steps or len(gates) != len(hidden):
        raise ValueError("Unexpected loop count")
    states = torch.stack([state[batch_indices, positions].float() for state in hidden], dim=1)
    gate_values = torch.stack([gate[batch_indices, positions, 0].float() for gate in gates], dim=1)
    if output.past_key_values is not None or not torch.isfinite(states).all() or not torch.isfinite(gate_values).all():
        raise ValueError("Invalid states, gates, or unexpected KV cache")
    return states, gate_values


def main():
    args = arguments()
    if min(args.seed + 1, args.batch_size, args.max_length, args.token_budget) < 1:
        raise ValueError("Invalid positive argument")
    requests = dict(item.split("=") for item in args.rows.split(","))
    if not set(requests).issubset({"train", "dev-cal", "validation"}):
        raise ValueError("Step 1 development accepts train/dev-cal/validation only")
    if any(int(count) < 1 for count in requests.values()):
        raise ValueError("Row counts must be positive")
    root, destination = Path(args.model).resolve(), Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / f"{split}.pt").exists() for split in requests):
        raise FileExistsError("Output cache exists; use a new out-dir")
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError("Ouro chat template missing")
    model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    yes, no = [tokenizer.encode(word, add_special_tokens=False) for word in ["Yes", "No"]]
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Yes/No must be single tokens")
    initial = (model.lm_head.weight[yes[0]].float() - model.lm_head.weight[no[0]].float()).detach()
    provenance = {
        "model": str(root), "kind": "choice", "seed": args.seed, "max_length": args.max_length,
        "total_ut_steps": model.model.total_ut_steps, "hidden_size": model.config.hidden_size,
        "prompt_format": PROMPT_VERSION, "openjev_revision": OPENJEV_REVISION,
        "model_sha256": {path.name: fingerprint(path) for path in sorted(root.glob("*.safetensors"))},
        "source_sha256": {path.name: fingerprint(path) for path in [root / "modeling_ouro.py", root / "config.json", root / "tokenizer.json", root / "tokenizer_config.json"]},
        "script_sha256": {path.name: fingerprint(path) for path in [Path(__file__), Path(__file__).with_name("step1_common.py")]},
        "packages": {package: version(package) for package in ["torch", "transformers", "numpy", "pyarrow"]},
        "loaded_model_source": inspect.getfile(type(model)), "yes_id": yes[0], "no_id": no[0],
        "sampling": "source-round-robin; one row per group; train dev-cal hash holdout 20%",
        "calibration": "dev-cal uses train groups; official calibration/test/ood unopened",
    }
    manifest, observed_groups = {}, set()
    for split, count in requests.items():
        started = time.perf_counter()
        records, data_hashes = select_rows(Path(args.data), split, int(count), args.seed)
        metadata, prompts, lengths, rejected = prepare(records, tokenizer, args)
        groups = {record["group_id"] for record in metadata}
        if groups & observed_groups:
            raise ValueError("Cross-split group leakage")
        observed_groups |= groups
        states = torch.empty((len(prompts), model.model.total_ut_steps, model.config.hidden_size), dtype=torch.float16)
        gates = torch.empty((len(prompts), model.model.total_ut_steps), dtype=torch.float16)
        zero_logits = torch.empty((len(prompts), model.model.total_ut_steps), dtype=torch.float32)
        reused = set()
        reuse_path = Path(args.reuse_cache_dir) / f"{split}.pt" if args.reuse_cache_dir else None
        reuse_hash = None
        if reuse_path is not None and reuse_path.exists():
            previous = torch.load(reuse_path, map_location="cpu", weights_only=True)
            for key in ["model_sha256", "source_sha256", "prompt_format", "total_ut_steps", "seed", "max_length"]:
                if previous["config"][key] != provenance[key]:
                    raise ValueError(f"Reuse mismatch: {key}")
            if previous["config"]["split"] != split or previous["config"]["data_sha256"] != data_hashes:
                raise ValueError("Reuse split/data mismatch")
            previous_rows = {record["id"]: index for index, record in enumerate(previous["rows"])}
            offset = 0
            for record in metadata:
                count = len(record["options"])
                old_index = previous_rows.get(record["id"])
                if old_index is not None:
                    old_record = previous["rows"][old_index]
                    if any(old_record[key] != record[key] for key in ["group_id", "prompt_sha256", "target", "options"]):
                        raise ValueError("Reuse row mismatch")
                    start, end = previous["row_offsets"][old_index:old_index + 2].tolist()
                    states[offset:offset + count] = previous["hidden"][start:end]
                    gates[offset:offset + count] = previous["gates"][start:end]
                    zero_logits[offset:offset + count] = previous["zero_logits"][start:end]
                    reused.update(range(offset, offset + count))
                offset += count
            reuse_hash = fingerprint(reuse_path)
            del previous
        pending = [index for index in range(len(prompts)) if index not in reused]
        for batch_number, local_indices in enumerate(batches([lengths[index] for index in pending], args)):
            indices = [pending[index] for index in local_indices]
            hidden, gate = forward([prompts[index] for index in indices], tokenizer, model)
            states[indices] = hidden.cpu().half()
            gates[indices] = gate.cpu().half()
            zero_logits[indices] = torch.einsum("clh,h->cl", hidden, initial).cpu()
            if batch_number % 200 == 0:
                print(f"{split}: batch={batch_number} candidates={len(indices)} max_tokens={max(lengths[index] for index in indices)}", flush=True)
        parity = None
        if args.verify_batching:
            indices = [int(np.argmin(lengths)), int(np.argmax(lengths))]
            batched, _ = forward([prompts[index] for index in indices], tokenizer, model)
            maximum = 0.0
            for position, index in enumerate(indices):
                single, _ = forward([prompts[index]], tokenizer, model)
                maximum = max(maximum, float((batched[position] - single[0]).abs().max()))
                if not torch.allclose(batched[position], single[0], atol=0.25, rtol=0.03):
                    raise ValueError(f"Batch/single parity failed: max_abs={maximum}")
            parity = {"max_abs": maximum, "atol": 0.25, "rtol": 0.03}
        offsets = [0]
        for record in metadata:
            offsets.append(offsets[-1] + len(record["options"]))
        report = {
            "split": split, "requested_rows": int(count), "sampled_rows": len(records), "accepted_rows": len(metadata),
            "rejected_rows": rejected, "candidates": len(prompts), "length_quantiles": dict(zip(["min", "p50", "p95", "max"], np.quantile(lengths, [0, .5, .95, 1]).tolist())),
            "sources": dict(collections.Counter(record["source"] for record in metadata)),
            "elapsed_seconds": time.perf_counter() - started, "batch_single_parity": parity,
            "reused_candidates": len(reused), "reused_cache_sha256": reuse_hash,
        }
        payload = {"hidden": states, "gates": gates, "zero_logits": zero_logits, "initial_head": initial.cpu(),
            "row_offsets": torch.tensor(offsets), "rows": metadata,
            "config": {**provenance, "split": split, "data_sha256": data_hashes}, "report": report}
        torch.save(payload, destination / f"{split}.pt")
        manifest[split] = report
        (destination / "manifest.json").write_text(json.dumps({"provenance": provenance, "splits": manifest}, ensure_ascii=False, indent=2) + "\n")
        print(f"{split}: rows={len(metadata)} candidates={len(prompts)} reused={len(reused)} rejected={len(rejected)} seconds={report['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
