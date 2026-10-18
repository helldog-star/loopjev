"""Pair completed Step2 conditions with group bootstrap and fixed noninferiority bounds."""

import argparse
import json
from pathlib import Path

import torch

from step1_finish import selection_group, task_group
from step2_evaluate import paired_difference, write_json


def load_run(path, allow_smoke=False):
    root = Path(path) / "results"
    metrics = json.loads((root / "metrics.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if metrics["status"] == "SMOKE ONLY" and not allow_smoke:
        raise ValueError("Smoke artifacts cannot be treated as experimental evidence")
    return {"metrics": metrics, "protocol": protocol,
            "predictions": torch.load(root / "validation_predictions.pt", map_location="cpu", weights_only=False)}


def check_pair(candidate, reference):
    for key in ["version", "rows", "source_cache_sha256", "lora", "probe", "model_provenance", "scripts_sha256"]:
        if candidate["protocol"][key] != reference["protocol"][key]:
            raise ValueError(f"Conditions are not paired: {key}")
    for key in ["steps", "batch_rows", "head_lr", "lora_lr", "l2", "seed", "candidate_batch", "token_budget", "smoke"]:
        if candidate["protocol"]["args"][key] != reference["protocol"]["args"][key]:
            raise ValueError(f"Condition settings differ: {key}")
    if candidate["metrics"]["training_cost"]["row_schedule_sha256"] != reference["metrics"]["training_cost"]["row_schedule_sha256"]:
        raise ValueError("Training data order differs")


def compare(candidate, reference, repeats):
    check_pair(candidate, reference)
    result = {}
    for method in ["final_only_head", "independent_probe"]:
        current = candidate["predictions"][method]
        baseline = reference["predictions"][method]
        if current["records"] != baseline["records"]:
            raise ValueError("Validation rows, targets, or candidate ordering differ")
        records = current["records"]
        subsets = {"validation_all": list(range(len(records))),
                   "validation_check": [index for index, record in enumerate(records) if not selection_group(record)]}
        for group in ["reasoning", "retrieval_matching", "other"]:
            subsets["validation_check/" + group] = [index for index in subsets["validation_check"] if task_group(records[index]) == group]
        comparisons = [("native_final", len(current["logits"]) - 1, len(baseline["logits"]) - 1)]
        comparisons.extend((f"same_loop_{loop + 1}", loop, loop)
                           for loop in range(min(len(current["logits"]), len(baseline["logits"]))))
        result[method] = {}
        for name, candidate_loop, reference_loop in comparisons:
            result[method][name] = {}
            for subset_name, indices in subsets.items():
                if not indices:
                    continue
                subset = [records[index] for index in indices]
                group_result = {}
                for scale in ["raw", "temperature_scaled"]:
                    candidate_temperature = 1.0 if scale == "raw" else current["temperatures"][candidate_loop]["temperature"]
                    reference_temperature = 1.0 if scale == "raw" else baseline["temperatures"][reference_loop]["temperature"]
                    scores = [current["logits"][candidate_loop][index] / candidate_temperature for index in indices]
                    reference_scores = [baseline["logits"][reference_loop][index] / reference_temperature for index in indices]
                    metrics = paired_difference(scores, reference_scores, subset, repeats)
                    lower = metrics["accuracy_one_sided_95_lower"]
                    metrics["quality_noninferiority_at_preregistered_bounds"] = bool(
                        lower is not None and lower > -0.01 and metrics["nll_one_sided_95_upper"] < 0.02)
                    group_result[scale] = metrics
                result[method][name][subset_name] = group_result
    latency, baseline_latency = candidate["metrics"]["latency"], reference["metrics"]["latency"]
    if latency["row_ids"] != baseline_latency["row_ids"]:
        raise ValueError("Latency benchmark rows differ")
    if any(latency[key] != baseline_latency[key] for key in ["candidate_batch", "repeats", "warmup_passes", "scope"]):
        raise ValueError("Latency benchmark settings differ")
    if candidate["protocol"]["gpu"] != reference["protocol"]["gpu"]:
        raise ValueError("GPU hardware differs")
    result["mean_latency_ratio"] = latency["mean_seconds_per_decision"] / baseline_latency["mean_seconds_per_decision"]
    result["latency_limitations"] = "ratio of measured pass means; no significance or FLOP claim"
    result["training_cost"] = {"candidate": candidate["metrics"]["training_cost"],
                               "reference": reference["metrics"]["training_cost"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("Positive bootstrap count required")
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    candidate = load_run(args.candidate, args.allow_smoke)
    reference = load_run(args.reference, args.allow_smoke)
    result = compare(candidate, reference, args.bootstrap)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, {"status": "SMOKE ONLY" if args.allow_smoke else "single-seed development comparison",
        "direction": "candidate minus reference", "candidate": args.candidate, "reference": args.reference,
        "bounds": {"accuracy_lower": -0.01, "nll_upper": 0.02},
        "limitations": "one-sided percentile group bootstrap; no multiplicity correction; noninferiority is not equivalence",
        "comparison": result})
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
