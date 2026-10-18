"""Finish the frozen choice development study with a bounded probe search."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from step1_common import fingerprint, fit_temperature, summarize
from step1_probe import bootstrap_difference, grouped_metrics, load_cache, scores_by_loop, train_probe


PROTOCOL = {
    "version": "step1-finish-v1",
    "rows": {"train": 4096, "dev-cal": 512, "validation": 2048},
    "learning_curve": [512, 2048, 4096],
    "trials": [{"steps": steps, "l2": penalty} for steps in [100, 300] for penalty in [0.001, 0.01]],
    "fixed": {"lr": 0.01, "brier_weight": 0.1, "seed": 0},
    "selection": "minimum mean raw NLL across four loops on validation-select; one configuration for every loop and sample size",
    "validation_partition": "SHA256(loopjev-validation-select-v1:group_id) first 8 hex modulo 2; 0 select, 1 check",
    "temperature": "fit separately per loop on dev-cal only, after configuration selection",
    "groups": {
        "reasoning": ["reasoning-control-v1"],
        "retrieval_matching": ["email-selection-control-v1", "ir-control-v1"],
        "other": "all remaining sources, including multi-rule community tasks",
    },
    "limitations": [
        "development choice experiment, source-balanced rather than natural-frequency sampling",
        "validation was previously inspected in the pilot; validation-check is not an untouched test",
        "retrieval_matching includes explicit filtering rules; not a pure lookup or matched causal control",
        "one seed; tiny control-group counts may prevent a conclusion about H3",
        "official calibration, test, ood and external reasoning benchmarks remain unopened",
    ],
}


def task_group(record):
    for name in ["reasoning", "retrieval_matching"]:
        if record["source"] in PROTOCOL["groups"][name]:
            return name
    return "other"


def selection_group(record):
    digest = hashlib.sha256(("loopjev-validation-select-v1:" + record["group_id"]).encode()).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def subset_payload(payload, count):
    end = int(payload["row_offsets"][count])
    return {**payload, "rows": payload["rows"][:count], "hidden": payload["hidden"][:end],
            "zero_logits": payload["zero_logits"][:end], "gates": payload["gates"][:end],
            "row_offsets": payload["row_offsets"][:count + 1]}


def select_scores(scores, indices):
    return [scores[index] for index in indices]


def compact(value):
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items() if key != "risk_coverage"}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write-protocol", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    protocol_path = destination / "protocol.json"
    if args.write_protocol:
        if protocol_path.exists():
            raise FileExistsError(protocol_path)
        protocol_path.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2) + "\n")
        print("Protocol fixed", flush=True)
        return
    if not args.cache_dir or args.bootstrap < 1:
        raise ValueError("Cache and positive bootstrap count required")
    if json.loads(protocol_path.read_text()) != PROTOCOL:
        raise ValueError("Protocol mismatch")
    if any((destination / name).exists() for name in ["metrics.json", "probe.pt", "validation_predictions.pt"]):
        raise FileExistsError("Completed outputs already exist")
    torch.set_num_threads(4)
    torch.manual_seed(PROTOCOL["fixed"]["seed"])
    root = Path(args.cache_dir)
    payloads = {split: load_cache(root / f"{split}.pt") for split in PROTOCOL["rows"]}
    observed = set()
    for split, payload in payloads.items():
        if payload["config"]["split"] != split or len(payload["rows"]) != PROTOCOL["rows"][split]:
            raise ValueError("Unexpected split/count; inspect length filtering before changing protocol")
        groups = {record["group_id"] for record in payload["rows"]}
        if len(groups) != len(payload["rows"]) or observed & groups:
            raise ValueError("Repeated or leaking group")
        observed |= groups
        for key in ["model_sha256", "source_sha256", "prompt_format", "script_sha256", "total_ut_steps", "seed", "max_length"]:
            if payload["config"][key] != payloads["train"]["config"][key]:
                raise ValueError(f"Cache mismatch: {key}")
    validation_records = payloads["validation"]["rows"]
    partitions = {
        "validation_select": [index for index, record in enumerate(validation_records) if selection_group(record)],
        "validation_check": [index for index, record in enumerate(validation_records) if not selection_group(record)],
        "validation_all": list(range(len(validation_records))),
    }
    if not all(partitions.values()):
        raise ValueError("Empty validation partition")
    selection_records = [validation_records[index] for index in partitions["validation_select"]]
    trials = []
    best_probe, best_config, best_objective = None, None, float("inf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for configuration in PROTOCOL["trials"]:
        settings = SimpleNamespace(**PROTOCOL["fixed"], **configuration, device=device)
        probe = train_probe(payloads["train"], settings)
        scores = scores_by_loop(payloads["validation"], probe)
        metrics = [summarize(select_scores(loop, partitions["validation_select"]), selection_records) for loop in scores]
        objective = float(np.mean([item["nll"] for item in metrics]))
        trials.append({**configuration, "selection_mean_raw_nll": objective, "selection_per_loop": compact(metrics)})
        print("Trial", configuration, "selection_nll", round(objective, 6), flush=True)
        if objective < best_objective:
            best_probe, best_config, best_objective = probe, configuration, objective
    report = {
        "status": "completed frozen choice development study; not final test or multi-seed evidence",
        "protocol": PROTOCOL, "selected_configuration": best_config,
        "selection_trials": trials, "learning_curve": [], "methods": {},
        "partition_counts": {name: len(indices) for name, indices in partitions.items()},
        "cache_sha256": {split: fingerprint(root / f"{split}.pt") for split in payloads},
        "script_sha256": {name: fingerprint(Path(__file__).with_name(name)) for name in
                          ["step1_finish.py", "step1_probe.py", "step1_common.py", "step1_extract.py"]},
        "provenance": payloads["train"]["config"],
        "bootstrap": "paired group bootstrap within each subset; 95% percentile CI; descriptive, no multiplicity adjustment",
    }
    for count in PROTOCOL["learning_curve"]:
        training = subset_payload(payloads["train"], count)
        probe = best_probe if count == PROTOCOL["rows"]["train"] else train_probe(
            training, SimpleNamespace(**PROTOCOL["fixed"], **best_config, device=device))
        scores = {split: scores_by_loop(payloads[split], probe) for split in ["dev-cal", "validation"]}
        train_scores = scores_by_loop(training, probe)
        loops = []
        for loop in range(len(train_scores)):
            temperature = fit_temperature(scores["dev-cal"][loop], payloads["dev-cal"]["rows"])
            evaluation = {"loop": loop + 1, "temperature_fit": temperature,
                          "train_raw": summarize(train_scores[loop], training["rows"])}
            for name, indices in partitions.items():
                records = [validation_records[index] for index in indices]
                logits = select_scores(scores["validation"][loop], indices)
                evaluation[name] = {
                    "raw": summarize(logits, records),
                    "temperature_scaled": summarize(logits, records, temperature["temperature"]),
                }
            loops.append(compact(evaluation))
        report["learning_curve"].append({"train_rows": count, "loops": loops})
        print("Learning curve", count, "check_accuracy", [round(item["validation_check"]["raw"]["accuracy"], 4) for item in loops], flush=True)
    predictions = {}
    for name, probe in [("zero_shot_yes_minus_no", None), ("independent_probe", best_probe)]:
        scores = scores_by_loop(payloads["validation"], probe)
        calibration = scores_by_loop(payloads["dev-cal"], probe)
        method = []
        for loop, logits in enumerate(scores):
            temperature = fit_temperature(calibration[loop], payloads["dev-cal"]["rows"])
            result = {"loop": loop + 1, "temperature_fit": temperature, "partitions": {}}
            for partition_name, indices in partitions.items():
                members = {"overall": indices}
                for group in ["reasoning", "retrieval_matching", "other"]:
                    members[group] = [index for index in indices if task_group(validation_records[index]) == group]
                partition_metrics = {}
                for group, group_indices in members.items():
                    if not group_indices:
                        partition_metrics[group] = {"rows": 0}
                        continue
                    records = [validation_records[index] for index in group_indices]
                    values = select_scores(logits, group_indices)
                    baseline = select_scores(scores[-1], group_indices)
                    scaled = [value / temperature["temperature"] for value in values]
                    baseline_temperature = fit_temperature(calibration[-1], payloads["dev-cal"]["rows"])["temperature"]
                    scaled_baseline = [value / baseline_temperature for value in baseline]
                    scaled_bootstrap = bootstrap_difference(scaled, scaled_baseline, records, args.bootstrap)
                    scaled_bootstrap["reference"] = "same method loop 4 dev-cal temperature scaled"
                    partition_metrics[group] = {
                        "rows": len(records), "groups": len({record["group_id"] for record in records}),
                        "sources": dict(collections.Counter(record["source"] for record in records)),
                        "raw": summarize(values, records),
                        "temperature_scaled": summarize(values, records, temperature["temperature"]),
                        "paired_bootstrap_raw": bootstrap_difference(values, baseline, records, args.bootstrap),
                        "paired_bootstrap_scaled": scaled_bootstrap,
                    }
                result["partitions"][partition_name] = compact(partition_metrics)
            result["validation_per_source"] = compact(grouped_metrics(logits, validation_records, temperature["temperature"]))
            result["risk_coverage_all"] = {
                "raw": summarize(logits, validation_records).get("risk_coverage", []),
                "temperature_scaled": summarize(logits, validation_records, temperature["temperature"]).get("risk_coverage", []),
            }
            method.append(result)
        report["methods"][name] = method
        predictions[name] = {"logits": scores, "records": validation_records,
                             "temperatures": [item["temperature_fit"]["temperature"] for item in method]}
    torch.save({**best_probe, "configuration": {**PROTOCOL["fixed"], **best_config},
                "cache_sha256": report["cache_sha256"], "protocol": PROTOCOL}, destination / "probe.pt")
    torch.save(predictions, destination / "validation_predictions.pt")
    (destination / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Completed", destination / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
