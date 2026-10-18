"""Metrics for Step 2 final-only conditions."""

import collections
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import logsumexp

from step1_common import fit_temperature, summarize
from step1_finish import compact, selection_group, task_group
from step1_probe import scores_by_loop, train_probe


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def row_scores(cache, weights, loops):
    flat = torch.einsum("clh,h->cl", cache["hidden"].float(), weights.float()).numpy()
    offsets = cache["row_offsets"].tolist()
    return [[flat[start:end, loop].astype(np.float64) for start, end in zip(offsets[:-1], offsets[1:])]
            for loop in range(loops)]


def paired_difference(scores, baseline, records, repeats, seed=0):
    clusters = collections.defaultdict(list)
    nll, accuracy, hard = [], [], []
    for index, (value, reference, record) in enumerate(zip(scores, baseline, records, strict=True)):
        target = np.asarray(record["target"], dtype=np.float64)
        value, reference = np.asarray(value), np.asarray(reference)
        nll.append((logsumexp(value) - target @ value) - (logsumexp(reference) - target @ reference))
        accuracy.append(float(value.argmax() == target.argmax()) - float(reference.argmax() == target.argmax()))
        hard.append(np.count_nonzero(target) == 1 and np.isclose(target.max(), 1))
        clusters[record["group_id"]].append(index)
    nll, accuracy, hard = np.array(nll), np.array(accuracy), np.array(hard)
    if not len(records):
        raise ValueError("Empty paired comparison")
    totals = np.array([[nll[members].sum(), len(members), (accuracy[members] * hard[members]).sum(), hard[members].sum()]
                       for members in clusters.values()])
    generator = np.random.default_rng(seed)
    sampled = totals[generator.integers(len(totals), size=(repeats, len(totals)))].sum(1)
    nll_draws = sampled[:, 0] / sampled[:, 1]
    valid = sampled[:, 3] > 0
    accuracy_draws = sampled[valid, 2] / sampled[valid, 3]
    return {"replicates": repeats, "groups": len(clusters), "rows": len(records),
        "nll_difference": float(nll.mean()), "nll_95ci": np.quantile(nll_draws, [.025, .975]).tolist(),
        "nll_one_sided_95_upper": float(np.quantile(nll_draws, .95)),
        "accuracy_difference": float(accuracy[hard].mean()) if hard.any() else None,
        "accuracy_95ci": np.quantile(accuracy_draws, [.025, .975]).tolist() if len(accuracy_draws) else None,
        "accuracy_one_sided_95_lower": float(np.quantile(accuracy_draws, .05)) if len(accuracy_draws) else None}


def evaluate(payloads, weights, args):
    loops = args.loops
    probe = train_probe(payloads["train"], SimpleNamespace(steps=300, lr=0.01, l2=0.01,
        brier_weight=0.1, device="cuda"))
    final_scores = {split: row_scores(payload, weights, loops) for split, payload in payloads.items()}
    probe_scores = {split: scores_by_loop(payload, probe) for split, payload in payloads.items()}
    records = payloads["validation"]["rows"]
    partitions = {
        "validation_all": list(range(len(records))),
        "validation_select": [index for index, record in enumerate(records) if selection_group(record)],
        "validation_check": [index for index, record in enumerate(records) if not selection_group(record)],
    }
    methods, predictions = {}, {}
    for name, values in [("final_only_head", final_scores), ("independent_probe", probe_scores)]:
        temperatures = [fit_temperature(values["dev-cal"][loop], payloads["dev-cal"]["rows"])
                        for loop in range(loops)]
        loop_results = []
        for loop in range(loops):
            groups = {}
            for partition, indices in partitions.items():
                members = {"overall": indices}
                for group in ["reasoning", "retrieval_matching", "other"]:
                    members[group] = [index for index in indices if task_group(records[index]) == group]
                groups[partition] = {}
                for group, selected in members.items():
                    if not selected:
                        groups[partition][group] = {"rows": 0}
                        continue
                    subset = [records[index] for index in selected]
                    logits = [values["validation"][loop][index] for index in selected]
                    result = {"rows": len(subset), "groups": len({record["group_id"] for record in subset}),
                        "raw": compact(summarize(logits, subset)),
                        "temperature_scaled": compact(summarize(logits, subset, temperatures[loop]["temperature"]))}
                    baseline = [values["validation"][loops - 1][index] for index in selected]
                    result["paired_bootstrap_raw"] = paired_difference(logits, baseline, subset, args.bootstrap)
                    result["paired_bootstrap_raw"]["reference"] = f"same method loop {loops} raw"
                    scaled = [value / temperatures[loop]["temperature"] for value in logits]
                    scaled_baseline = [value / temperatures[-1]["temperature"] for value in baseline]
                    result["paired_bootstrap_scaled"] = paired_difference(scaled, scaled_baseline, subset, args.bootstrap)
                    by_source = collections.defaultdict(list)
                    for index in selected:
                        by_source[records[index]["source"]].append(index)
                    result["per_source"] = {
                        source: {"raw": compact(summarize([values["validation"][loop][index] for index in members],
                                                          [records[index] for index in members])),
                                 "temperature_scaled": compact(summarize([values["validation"][loop][index] for index in members],
                                                          [records[index] for index in members], temperatures[loop]["temperature"]))}
                        for source, members in by_source.items()}
                    result["source_macro"] = {
                        scale: {metric: float(np.mean([item[scale][metric] for item in result["per_source"].values() if metric in item[scale]]))
                                for metric in ["accuracy", "nll", "brier", "aurc"]
                                if any(metric in item[scale] for item in result["per_source"].values())}
                        for scale in ["raw", "temperature_scaled"]}
                    groups[partition][group] = result
            full = values["validation"][loop]
            soft_indices = [index for index, record in enumerate(records)
                            if not (np.count_nonzero(record["target"]) == 1 and np.isclose(max(record["target"]), 1))]
            soft_metrics = compact(summarize([values["validation"][loop][index] for index in soft_indices],
                                            [records[index] for index in soft_indices])) if soft_indices else {"rows": 0}
            loop_results.append({"loop": loop + 1, "temperature_fit": temperatures[loop], "partitions": groups,
                "train_raw": compact(summarize(values["train"][loop], payloads["train"]["rows"])),
                "soft_label_validation_raw": soft_metrics,
                "risk_coverage": {"raw": compact(summarize(full, records).get("risk_coverage", [])),
                    "scaled": compact(summarize(full, records, temperatures[loop]["temperature"]).get("risk_coverage", []))}})
        methods[name] = loop_results
        predictions[name] = {"logits": values["validation"], "records": records,
            "temperatures": temperatures}
    return methods, predictions, probe
