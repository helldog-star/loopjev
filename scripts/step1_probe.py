"""Train independent frozen-Ouro probes and evaluate raw/dev-cal-scaled scores."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from step1_common import fit_temperature, summarize


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def load_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["config"]["kind"] != "choice":
        raise ValueError("The pilot supports choice only")
    if payload["row_offsets"][-1].item() != len(payload["hidden"]):
        raise ValueError("Invalid candidate offsets")
    if not torch.isfinite(payload["hidden"]).all():
        raise ValueError("Non-finite hidden state")
    for index, record in enumerate(payload["rows"]):
        count = int(payload["row_offsets"][index + 1] - payload["row_offsets"][index])
        if count != len(record["target"]):
            raise ValueError("Target/feature mismatch")
    return payload


def flatten_targets(payload):
    targets, row_indices = [], []
    for index, record in enumerate(payload["rows"]):
        targets.extend(record["target"])
        row_indices.extend([index] * len(record["target"]))
    return torch.tensor(targets), torch.tensor(row_indices, dtype=torch.int64)


def train_probe(payload, args):
    device = getattr(args, "device", "cpu")
    features = payload["hidden"].to(device=device, dtype=torch.float32)
    mean = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(0.05)
    features = ((features - mean) / scale).contiguous()
    targets, row_indices = flatten_targets(payload)
    targets, row_indices = targets.to(device), row_indices.to(device)
    loops, dimension = features.shape[1:]
    weights = torch.zeros(loops, dimension, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=args.lr)
    indices = row_indices[:, None].expand(-1, loops)
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad()
        scores = torch.einsum("clh,lh->cl", features, weights)
        maxima = scores.new_full((len(payload["rows"]), loops), -torch.inf)
        maxima.scatter_reduce_(0, indices, scores.detach(), reduce="amax", include_self=True)
        centered = scores - maxima[row_indices]
        sums = scores.new_zeros((len(payload["rows"]), loops))
        sums.scatter_add_(0, indices, centered.exp())
        log_probabilities = centered - sums[row_indices].log()
        cross_entropy = -(targets[:, None] * log_probabilities).sum() / (len(payload["rows"]) * loops)
        brier = (log_probabilities.exp() - targets[:, None]).square().sum() / (len(payload["rows"]) * loops)
        loss = cross_entropy + args.brier_weight * brier + args.l2 * weights.square().sum() / loops
        if not torch.isfinite(loss):
            raise ValueError("Non-finite probe loss")
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == args.steps - 1:
            losses.append({"step": step + 1, "loss": float(loss.detach()), "ce": float(cross_entropy.detach())})
    return {"weights": weights.detach().cpu(), "mean": mean.cpu(), "scale": scale.cpu(), "training_curve": losses}


def scores_by_loop(payload, probe=None):
    if probe is None:
        scores = payload["zero_logits"].numpy()
    else:
        standardized = (payload["hidden"].float() - probe["mean"]) / probe["scale"]
        scores = torch.einsum("clh,lh->cl", standardized, probe["weights"]).numpy()
    offsets = payload["row_offsets"].tolist()
    return [[scores[start:end, loop].astype(np.float64) for start, end in zip(offsets[:-1], offsets[1:], strict=True)] for loop in range(scores.shape[1])]


def grouped_metrics(logits, records, temperature):
    sources = collections.defaultdict(list)
    for index, record in enumerate(records):
        sources[record["source"]].append(index)
    per_source = {source: summarize([logits[index] for index in members], [records[index] for index in members], temperature) for source, members in sources.items()}
    macro = {}
    for metric in ["accuracy", "nll", "brier", "aurc"]:
        values = [item[metric] for item in per_source.values() if metric in item]
        if values:
            macro[metric] = float(np.mean(values))
    return {"overall": summarize(logits, records, temperature), "source_macro": macro, "per_source": per_source}


def bootstrap_difference(logits, reference, records, repeats):
    from scipy.special import logsumexp
    groups = collections.defaultdict(list)
    differences, hard = [], []
    for index, (scores, baseline, record) in enumerate(zip(logits, reference, records, strict=True)):
        target = np.asarray(record["target"])
        differences.append((logsumexp(scores) - target @ scores) - (logsumexp(baseline) - target @ baseline))
        hard.append(np.count_nonzero(target) == 1 and np.isclose(target.max(), 1))
        groups[record["group_id"]].append(index)
    accuracy = np.array([float(scores.argmax() == np.argmax(record["target"])) - float(baseline.argmax() == np.argmax(record["target"])) for scores, baseline, record in zip(logits, reference, records, strict=True)])
    differences, hard = np.asarray(differences), np.asarray(hard)
    clusters = list(groups.values())
    rng, nll_samples, accuracy_samples = np.random.default_rng(0), [], []
    for _ in range(repeats):
        selected = np.concatenate([clusters[index] for index in rng.integers(len(clusters), size=len(clusters))])
        nll_samples.append(differences[selected].mean())
        if hard[selected].any():
            accuracy_samples.append(accuracy[selected][hard[selected]].mean())
    return {
        "reference": "same method loop 4 raw", "replicates": repeats,
        "nll_difference": float(differences.mean()), "nll_95ci": np.quantile(nll_samples, [.025, .975]).tolist(),
        "accuracy_difference": float(accuracy[hard].mean()) if hard.any() else None,
        "accuracy_95ci": np.quantile(accuracy_samples, [.025, .975]).tolist() if accuracy_samples else None,
    }


def main():
    args = arguments()
    torch.set_num_threads(4)
    if min(args.steps, args.bootstrap) < 1 or args.lr <= 0 or args.l2 < 0:
        raise ValueError("Invalid optimization arguments")
    root, destination = Path(args.cache_dir), Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "metrics.json").exists():
        raise FileExistsError("Metrics exist; use a new out-dir")
    payloads = {split: load_cache(root / f"{split}.pt") for split in ["train", "dev-cal", "validation"]}
    observed = set()
    for split, payload in payloads.items():
        if payload["config"]["split"] != split:
            raise ValueError("Unexpected cache split")
        groups = {record["group_id"] for record in payload["rows"]}
        if observed & groups:
            raise ValueError("Train/dev-cal/validation group leakage")
        observed |= groups
        for key in ["model_sha256", "source_sha256", "prompt_format", "script_sha256", "total_ut_steps", "seed", "max_length"]:
            if payload["config"][key] != payloads["train"]["config"][key]:
                raise ValueError(f"Cache mismatch: {key}")
    probe = train_probe(payloads["train"], args)
    torch.save({**probe, "config": payloads["train"]["config"], "args": vars(args)}, destination / "probe.pt")
    report = {"status": "development pilot; not a test result or Go/No-go", "args": vars(args), "methods": {},
        "probe": "independent zero-initialized heads; train-only per-loop feature standardization; CE+0.1 Brier; L2",
        "temperature_fit": "dev-cal held out by group from original train; official calibration unopened",
        "confidence": "max candidate probability; hard-label AURC, ties averaged within equal-confidence blocks",
        "training_curve": probe["training_curve"], "provenance": payloads["train"]["config"]}
    predictions = {}
    for name, fitted in [("zero_shot_yes_minus_no", None), ("independent_probe", probe)]:
        calibrated = scores_by_loop(payloads["dev-cal"], fitted)
        validation = scores_by_loop(payloads["validation"], fitted)
        train_scores = scores_by_loop(payloads["train"], fitted)
        method = []
        for loop, (calibration_scores, scores) in enumerate(zip(calibrated, validation, strict=True)):
            temperature = fit_temperature(calibration_scores, payloads["dev-cal"]["rows"])
            method.append({"loop": loop + 1, "temperature_fit": temperature,
                "train_raw": summarize(train_scores[loop], payloads["train"]["rows"]),
                "validation_raw": grouped_metrics(scores, payloads["validation"]["rows"], 1),
                "validation_temperature_scaled": grouped_metrics(scores, payloads["validation"]["rows"], temperature["temperature"]),
                "paired_group_bootstrap_raw": bootstrap_difference(scores, validation[-1], payloads["validation"]["rows"], args.bootstrap)})
            print(name, "loop", loop + 1, "raw", method[-1]["validation_raw"]["overall"], flush=True)
        report["methods"][name] = method
        predictions[name] = {"logits": validation, "records": payloads["validation"]["rows"]}
    (destination / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    torch.save(predictions, destination / "validation_predictions.pt")
    print("Saved", destination / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
