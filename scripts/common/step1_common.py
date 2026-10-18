"""Shared input and metric definitions for frozen LoopJev experiments."""

import hashlib
import json

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


OPENJEV_REVISION = "bd4118882f733574a3250a4b65fe4d884130c08b"
PROMPT_VERSION = "openjev-candidate-prompts+ouro-chat-v1"


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def candidate_prompts(record):
    prefix = f"Context:\n{render(record['state'])}\n\nQuestion: {render(record['question'])}\n"
    if record["kind"] == "noul":
        return [prefix + "Is the answer to this question yes? Answer Yes or No."]
    return [
        prefix + f"Proposed answer: {render(option)}\nIs this proposed answer correct? Answer Yes or No."
        for option in record["options"]
    ]


def summarize(logits, records, temperature=1.0):
    nll, brier, confidence, correct = [], [], [], []
    for scores, record in zip(logits, records, strict=True):
        target = np.asarray(record["target"], dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64) / temperature
        log_probabilities = scores - logsumexp(scores)
        probabilities = np.exp(log_probabilities)
        nll.append(-float(target @ log_probabilities))
        brier.append(float(np.square(probabilities - target).sum()))
        if np.count_nonzero(target) == 1 and np.isclose(target.max(), 1.0):
            confidence.append(float(probabilities.max()))
            correct.append(float(probabilities.argmax() == target.argmax()))
    result = {"rows": len(records), "hard_rows": len(correct), "nll": float(np.mean(nll)), "brier": float(np.mean(brier))}
    if not correct:
        return result
    confidence, correct = np.asarray(confidence), np.asarray(correct)
    bins = np.minimum((confidence * 15).astype(int), 14)
    equal_width = 0.0
    for bin_index in range(15):
        members = bins == bin_index
        if members.any():
            equal_width += members.mean() * abs(correct[members].mean() - confidence[members].mean())
    equal_mass = 0.0
    for members in np.array_split(np.argsort(confidence, kind="stable"), min(15, len(correct))):
        equal_mass += len(members) / len(correct) * abs(correct[members].mean() - confidence[members].mean())
    order = np.argsort(-confidence, kind="stable")
    ordered_confidence = confidence[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_confidence) != 0) + 1]
    ends = np.r_[starts[1:], len(correct)]
    errors = 1.0 - correct[order]
    for start, end in zip(starts, ends, strict=True):
        errors[start:end] = errors[start:end].mean()
    risk = np.cumsum(errors) / np.arange(1, len(correct) + 1)
    points = np.unique(np.linspace(1, len(correct), min(50, len(correct))).astype(int))
    result.update({
        "accuracy": float(correct.mean()),
        "ece_15_equal_width": float(equal_width),
        "ece_15_equal_mass": float(equal_mass),
        "aurc": float(risk.mean()),
        "risk_coverage": [{"coverage": float(count / len(correct)), "risk": float(risk[count - 1])} for count in points],
    })
    return result


def fit_temperature(logits, records):
    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        losses = []
        for scores, record in zip(logits, records, strict=True):
            scaled = np.asarray(scores, dtype=np.float64) / temperature
            losses.append(logsumexp(scaled) - np.asarray(record["target"]) @ scaled)
        return float(np.mean(losses))
    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fit failed")
    return {"temperature": float(np.exp(result.x)), "log_bounds": [-4.0, 4.0], "bound_hit": bool(abs(result.x) > 3.95)}
