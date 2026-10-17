"""Step 0 smoke test for LoopJev.

Checks (see ../plan.md, Step 0):
  1. Ouro config: total_ut_steps.
  2. Base model forward exposes per-loop hidden states (hidden_states_list) and gates (gate_list).
  3. Last-loop hidden state through lm_head matches the full model's logits (i.e. states are post-norm).
  4. Per-loop next-token predictions look sane; Yes-vs-No margin per loop on a toy question.
  5. "Yes"/"No" tokenization variants are single tokens.
  6. Open-Jev dataset splits, row format, and kind distribution.

Usage:
  python step0_smoke.py --model /path/to/Ouro-1.4B --data /path/to/Open-Jev-v1.1
"""

import argparse
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def describe(obj, name="out", depth=0, max_depth=2):
    """Print the structure of a model output (tensors, lists, tuples, ModelOutput)."""
    pad = "  " * depth
    if torch.is_tensor(obj):
        print(f"{pad}{name}: Tensor{tuple(obj.shape)} {obj.dtype}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__}[len={len(obj)}]")
        if depth < max_depth:
            for i, x in enumerate(obj[:6]):
                describe(x, f"[{i}]", depth + 1, max_depth)
    elif hasattr(obj, "keys"):
        print(f"{pad}{name}: {type(obj).__name__} keys={list(obj.keys())}")
        if depth < max_depth:
            for k in obj.keys():
                describe(obj[k], k, depth + 1, max_depth)
    else:
        print(f"{pad}{name}: {type(obj).__name__}")


def find_loop_lists(out, n_loops, hidden_size):
    """Locate per-loop hidden states and gates in the base-model output.

    Prefers named attributes; falls back to scanning tuple elements for a list of
    n_loops tensors whose last dim is hidden_size (states) or 1 (gates).
    """
    hs = getattr(out, "hidden_states_list", None)
    gates = getattr(out, "gate_list", None)
    if hs is not None:
        return hs, gates
    items = out if isinstance(out, (list, tuple)) else list(out.values()) if hasattr(out, "values") else []
    for x in items:
        if isinstance(x, (list, tuple)) and len(x) == n_loops and all(torch.is_tensor(t) for t in x):
            if x[0].shape[-1] == hidden_size and hs is None:
                hs = x
            elif x[0].shape[-1] == 1 and gates is None:
                gates = x
    return hs, gates


def check_tokens(tok):
    print("\n== Yes/No tokenization ==")
    for w in ["Yes", "No", " Yes", " No", "yes", "no", " yes", " no"]:
        ids = tok.encode(w, add_special_tokens=False)
        flag = "single" if len(ids) == 1 else "MULTI"
        print(f"  {w!r:8} -> {ids} [{flag}]")


@torch.no_grad()
def check_model(args, tok, model):
    cfg = model.config
    n_loops = getattr(cfg, "total_ut_steps", None)
    print(f"\n== Config ==\n  total_ut_steps={n_loops} hidden_size={cfg.hidden_size} vocab={cfg.vocab_size}")

    base = model.model
    enc = tok(args.prompt, return_tensors="pt").to(model.device)
    out = base(**enc)
    print("\n== Base forward output structure ==")
    describe(out)

    hs, gates = find_loop_lists(out, n_loops, cfg.hidden_size)
    assert hs is not None, "could not find per-loop hidden states; inspect structure above"
    assert len(hs) == n_loops, f"expected {n_loops} loops, got {len(hs)}"
    print(f"\n  per-loop states: {len(hs)} x {tuple(hs[0].shape)}; gates: {None if gates is None else (len(gates), tuple(gates[0].shape))}")

    # Post-norm check: last loop through lm_head should equal full-model logits.
    full = model(**enc).logits[:, -1].float()
    last = model.lm_head(hs[-1][:, -1]).float()
    print(f"\n== Post-norm check ==\n  max|lm_head(h_last) - logits| = {(full - last).abs().max().item():.4g}")

    print("\n== Per-loop next-token top-5 (prompt) ==")
    for i, h in enumerate(hs):
        top = model.lm_head(h[:, -1]).float().softmax(-1).topk(5)
        toks = [tok.decode([t]) for t in top.indices[0].tolist()]
        probs = [round(p, 3) for p in top.values[0].tolist()]
        g = "" if gates is None else f" gate_sigmoid={torch.sigmoid(gates[i][0, -1].float()).item():.3f}"
        print(f"  loop {i + 1}: {list(zip(toks, probs))}{g}")

    # Yes-vs-No margin per loop on toy questions with known answers.
    yes_id = tok.encode(" Yes", add_special_tokens=False)[-1]
    no_id = tok.encode(" No", add_special_tokens=False)[-1]
    print("\n== Per-loop logit(Yes) - logit(No) ==")
    for q, ans in [("Is 7 a prime number?", "Yes"), ("Is 9 a prime number?", "No"), ("Is Paris the capital of France?", "Yes")]:
        e = tok(f"Question: {q}\nAnswer (Yes or No):", return_tensors="pt").to(model.device)
        loop_h, _ = find_loop_lists(base(**e), n_loops, cfg.hidden_size)
        margins = []
        for h in loop_h:
            logit = model.lm_head(h[:, -1]).float()[0]
            margins.append(round((logit[yes_id] - logit[no_id]).item(), 2))
        print(f"  [{ans}] {q:35} {margins}")


def check_data(args):
    from datasets import load_dataset, load_from_disk

    print("\n== Open-Jev dataset ==")
    try:
        ds = load_from_disk(args.data)
    except Exception:
        ds = load_dataset(args.data)
    for split, d in ds.items():
        print(f"  {split}: {len(d)} rows, columns={d.column_names}")
    split = "train" if "train" in ds else next(iter(ds))
    row = ds[split][0]
    print(f"\n  sample row ({split}):")
    for k, v in row.items():
        s = repr(v)
        print(f"    {k}: {s[:300]}{'...' if len(s) > 300 else ''}")
    if "kind" in ds[split].column_names:
        n = min(len(ds[split]), 20000)
        print(f"\n  kind counts (first {n} of {split}): {Counter(ds[split].select(range(n))['kind'])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="local path or HF id of Ouro-1.4B")
    p.add_argument("--data", default=None, help="local path or HF id of Open-Jev-v1.1")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=getattr(torch, args.dtype), device_map="cuda"
    ).eval()
    model.config.use_cache = False

    check_tokens(tok)
    check_model(args, tok, model)
    if args.data:
        check_data(args)


if __name__ == "__main__":
    main()
