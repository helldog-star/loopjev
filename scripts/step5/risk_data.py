"""Prepare isolated groups and extract compact frozen-E2 predictions for Step5C."""
import argparse
import collections
import hashlib
import json
import random
import re
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as parquet
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from risk_core import FEATURES, TAIL_DELTA, features_and_targets
from step1_common import fingerprint
from step1_extract import dev_group, prepare
from step2_evaluate import write_json
from step2_train import candidate_batches, last_states, load_model


COUNTS = {'risk_train': 2048, 'risk_val': 1024, 'policy_fit': 1024, 'final_check': 2048}
SEED = 17
PRIME = (1 << 61) - 1


def digest_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return ' '.join(unicodedata.normalize('NFKC', value).lower().split())


def identities(row):
    state = json.loads(row['state_json'])
    question = normalized(state) + '\n' + normalized(row['question'])
    content = question + '\n' + '\n'.join(sorted(normalized(option) for option in row['options']))
    return digest_text(question), content


def shingles(content):
    tokens = re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]|[^\w\s]', content)
    if len(tokens) < 5:
        return frozenset([int.from_bytes(hashlib.blake2b(content.encode(), digest_size=8).digest(), 'little') % PRIME])
    return frozenset(int.from_bytes(hashlib.blake2b(' '.join(tokens[index:index + 5]).encode(),
                         digest_size=8).digest(), 'little') % PRIME for index in range(len(tokens) - 4))


class NearDuplicateIndex:
    def __init__(self):
        generator = np.random.default_rng(SEED)
        self.multiply = generator.integers(1, 2 ** 30, size=32, dtype=np.uint64)
        self.add = generator.integers(0, 2 ** 30, size=32, dtype=np.uint64)
        self.buckets = collections.defaultdict(set)
        self.tokens = []
        self.owners = []

    def keys(self, tokens):
        values = np.array(list(tokens), dtype=np.uint64)
        signature = ((values[:, None] * self.multiply + self.add) % np.uint64(PRIME)).min(axis=0)
        return [(band, tuple(signature[band * 4:(band + 1) * 4])) for band in range(8)]

    def matches(self, tokens):
        candidates = set()
        for key in self.keys(tokens):
            candidates.update(self.buckets[key])
        return [self.owners[index] for index in candidates
                if len(tokens & self.tokens[index]) / len(tokens | self.tokens[index]) >= .9]

    def add_tokens(self, tokens, owner):
        index = len(self.tokens)
        self.tokens.append(tokens)
        self.owners.append(owner)
        for key in self.keys(tokens):
            self.buckets[key].add(index)


def read_rows(paths):
    return parquet.read_table(paths, columns=['id', 'group_id', 'kind', 'source', 'question',
                                             'options', 'target', 'state_json']).to_pylist()


def prepare_data(root, run):
    if root.exists():
        raise FileExistsError(root)
    e2_protocol = json.loads((run / 'results/protocol.json').read_text())
    if e2_protocol['args']['smoke'] or e2_protocol['args']['weighting'] != 'uniform':
        raise ValueError('Expected formal Uniform-4 model')
    old_rows = []
    for split in ['train', 'dev-cal', 'validation']:
        cache = torch.load(run / f'cache/{split}.pt', map_location='cpu', weights_only=False)
        old_rows.extend(cache['rows'])
        del cache
    excluded_groups = {row['group_id'] for row in old_rows}
    old_ids = {row['id'] for row in old_rows}
    data_path = Path(e2_protocol['args']['data']) / 'data/community-hard-mix-v2-redistributable'
    train_paths = sorted(data_path.glob('train-*.parquet'))
    validation_paths = sorted(data_path.glob('validation-*.parquet'))
    train = read_rows(train_paths)
    validation = [row for row in read_rows(validation_paths) if row['id'] in old_ids]
    old_content = [row for row in train if row['group_id'] in excluded_groups] + validation
    exact = set()
    near = NearDuplicateIndex()
    for row in old_content:
        key, content = identities(row)
        if key in exact:
            continue
        exact.add(key)
        near.add_tokens(shingles(content), 'old:' + row['group_id'])
    buckets = collections.defaultdict(list)
    counts = collections.Counter()
    for row in train:
        if row['kind'] != 'choice':
            continue
        counts['choice_rows'] += 1
        if row['group_id'] in excluded_groups or dev_group(row['group_id']):
            counts['old_or_calibration_reserved_rows'] += 1
            continue
        if len(row['options']) < 2:
            counts['single_candidate_rows'] += 1
            continue
        target = np.asarray(row['target'], dtype=float)
        if len(target) != len(row['options']) or not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(), 1):
            raise ValueError('Invalid target in pool')
        buckets[row['source']].append(row)
    generator = random.Random(SEED)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: row['id'])
        generator.shuffle(bucket)
    tokenizer = AutoTokenizer.from_pretrained(e2_protocol['args']['model'], trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selected = {name: [] for name in COUNTS}
    sources = sorted(buckets)
    generator.shuffle(sources)
    groups = set()
    while sources and any(len(selected[name]) < count for name, count in COUNTS.items()):
        remaining = []
        for source in sources:
            accepted = None
            while buckets[source]:
                row = buckets[source].pop()
                if row['group_id'] in groups:
                    continue
                key, content = identities(row)
                if key in exact:
                    counts['exact_input_rejected_rows'] += 1
                    continue
                token_set = shingles(content)
                if near.matches(token_set):
                    counts['near_input_rejected_rows'] += 1
                    continue
                try:
                    metadata, _, _, rejected = prepare([row], tokenizer, SimpleNamespace(seed=0, max_length=4096))
                except ValueError as error:
                    if str(error) != 'No accepted rows after length filtering':
                        raise
                    counts['overlength_rows'] += 1
                    continue
                if rejected:
                    counts['overlength_rows'] += 1
                    continue
                accepted = (row, key, token_set)
                break
            if buckets[source]:
                remaining.append(source)
            if accepted is None:
                continue
            available = [name for name in COUNTS if len(selected[name]) < COUNTS[name]]
            split = generator.choices(available, weights=[COUNTS[name] - len(selected[name]) for name in available])[0]
            row, key, token_set = accepted
            selected[split].append(row)
            groups.add(row['group_id'])
            exact.add(key)
            near.add_tokens(token_set, split + ':' + row['group_id'])
            if all(len(selected[name]) == count for name, count in COUNTS.items()):
                break
        sources = remaining
        if len(groups) and len(groups) % 1024 < len(sources):
            print(f'prepare: selected={len(groups)}/{sum(COUNTS.values())}', flush=True)
    if any(len(selected[name]) != count for name, count in COUNTS.items()):
        raise ValueError(f'Insufficient eligible groups: { {name: len(rows) for name, rows in selected.items()} }')
    results = root / 'results'
    results.mkdir(parents=True)
    manifest = root / 'cache/manifest'
    manifest.mkdir(parents=True)
    for name, rows in selected.items():
        write_json(manifest / f'{name}.json', rows)
    prediction_path = run / 'results/validation_predictions.pt'
    predictions = torch.load(prediction_path, map_location='cpu', weights_only=False)['multi_loop_head']
    artifacts = [run / 'results/protocol.json', run / 'results/head.pt', prediction_path]
    artifacts += sorted(path for path in (run / 'results/adapter').iterdir() if path.is_file())
    provenance = e2_protocol['model_provenance']
    template_counter = collections.defaultdict(collections.Counter)
    for name, rows in selected.items():
        for row in rows:
            template = re.sub(r'\d+', '<number>', normalized(row['question']))
            template_counter[digest_text(template)][name] += 1
    audit = {'counts': {str(key): int(value) for key, value in counts.items()}, 'excluded_old_groups': len(excluded_groups),
        'selected_groups': len(groups), 'partitions': {name: {'rows': len(rows),
            'hard_rows': int(sum(np.count_nonzero(row['target']) == 1 and np.isclose(max(row['target']), 1.) for row in rows)),
            'sources': dict(collections.Counter(row['source'] for row in rows))} for name, rows in selected.items()},
        'normalized_question_templates_shared_across_partitions': sum(len(value) > 1 for value in template_counter.values()),
        'template_note': 'Shared templates are audited, not a template-disjoint benchmark. Near-duplicate LSH is heuristic and can miss matches.'}
    write_json(results / 'data_audit.json', audit)
    spec = {'version': 'step5c-risk-v1', 'run': str(run), 'counts': COUNTS, 'seed': SEED,
        'e2_artifact_sha256': {str(path): fingerprint(path) for path in artifacts},
        'model': e2_protocol['args']['model'], 'model_sha256': provenance['model_sha256'],
        'model_source_sha256': provenance['source_sha256'],
        'temperature': [item['temperature'] for item in predictions['temperatures']],
        'data_sha256': {str(path): fingerprint(path) for path in train_paths + validation_paths},
        'manifest_sha256': {name: fingerprint(manifest / f'{name}.json') for name in COUNTS},
        'scripts_sha256': {path.name: fingerprint(path) for path in sorted(Path(__file__).parent.glob('risk_*.py'))},
        'features': FEATURES, 'tail_delta_nats': TAIL_DELTA,
        'ridge_alpha': [.1, 1., 10., 100.], 'logistic_C': [.1, 1., 10.],
        'selection': 'Risk-val chooses regularization; policy-fit chooses thresholds; final-check accessed only after models/policies locked',
        'sampling': 'seeded source round robin; one row/group; all dev-cal-reserved train groups excluded',
        'duplicates': 'normalized context+question exact exclusion; 5-token shingle Jaccard>=0.9 with 32-signature/8-band LSH candidate retrieval',
        'limits': ['single E2 seed', 'fresh groups sampled from train, not official test',
                   'source/template overlap remains; no template-disjoint generalization claim',
                   'approximate near-duplicate retrieval can miss matches', 'no real early stopping or measured acceleration']}
    write_json(results / 'protocol.json', spec)
    print(f'prepared independent groups: {COUNTS}; audit={results / "data_audit.json"}', flush=True)


def verify_sources(root):
    spec = json.loads((root / 'results/protocol.json').read_text())
    for group in ['e2_artifact_sha256', 'data_sha256']:
        for path, expected in spec[group].items():
            if fingerprint(Path(path)) != expected:
                raise ValueError(f'Source changed: {path}')
    for name, expected in spec['scripts_sha256'].items():
        if fingerprint(Path(__file__).parent / name) != expected:
            raise ValueError(f'Step5C script changed: {name}')
    for group in ['model_sha256', 'model_source_sha256']:
        for name, expected in spec[group].items():
            if fingerprint(Path(spec['model']) / name) != expected:
                raise ValueError(f'Model changed: {name}')
    return spec


@torch.inference_mode()
def verify_e2(model, tokenizer, weights, run, spec):
    previous = torch.load(run / 'results/validation_predictions.pt', map_location='cpu', weights_only=False)['multi_loop_head']
    paths = [Path(path) for path in spec['data_sha256'] if Path(path).name.startswith('validation-')]
    wanted = {row['id']: index for index, row in enumerate(previous['records'][:2])}
    rows = [row for row in read_rows(paths) if row['id'] in wanted]
    rows.sort(key=lambda row: wanted[row['id']])
    metadata, prompts, _, rejected = prepare(rows, tokenizer, SimpleNamespace(seed=0, max_length=4096))
    if rejected or len(metadata) != 2 or metadata != previous['records'][:2]:
        raise ValueError('E2 parity prompt mismatch')
    tokens = tokenizer(prompts, add_special_tokens=False, truncation=False)['input_ids']
    actual = []
    for token_ids in tokens:
        states, _, _ = last_states(model, tokenizer, [token_ids])
        actual.append((states[0] @ weights).cpu().numpy())
    expected = np.concatenate([np.stack([previous['logits'][loop][index] for loop in range(4)], axis=1)
                               for index in range(2)])
    actual = np.array(actual)
    if not np.allclose(actual, expected, atol=.5, rtol=.03):
        raise ValueError(f'E2 prediction parity failed: max_abs={np.abs(actual - expected).max()}')
    return {'rows': 2, 'max_logit_absolute_difference': float(np.abs(actual - expected).max()),
            'atol': .5, 'rtol': .03, 'purpose': 'Implementation identity check only, no new model selection'}


@torch.inference_mode()
def extract_partitions(root, partitions):
    spec = verify_sources(root)
    run = Path(spec['run'])
    if 'final_check' in partitions and not (root / 'results/locked.json').is_file():
        raise ValueError('Final-check cannot be extracted before lock')
    for name in partitions:
        if name not in COUNTS or (root / f'cache/{name}.npz').exists():
            raise ValueError(f'Unknown partition or existing cache: {name}')
    model = load_model(SimpleNamespace(model=spec['model'], loops=4), adapter=False)
    model = PeftModel.from_pretrained(model, run / 'results/adapter', is_trainable=False).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(spec['model'], trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    weights = torch.load(run / 'results/head.pt', map_location='cpu', weights_only=False)['weights'].float().cuda()
    parity = verify_e2(model, tokenizer, weights, run, spec)
    args = SimpleNamespace(candidate_batch=4, token_budget=4096)
    for partition in partitions:
        started = time.perf_counter()
        path = root / f'cache/manifest/{partition}.json'
        if fingerprint(path) != spec['manifest_sha256'][partition]:
            raise ValueError('Manifest changed')
        source = json.loads(path.read_text())
        metadata, prompts, lengths, rejected = prepare(source, tokenizer, SimpleNamespace(seed=0, max_length=4096))
        if rejected or len(metadata) != COUNTS[partition]:
            raise ValueError('Prepared partition changed')
        tokens = tokenizer(prompts, add_special_tokens=False, truncation=False)['input_ids']
        scores = np.empty((len(tokens), 4), dtype=np.float64)
        gates = np.empty((len(tokens), 4), dtype=np.float64)
        completed = 0
        checkpoint = 0
        for batch in candidate_batches(sorted(range(len(tokens)), key=lambda index: len(tokens[index])), tokens, args):
            hidden, gate, _ = last_states(model, tokenizer, [tokens[index] for index in batch])
            scores[batch] = (hidden @ weights).cpu().numpy()
            gates[batch] = gate.cpu().numpy()
            completed += len(batch)
            if completed >= checkpoint:
                print(f'{partition}: candidates={completed}/{len(tokens)}', flush=True)
                checkpoint += 2000
        offsets = np.r_[0, np.cumsum([len(row['target']) for row in metadata])]
        row_logits = [scores[start:end].T for start, end in zip(offsets[:-1], offsets[1:])]
        row_gates = [gates[start:end] for start, end in zip(offsets[:-1], offsets[1:])]
        values = features_and_targets(row_logits, row_gates, metadata, spec['temperature'])
        np.savez_compressed(root / f'cache/{partition}.npz', **values,
            ids=np.array([row['id'] for row in metadata]), groups=np.array([row['group_id'] for row in metadata]),
            sources=np.array([row['source'] for row in metadata]), lengths=np.array([sum(row['token_lengths']) for row in metadata]),
            candidate_counts=np.array([len(row['target']) for row in metadata]))
        write_json(root / f'results/extraction_{partition}.json', {'rows': len(metadata), 'candidates': len(tokens),
            'elapsed_seconds': time.perf_counter() - started, 'model_frozen': True,
            'existing_e2_parity': parity,
            'manifest_sha256': spec['manifest_sha256'][partition],
            'cache_sha256': fingerprint(root / f'cache/{partition}.npz'),
            'length_quantiles': np.quantile(lengths, [0, .5, .95, 1]).tolist()})
        print(f'{partition}: saved compact six-feature cache, seconds={time.perf_counter() - started:.1f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'extract'])
    parser.add_argument('--root', default='artifacts/step5/risk_uniform4_v1')
    parser.add_argument('--run', default='artifacts/step3/uniform4')
    parser.add_argument('--partitions', nargs='+', choices=list(COUNTS), default=['risk_train', 'risk_val', 'policy_fit'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare_data(Path(args.root), Path(args.run))
    else:
        extract_partitions(Path(args.root), args.partitions)


if __name__ == '__main__':
    main()
