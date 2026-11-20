"""
Diagnose an archive comparison (OSTdata: classify_exposure_types --compare --json compare.json) on the
training machine: for every compared file, is it in the cache (split, label at export time), was it
unreadable when the cache was built (errors.jsonl), or not in the export at all? For files in the cache the
model is run on the cached image/features and its top class is compared with the archive's top class.

- cache top == archive top for (almost) all files: the archive computes the same inputs; errors come from
  the model or the labels.
- many differences: the inputs differ between the machines (library versions, file access).

Example (a12):
    python diagnose_compare.py --cache /mnt/raid/classifier_cache/v2_448b \\
        --model models/v2b_cnn+v2b_lgbm_ensemble.json --compare compare.json \\
        --archive_prefix /mnt/data_archive/ --out reports/diagnose_compare.json
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

import features
from cache_dataset import CacheIndex
from predictors import load_predictor

UNKNOWN = 'UK'


def relative(path, prefix):
    return path[len(prefix):] if prefix and path.startswith(prefix) else path


def load_errors(cache_dir):
    path = os.path.join(cache_dir, 'errors.jsonl')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {entry['path']: entry.get('error', '') for entry in map(json.loads, f) if entry.get('path')}


def cache_predictions(index, predictor, selection, batch_size=64):
    """Top class per cached file (temperature does not change the top class)."""
    top = []
    for start in range(0, len(selection), batch_size):
        chunk = selection[start:start + batch_size]
        images = index.image_batch(chunk) if predictor.needs_images else None
        top.extend(np.argmax(predictor.predict(images, index.feature_matrix(chunk)), axis=1).tolist())
    return [predictor.classes[i] for i in top]


def diagnose(rows, index, predictor, errors, prefix):
    position = {}
    for shard, meta in enumerate(index.meta):
        for row, path in enumerate(meta['paths']):
            position[path] = (shard, row)

    in_cache = [r for r in rows if relative(r['path'], prefix) in position]
    selection = [position[relative(r['path'], prefix)] for r in in_cache]
    tops = cache_predictions(index, predictor, selection)
    splits = index.column(selection, 'splits')
    labels = [index.classes[i] for i in index.labels(selection)]

    files = []
    for r in rows:
        rel = relative(r['path'], prefix)
        files.append({'path': r['path'], 'run': r.get('run'), 'manual': r.get('user') or '',
                      'archive_top': r.get('new_top_class') or '', 'status': 'not_in_export',
                      'cache_error': errors.get(rel)})
        if rel in errors:
            files[-1]['status'] = 'unreadable_in_cache'
    by_path = {f['path']: f for f in files}
    for r, top, split, label in zip(in_cache, tops, splits, labels):
        by_path[r['path']].update(status=split, cache_top=top, cache_label=label)
    return files


def summarize(files, examples=3):
    expo = features.CLASS_EXPOSURE
    cached = [f for f in files if 'cache_top' in f]
    differ = [f for f in cached if f['cache_top'] != f['archive_top']]
    manual = [f for f in files if f['manual'] and f['manual'] != UNKNOWN]

    def errors_of(subset):
        wrong = [f for f in subset if expo.get(f['archive_top']) != f['manual']]
        return {'n': len(subset), 'top_class_wrong': len(wrong)}, wrong

    by_status, wrong_runs, run_examples = {}, Counter(), defaultdict(list)
    for status in sorted({f['status'] for f in manual}):
        stats, wrong = errors_of([f for f in manual if f['status'] == status])
        by_status[status] = stats
        for f in wrong:
            key = f"{f['run']} ({f['status']})"
            wrong_runs[key] += 1
            if len(run_examples[key]) < examples:
                run_examples[key].append(f"{os.path.basename(f['path'])}  ({f['manual']} -> {f['archive_top']})")
    relabelled = [f for f in manual if 'cache_label' in f and expo.get(f['cache_label']) != f['manual']]
    return {
        'files': len(files),
        'status': dict(Counter(f['status'] for f in files)),
        'cached': len(cached),
        'cache_vs_archive_top_differ': len(differ),
        'differ_examples': [f"{f['path']}  (cache {f['cache_top']}, archive {f['archive_top']})"
                            for f in differ[:10]],
        'manual_by_status': by_status,
        'manual_label_changed_since_export': len(relabelled),
        'wrong_runs': [{'run': key, 'n': n, 'examples': run_examples[key]} for key, n in wrong_runs.most_common(15)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Diagnose an archive comparison against the v2 cache')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--model', required=True, help='v2 model or ensemble file used in the archive')
    parser.add_argument('--compare', required=True, help='JSON of classify_exposure_types --compare --json')
    parser.add_argument('--archive_prefix', default='', help='Archive root in the compare paths, e.g. /mnt/data_archive/')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)

    with open(args.compare) as f:
        rows = json.load(f)['files']
    index = CacheIndex(args.cache)
    predictor, _ = load_predictor(args.model)
    files = diagnose(rows, index, predictor, load_errors(args.cache), args.archive_prefix)
    summary = summarize(files)

    print(f"{summary['files']} compared files: " + ', '.join(f'{k} {v}' for k, v in sorted(summary['status'].items())))
    print(f"In the cache: {summary['cached']}; cache top class != archive top class: "
          f"{summary['cache_vs_archive_top_differ']}")
    for line in summary['differ_examples']:
        print(f'    {line}')
    print('\nManually labelled files (top class wrong / n):')
    for status, stats in summary['manual_by_status'].items():
        print(f"    {status:<20} {stats['top_class_wrong']:>4} / {stats['n']}")
    print(f"Manual label changed since the export: {summary['manual_label_changed_since_export']}")
    print('\nRuns with most wrong top classes (manual label -> archive top class):')
    for entry in summary['wrong_runs']:
        print(f"    {entry['run']}: {entry['n']}")
        for example in entry['examples']:
            print(f'        {example}')
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'summary': summary, 'files': files}, f, indent=1)
        print(f'\nWritten to {args.out}')


if __name__ == '__main__':
    main()
