"""
Build the feature/image cache for training v2 from an archive export (format v2, see
ost_data_archive export_cnn_training_data).

Per shard: <cache>/shard_00000.images.npy (uint8 N x S x S), .features.npy (float32 N x F) and
.meta.json (labels, splits, paths, run, instrument, year, hard). Shards can be memory-mapped, so
training reads them without loading everything into RAM. Existing shards are skipped (resumable);
unreadable files are listed in <cache>/errors.jsonl.

Example (a12):
    python build_cache.py --export export_v2.json --base_path /mnt/raid/data_archive \\
        --out /mnt/raid/classifier_cache/v2_448 --image_size 448 --workers 5
"""
import argparse
import json
import os
import time
import multiprocessing

import numpy as np

import features

EXPORT_FORMAT = 'ost-classification-export/v2'


def _process(task):
    index, record, base_path, image_size = task
    path = os.path.join(base_path, record['path']) if base_path else record['path']
    metadata = {
        'instrument': record.get('instrument'),
        'plate_solved': record.get('plate_solved'),
        'run_spectroscopy': record.get('run_spectroscopy'),
    }
    try:
        image, vector = features.extract(path, metadata=metadata, image_size=image_size)
        return index, image, vector, None
    except Exception as e:  # unreadable/corrupt file: reported, not fatal
        return index, None, None, f'{type(e).__name__}: {e}'


def shard_paths(out_dir, shard):
    stem = os.path.join(out_dir, f'shard_{shard:05d}')
    return stem + '.images.npy', stem + '.features.npy', stem + '.meta.json'


def build(export_path, base_path, out_dir, image_size=448, workers=5, shard_size=2000, limit=None):
    with open(export_path) as f:
        export = json.load(f)
    if export.get('format') != EXPORT_FORMAT:
        raise ValueError(f'{export_path} is not an export in format {EXPORT_FORMAT}')
    records = export['files'][:limit] if limit else export['files']
    class_index = {name: i for i, name in enumerate(features.CLASSES)}
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = os.path.join(out_dir, 'cache.json')
    manifest = features.schema(image_size, kind='cache', export=export_path, export_params=export.get('params', {}))
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
        if existing.get('feature_names') != manifest['feature_names'] or existing.get('image_size') != image_size:
            raise ValueError(f'{out_dir} contains a cache with a different schema or image size')

    started = time.time()
    shards = []
    # spawn: safe even if the calling process already runs threads (e.g. TensorFlow)
    with multiprocessing.get_context('spawn').Pool(processes=workers) as pool:
        for shard, start in enumerate(range(0, len(records), shard_size)):
            images_path, features_path, meta_path = shard_paths(out_dir, shard)
            shards.append(os.path.basename(meta_path))
            if os.path.exists(meta_path):
                continue
            chunk = records[start:start + shard_size]
            tasks = [(i, record, base_path, image_size) for i, record in enumerate(chunk)]
            results = sorted(pool.imap_unordered(_process, tasks, chunksize=4), key=lambda r: r[0])

            ok = [r for r in results if r[3] is None]
            with open(os.path.join(out_dir, 'errors.jsonl'), 'a') as f:
                for index, _, _, error in results:
                    if error is not None:
                        f.write(json.dumps({'path': chunk[index]['path'], 'error': error}) + '\n')
            images = np.stack([r[1] for r in ok]) if ok else np.zeros((0, image_size, image_size), np.uint8)
            vectors = np.stack([r[2] for r in ok]) if ok else np.zeros((0, len(features.FEATURE_NAMES)), np.float32)
            rows = [chunk[r[0]] for r in ok]
            meta = {
                'labels': [class_index[r['class']] for r in rows],
                'splits': [r['split'] for r in rows],
                'hard': [bool(r.get('hard')) for r in rows],
                'paths': [r['path'] for r in rows],
                'runs': [r.get('run') for r in rows],
                'instruments': [r.get('instrument') or '' for r in rows],
                'years': [r.get('year') for r in rows],
            }
            # Arrays first, metadata last: a shard counts as complete once its .meta.json exists
            np.save(images_path, images)
            np.save(features_path, vectors)
            with open(meta_path + '.tmp', 'w') as f:
                json.dump(meta, f)
            os.replace(meta_path + '.tmp', meta_path)
            done = min(start + shard_size, len(records))
            print(f'shard {shard}: {len(ok)}/{len(chunk)} ok, {done}/{len(records)} files, '
                  f'{time.time() - started:.0f} s', flush=True)

    manifest['shards'] = shards
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=1)
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build the v2 feature/image cache from an archive export')
    parser.add_argument('--export', required=True, help='Export JSON (format v2)')
    parser.add_argument('--base_path', default='', help='Archive root the export paths are relative to')
    parser.add_argument('--out', required=True, help='Cache directory')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--shard_size', type=int, default=2000)
    parser.add_argument('--limit', type=int, default=None, help='Only the first N files (testing)')
    args = parser.parse_args()
    manifest = build(args.export, args.base_path, args.out, args.image_size, args.workers, args.shard_size, args.limit)
    print(f"Cache with {len(manifest['shards'])} shards written to {args.out}")


if __name__ == '__main__':
    main()
