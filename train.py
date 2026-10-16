"""
Train the v2 CNN (image + features) from a cache built by build_cache.py. Reads the cache lazily
(memory-mapped), so peak memory stays at a few GB independent of the dataset size.

Writes <out>.keras, <out stem>_features.json (schema) and <out stem>_history.csv.

Example (a12, CPU):
    python train.py --cache /mnt/raid/classifier_cache/v2_448 --out models/v2_cnn.keras --threads 6
"""
import argparse
import json
import os

import numpy as np

import features
from cache_dataset import CacheIndex, make_dataset, steps_per_epoch
from model_v2 import build_model, feature_moments


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train the v2 CNN from a feature/image cache')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True, help='Model path (.keras)')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--balance_cap', type=float, default=4.0,
                        help='Oversample each class up to cap x its size (at most the largest class); 0 = off')
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--base_filters', type=int, default=32)
    parser.add_argument('--separable', action='store_true', help='Separable convolutions (faster on CPU)')
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--threads', type=int, default=0, help='CPU threads for TensorFlow (0 = default)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    import tensorflow as tf
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, args.threads // 2))
    tf.keras.utils.set_random_seed(args.seed)

    index = CacheIndex(args.cache)
    train_sel, val_sel = index.select('train'), index.select('val')
    if not train_sel or not val_sel:
        raise SystemExit('The cache needs train and val files')
    train_labels = index.labels(train_sel)
    mean, variance = feature_moments(index.feature_matrix(train_sel))
    print(f'Train {len(train_sel)}, val {len(val_sel)} files; class counts {np.bincount(train_labels, minlength=len(index.classes)).tolist()}')

    model = build_model(index.image_size, mean, variance, len(index.classes),
                        base_filters=args.base_filters, separable=args.separable)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=['accuracy'],
    )
    train_ds = make_dataset(index, train_sel, args.batch_size, training=True, balance_cap=args.balance_cap,
                            augment=not args.no_augment, seed=args.seed)
    val_ds = make_dataset(index, val_sel, args.batch_size)

    stem = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=max(2, args.patience // 3)),
        tf.keras.callbacks.CSVLogger(stem + '_history.csv'),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, shuffle=False,
              steps_per_epoch=steps_per_epoch(train_labels, args.balance_cap, args.batch_size), callbacks=callbacks)

    model.save(args.out)
    schema = features.schema(index.image_size, kind='cnn', classes=index.classes,
                             training={k: v for k, v in vars(args).items()}, cache=index.manifest.get('export_params'))
    with open(stem + '_features.json', 'w') as f:
        json.dump(schema, f, indent=1)
    print(f'Model written to {args.out}')


if __name__ == '__main__':
    main()
