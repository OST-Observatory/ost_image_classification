"""
Tabular baseline: LightGBM on the v2 features only (no image), from a cache built by build_cache.py.
Trains in minutes on CPU and shows how much the image branch actually adds.

--target class     all 11 classes (deployable like the CNN, kind 'lightgbm')
--target exposure  exposure type only (BI/DA/FL/LI/WA), for analysis / a possible hybrid

Writes <out>.txt (LightGBM model) and <out stem>_features.json (schema).

Example:
    python train_tabular.py --cache /mnt/raid/classifier_cache/v2_448 --out models/v2_lgbm.txt
"""
import argparse
import json
import os

import numpy as np

import features
from cache_dataset import CacheIndex


def target_labels(labels, classes, target):
    """Labels and class names for the chosen target."""
    if target == 'class':
        return labels, list(classes)
    exposure = [features.EXPOSURE_TYPES.index(features.CLASS_EXPOSURE[c]) for c in classes]
    return np.array([exposure[label] for label in labels]), list(features.EXPOSURE_TYPES)


def class_weights(labels, num_classes, beta):
    """Per-sample weights (largest count / class count) ** beta."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weight = np.where(counts > 0, (counts.max() / np.maximum(counts, 1)) ** beta, 0.0)
    return weight[labels]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train the LightGBM baseline on v2 features')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True, help='Model path (.txt)')
    parser.add_argument('--target', choices=['class', 'exposure'], default='class')
    parser.add_argument('--num_leaves', type=int, default=63)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--rounds', type=int, default=2000)
    parser.add_argument('--early_stopping', type=int, default=100)
    parser.add_argument('--class_weight_beta', type=float, default=0.5)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args(argv)

    import lightgbm as lgb

    index = CacheIndex(args.cache)
    train_sel, val_sel = index.select('train'), index.select('val')
    y_train, names = target_labels(index.labels(train_sel), index.classes, args.target)
    y_val, _ = target_labels(index.labels(val_sel), index.classes, args.target)
    x_train, x_val = index.feature_matrix(train_sel), index.feature_matrix(val_sel)

    params = {
        'objective': 'multiclass', 'num_class': len(names), 'learning_rate': args.learning_rate,
        'num_leaves': args.num_leaves, 'min_data_in_leaf': 20, 'feature_fraction': 0.9, 'bagging_fraction': 0.9,
        'bagging_freq': 1, 'verbose': -1, 'num_threads': args.threads,
    }
    train_set = lgb.Dataset(x_train, y_train, weight=class_weights(y_train, len(names), args.class_weight_beta),
                            feature_name=features.FEATURE_NAMES, free_raw_data=True)
    val_set = lgb.Dataset(x_val, y_val, reference=train_set)
    booster = lgb.train(params, train_set, num_boost_round=args.rounds, valid_sets=[val_set],
                        callbacks=[lgb.early_stopping(args.early_stopping, verbose=False), lgb.log_evaluation(100)])

    accuracy = float(np.mean(booster.predict(x_val).argmax(axis=1) == y_val)) if len(y_val) else float('nan')
    print(f'Validation accuracy: {accuracy:.4f} ({booster.best_iteration} rounds)')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    booster.save_model(args.out, num_iteration=booster.best_iteration)
    schema = features.schema(index.image_size, kind='lightgbm', classes=names, target=args.target,
                             training=vars(args), cache=index.manifest.get('export_params'))
    with open(os.path.splitext(args.out)[0] + '_features.json', 'w') as f:
        json.dump(schema, f, indent=1)
    importance = sorted(zip(features.FEATURE_NAMES, booster.feature_importance('gain')), key=lambda kv: -kv[1])
    print('Top features (gain):', ', '.join(f'{n} {v:.0f}' for n, v in importance[:12]))
    print(f'Model written to {args.out}')


if __name__ == '__main__':
    main()
