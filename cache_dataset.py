"""
Access to the v2 cache built by build_cache.py: memory-mapped shards, selections per split and a
tf.data pipeline for training. Images stay on disk (memory-mapped); only the current batch is in RAM.
"""
import json
import os

import numpy as np

import features


class CacheIndex:
    def __init__(self, cache_dir):
        with open(os.path.join(cache_dir, 'cache.json')) as f:
            self.manifest = json.load(f)
        if self.manifest['feature_names'] != features.FEATURE_NAMES:
            raise ValueError(f'{cache_dir} was built with another feature schema; rebuild the cache')
        self.image_size = self.manifest['image_size']
        self.classes = self.manifest['classes']
        self.images, self.features, self.meta = [], [], []
        for meta_name in self.manifest['shards']:
            stem = os.path.join(cache_dir, meta_name[:-len('.meta.json')])
            with open(stem + '.meta.json') as f:
                self.meta.append(json.load(f))
            self.images.append(np.load(stem + '.images.npy', mmap_mode='r'))
            self.features.append(np.load(stem + '.features.npy', mmap_mode='r'))

    def select(self, split=None, hard=None):
        """(shard, row) pairs of a split ('train', 'val', 'test' or None for all), optionally hard files only."""
        selection = []
        for shard, meta in enumerate(self.meta):
            for row, (s, h) in enumerate(zip(meta['splits'], meta['hard'])):
                if (split is None or s == split) and (hard is None or h == hard):
                    selection.append((shard, row))
        return selection

    def labels(self, selection):
        return np.array([self.meta[s]['labels'][r] for s, r in selection], dtype=np.int64)

    def column(self, selection, key):
        return [self.meta[s][key][r] for s, r in selection]

    def feature_matrix(self, selection):
        if not selection:
            return np.zeros((0, len(features.FEATURE_NAMES)), np.float32)
        return np.stack([self.features[s][r] for s, r in selection]).astype(np.float32)

    def image_batch(self, selection):
        """Float images in [0, 1] with a channel axis, for a (small) selection."""
        return np.stack([self.images[s][r] for s, r in selection]).astype(np.float32)[..., None] / 255.0


def balanced_plan(labels, cap, rng):
    """
    Index plan for one epoch: every class is oversampled up to min(cap * count, largest count),
    without copying data (indices only).
    """
    counts = np.bincount(labels)
    largest = counts.max()
    plan = []
    for cls, count in enumerate(counts):
        if not count:
            continue
        members = np.flatnonzero(labels == cls)
        target = int(min(largest, cap * count)) if cap else count
        plan.append(np.concatenate([np.tile(members, target // count), rng.choice(members, target % count, replace=False)]))
    plan = np.concatenate(plan)
    rng.shuffle(plan)
    return plan


def make_dataset(index, selection, batch_size, training=False, balance_cap=4.0, augment=True, seed=42):
    """tf.data dataset of ({'image', 'features'}, one-hot label) batches read lazily from the cache."""
    import tensorflow as tf

    labels = index.labels(selection)
    num_classes = len(index.classes)
    size = index.image_size
    num_features = len(features.FEATURE_NAMES)
    rng = np.random.default_rng(seed)

    def generator():
        order = balanced_plan(labels, balance_cap, rng) if training else np.arange(len(selection))
        for i in order:
            shard, row = selection[i]
            yield index.images[shard][row], index.features[shard][row], labels[i]

    signature = (
        tf.TensorSpec((size, size), tf.uint8),
        tf.TensorSpec((num_features,), tf.float32),
        tf.TensorSpec((), tf.int64),
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)

    def prepare(image, vector, label):
        image = tf.cast(image, tf.float32)[..., None] / 255.0
        if training and augment:
            # Flips keep the band orientation of spectra; mild gain jitter for exposure level robustness
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.clip_by_value(image * tf.random.uniform((), 0.9, 1.1), 0.0, 1.0)
        return {'image': image, 'features': vector}, tf.one_hot(label, num_classes)

    if training:
        # A new balanced plan per pass; repeat() so fit() with steps_per_epoch never runs out of data
        dataset = dataset.shuffle(2048, seed=seed).repeat()
    else:
        # Known length, so Keras does not treat the end of the validation data as running out of data
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(selection)))
    return dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def steps_per_epoch(labels, balance_cap, batch_size):
    counts = np.bincount(labels)
    largest = counts.max()
    total = sum(int(min(largest, balance_cap * c)) if balance_cap else int(c) for c in counts if c)
    return max(1, total // batch_size)
