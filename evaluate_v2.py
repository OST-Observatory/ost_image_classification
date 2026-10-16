"""
Evaluate v2 models (cnn / lightgbm) and, for comparison, the legacy production model on the cache splits.

Temperature and per-class thresholds are fitted on 'val' and applied to 'test' (never tuned on test).
Reported for the test split and its hard subset (files where header or classifier had been wrong):
accuracy, per-class report, exposure-type and spectrograph confusion (e.g. LI -> DA), calibration (ECE),
coverage and accuracy at coverage with abstaining, and accuracy per instrument and year.

Example (a12):
    python evaluate_v2.py --cache /mnt/raid/classifier_cache/v2_448 --model models/v2_cnn.keras \\
        --model models/v2_lgbm.txt --legacy_model OSTdata/keras_model/multimodal_classifier_mixup_warmup_randaug_2.keras \\
        --legacy_thresholds thresholds_prod_precision.json --base_path /mnt/raid/data_archive \\
        --out reports/eval_v2.json --save_calibration
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

import features
from cache_dataset import CacheIndex
from predictors import V2Predictor, load_schema

TEMPERATURES = np.round(np.arange(0.3, 3.01, 0.1), 2)


def apply_temperature(proba, temperature):
    """Temperature scaling in probability space (p ** (1/T), renormalised), as in the inference runner."""
    q = np.power(np.clip(proba, 1e-8, 1.0), 1.0 / max(temperature, 1e-8))
    return q / q.sum(axis=1, keepdims=True)


def fit_temperature(proba, labels):
    def nll(t):
        p = apply_temperature(proba, t)
        return float(-np.mean(np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0))))
    return float(min(TEMPERATURES, key=nll)) if len(labels) else 1.0


def fit_thresholds(proba, labels, classes, target_precision):
    """Per class: lowest threshold reaching the target precision on val, else the max-F1 threshold."""
    from sklearn.metrics import precision_recall_curve
    thresholds = {}
    for c, name in enumerate(classes):
        positives = labels == c
        if not positives.any():
            thresholds[name] = 0.5
            continue
        precision, recall, thr = precision_recall_curve(positives, proba[:, c])
        precision, recall = precision[:-1], recall[:-1]
        ok = np.flatnonzero(precision >= target_precision)
        if ok.size:
            thresholds[name] = float(thr[ok[0]])
        else:
            f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
            thresholds[name] = float(thr[int(np.argmax(f1))])
    return thresholds


def gate(proba, thresholds, classes):
    """Highest class passing its threshold; -1 (abstain) if none passes."""
    thr = np.array([thresholds.get(name, 0.5) for name in classes])
    passing = np.where(proba >= thr[None, :], proba, -1.0)
    pred = passing.argmax(axis=1)
    pred[passing.max(axis=1) < 0] = -1
    return pred


def expected_calibration_error(proba, labels, bins=15):
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def _group_accuracy(groups, correct):
    by = defaultdict(list)
    for group, ok in zip(groups, correct):
        by[group].append(ok)
    return {str(k): {'n': len(v), 'accuracy': round(float(np.mean(v)), 4)} for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}


def metrics(proba, labels, classes, thresholds=None, instruments=None, years=None):
    from sklearn.metrics import classification_report, confusion_matrix
    if not len(labels):
        return {'n': 0}
    pred = proba.argmax(axis=1)
    correct = pred == labels
    exposure_of = [features.CLASS_EXPOSURE.get(c, c) for c in classes]
    spectrograph_of = [features.CLASS_SPECTROGRAPH.get(c, 'N') for c in classes]
    exposure_confusion = Counter()
    spectrograph_confusion = Counter()
    for t, p in zip(labels, pred):
        if exposure_of[t] != exposure_of[p]:
            exposure_confusion[f'{exposure_of[t]} -> {exposure_of[p]}'] += 1
        if spectrograph_of[t] != spectrograph_of[p]:
            spectrograph_confusion[f'{spectrograph_of[t]} -> {spectrograph_of[p]}'] += 1
    result = {
        'n': int(len(labels)),
        'accuracy': round(float(correct.mean()), 4),
        'exposure_accuracy': round(float(np.mean([exposure_of[t] == exposure_of[p] for t, p in zip(labels, pred)])), 4),
        'exposure_errors': dict(exposure_confusion.most_common()),
        'spectrograph_errors': dict(spectrograph_confusion.most_common()),
        'ece': round(expected_calibration_error(proba, labels), 4),
        'per_class': classification_report(labels, pred, labels=list(range(len(classes))), target_names=classes,
                                           output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(labels, pred, labels=list(range(len(classes)))).tolist(),
    }
    if thresholds is not None:
        gated = gate(proba, thresholds, classes)
        covered = gated >= 0
        result['coverage'] = round(float(covered.mean()), 4)
        result['accuracy_at_coverage'] = round(float((gated[covered] == labels[covered]).mean()), 4) if covered.any() else None
    if instruments is not None:
        result['by_instrument'] = _group_accuracy(instruments, correct)
    if years is not None:
        result['by_year'] = _group_accuracy(years, correct)
    return result


def predict_v2(predictor, index, selection, batch_size=64):
    out = []
    for start in range(0, len(selection), batch_size):
        chunk = selection[start:start + batch_size]
        images = index.image_batch(chunk) if predictor.needs_images else None
        out.append(predictor.predict(images, index.feature_matrix(chunk)))
    return np.concatenate(out) if out else np.zeros((0, len(predictor.classes)), np.float32)


def predict_legacy(model_path, thresholds_path, base_path, paths, batch_size=16):
    """Probabilities of the legacy production model (temperature 0.7 as in production); None if unreadable."""
    from inference_runner import ClassifierService
    service = ClassifierService(model_path, thresholds_path=thresholds_path, temperature=0.7, abstain_unknown=True)
    full = [os.path.join(base_path, p) for p in paths]
    payload = service.predict_paths(full, batch_size=batch_size)
    probs = {r['path']: r['_probs'] for r in payload['results'] if '_probs' in r}
    return [probs.get(p) for p in full], service.thresholds, service.class_names


def evaluate_split(proba, index, selection, labels, classes, thresholds):
    hard = np.array(index.column(selection, 'hard'), dtype=bool)
    instruments = index.column(selection, 'instruments')
    years = index.column(selection, 'years')
    return {
        'all': metrics(proba, labels, classes, thresholds, instruments, years),
        'hard': metrics(proba[hard], labels[hard], classes, thresholds),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate v2 models and the legacy model on the cache splits')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--model', action='append', default=[], help='v2 model (repeatable)')
    parser.add_argument('--legacy_model', help='Legacy production model (.keras) for comparison')
    parser.add_argument('--legacy_thresholds', help='Thresholds JSON of the legacy model')
    parser.add_argument('--base_path', default='', help='Archive root (needed for the legacy model)')
    parser.add_argument('--split', default='test')
    parser.add_argument('--target_precision', type=float, default=0.95)
    parser.add_argument('--out', required=True, help='Report JSON')
    parser.add_argument('--save_calibration', action='store_true',
                        help='Write <model stem>_thresholds.json and _calibration.json (temperature) per v2 model')
    args = parser.parse_args(argv)

    index = CacheIndex(args.cache)
    val_sel, test_sel = index.select('val'), index.select(args.split)
    report = {'cache': args.cache, 'split': args.split, 'models': {}}

    for model_path in args.model:
        schema = load_schema(model_path)
        if schema is None:
            raise SystemExit(f'{model_path} has no schema file; use --legacy_model for legacy models')
        predictor = V2Predictor(model_path, schema)
        classes = predictor.classes
        to_target = None
        if schema.get('target') == 'exposure':
            mapping = np.array([features.EXPOSURE_TYPES.index(features.CLASS_EXPOSURE[c]) for c in index.classes])
            to_target = mapping
        y_val, y_test = index.labels(val_sel), index.labels(test_sel)
        if to_target is not None:
            y_val, y_test = to_target[y_val], to_target[y_test]
        raw_val = predict_v2(predictor, index, val_sel)
        temperature = fit_temperature(raw_val, y_val)
        p_val = apply_temperature(raw_val, temperature)
        thresholds = fit_thresholds(p_val, y_val, classes, args.target_precision)
        p_test = apply_temperature(predict_v2(predictor, index, test_sel), temperature)
        report['models'][model_path] = {
            'kind': schema['kind'], 'temperature': temperature, 'thresholds': thresholds,
            **evaluate_split(p_test, index, test_sel, y_test, classes, thresholds),
        }
        if args.save_calibration:
            stem = os.path.splitext(model_path)[0]
            with open(stem + '_thresholds.json', 'w') as f:
                json.dump(thresholds, f, indent=1)
            with open(stem + '_calibration.json', 'w') as f:
                json.dump({'temperature': temperature, 'target_precision': args.target_precision}, f, indent=1)

    if args.legacy_model:
        paths = index.column(test_sel, 'paths')
        probs, legacy_thresholds, legacy_classes = predict_legacy(
            args.legacy_model, args.legacy_thresholds, args.base_path, paths)
        keep = [i for i, p in enumerate(probs) if p is not None]
        selection = [test_sel[i] for i in keep]
        proba = np.array([probs[i] for i in keep], dtype=np.float32)
        thresholds = dict(zip(legacy_classes, map(float, legacy_thresholds)))
        report['models'][args.legacy_model] = {
            'kind': 'legacy', 'unreadable': len(paths) - len(keep), 'thresholds': thresholds,
            **evaluate_split(proba, index, selection, index.labels(selection), legacy_classes, thresholds),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=1)

    print(f"{'model':<48} {'n':>6} {'acc':>7} {'hard':>7} {'expo':>7} {'LI->DA':>7} {'DA->LI':>7} {'ECE':>6} {'cov':>6} {'acc@cov':>8}")
    for name, result in report['models'].items():
        a, h = result['all'], result['hard']
        print(f"{os.path.basename(name):<48} {a['n']:>6} {a['accuracy']:>7} {h.get('accuracy', '-'):>7} "
              f"{a['exposure_accuracy']:>7} {a['exposure_errors'].get('LI -> DA', 0):>7} "
              f"{a['exposure_errors'].get('DA -> LI', 0):>7} {a['ece']:>6} {a.get('coverage', '-'):>6} "
              f"{a.get('accuracy_at_coverage', '-'):>8}")
    print(f'Report written to {args.out}')


if __name__ == '__main__':
    main()
