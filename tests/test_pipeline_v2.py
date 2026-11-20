"""Cache, training (CNN + LightGBM), inference and evaluation of the v2 pipeline on a tiny dataset."""
import json
import os

import numpy as np
import pytest

import build_cache
import features
from cache_dataset import CacheIndex, balanced_plan


def test_cache_contents_and_resume(small_cache, tmp_path):
    index = CacheIndex(small_cache['cache'])
    assert index.image_size == 32
    assert len(index.select('train')) == 18 and len(index.select('test')) == 9
    assert len(index.select('test', hard=True)) == 3
    batch = index.image_batch(index.select('val')[:4])
    assert batch.shape == (4, 32, 32, 1) and 0.0 <= batch.min() and batch.max() <= 1.0

    before = {name: os.path.getmtime(os.path.join(small_cache['cache'], name)) for name in index.manifest['shards']}
    build_cache.build(small_cache['export'], small_cache['archive'], small_cache['cache'], image_size=32,
                      workers=1, shard_size=10)
    after = {name: os.path.getmtime(os.path.join(small_cache['cache'], name)) for name in before}
    assert before == after  # existing shards are not rebuilt


def test_cache_reports_unreadable_files(tmp_path, small_cache):
    records = [{'path': 'missing.fits', 'class': 'darks', 'split': 'train'}]
    export = tmp_path / 'broken.json'
    export.write_text(json.dumps({'format': 'ost-classification-export/v2', 'files': records}))
    cache = tmp_path / 'broken_cache'
    build_cache.build(str(export), small_cache['archive'], str(cache), image_size=32, workers=1)
    errors = (cache / 'errors.jsonl').read_text().splitlines()
    assert json.loads(errors[0])['path'] == 'missing.fits'


def test_cache_rejects_other_schema(small_cache):
    with pytest.raises(ValueError):
        build_cache.build(small_cache['export'], small_cache['archive'], small_cache['cache'], image_size=64)


def test_balanced_plan():
    labels = np.array([0] * 100 + [1] * 10 + [2] * 1)
    plan = balanced_plan(labels, cap=4.0, rng=np.random.default_rng(0))
    counts = np.bincount(labels[plan])
    assert counts.tolist() == [100, 40, 4]


@pytest.fixture
def cnn_model(small_cache, tmp_path):
    import train
    out = str(tmp_path / 'models' / 'cnn.keras')
    train.main(['--cache', small_cache['cache'], '--out', out, '--epochs', '1', '--batch_size', '8',
                '--base_filters', '4', '--threads', '2'])
    return out


@pytest.fixture
def lgbm_model(small_cache, tmp_path):
    import train_tabular
    out = str(tmp_path / 'models' / 'lgbm.txt')
    train_tabular.main(['--cache', small_cache['cache'], '--out', out, '--rounds', '20', '--threads', '2'])
    return out


def test_training_writes_schema(cnn_model):
    schema = json.load(open(os.path.splitext(cnn_model)[0] + '_features.json'))
    assert schema['kind'] == 'cnn' and schema['image_size'] == 32
    assert schema['feature_names'] == features.FEATURE_NAMES


@pytest.mark.parametrize('model_fixture', ['cnn_model', 'lgbm_model'])
def test_classifier_service_v2(model_fixture, request, small_cache):
    from inference_runner import ClassifierService
    model_path = request.getfixturevalue(model_fixture)
    service = ClassifierService(model_path, temperature=1.0, abstain_unknown=True)
    paths = [os.path.join(small_cache['archive'], r['path']) for r in small_cache['records'][:3]]
    paths.append('/nonexistent.fits')
    metadata = {paths[0]: {'instrument': 'QHY 600M', 'plate_solved': True, 'run_spectroscopy': False}}

    payload = service.predict_paths(paths, batch_size=2, metadata=metadata)

    ok = [r for r in payload['results'] if 'error' not in r]
    assert len(ok) == 3 and len(payload['defects']) == 1
    assert all(r['top_class'] in features.CLASSES and 0 <= r['top_score'] <= 1 for r in ok)
    assert payload['meta']['feature_schema'] == features.SCHEMA_VERSION


def test_schema_mismatch_is_rejected(lgbm_model):
    from inference_runner import ClassifierService
    schema_path = os.path.splitext(lgbm_model)[0] + '_features.json'
    schema = json.load(open(schema_path))
    schema['feature_names'] = schema['feature_names'][:-1]
    json.dump(schema, open(schema_path, 'w'))
    with pytest.raises(ValueError, match='feature schema'):
        ClassifierService(lgbm_model)


def test_evaluate_v2(small_cache, cnn_model, lgbm_model, tmp_path):
    import evaluate_v2
    out = tmp_path / 'report.json'
    evaluate_v2.main(['--cache', small_cache['cache'], '--model', cnn_model, '--model', lgbm_model,
                      '--out', str(out), '--save_calibration'])
    report = json.loads(out.read_text())
    for path in (cnn_model, lgbm_model):
        result = report['models'][path]
        assert result['all']['n'] == 9 and result['hard']['n'] == 3
        assert {'accuracy', 'exposure_accuracy', 'ece', 'coverage', 'by_instrument', 'per_class',
                'run_accuracy', 'worst_runs'} <= set(result['all'])
        thresholds = json.load(open(os.path.splitext(path)[0] + '_thresholds.json'))
        assert set(thresholds) == set(features.CLASSES)  # flat {class: threshold}, readable by the runner


def test_evaluate_excludes_paths(small_cache, lgbm_model, tmp_path):
    import evaluate_v2
    test_paths = [r['path'] for r in small_cache['records'] if r['split'] == 'test'][:4]
    legacy_export = tmp_path / 'legacy.json'
    legacy_export.write_text(json.dumps({'train': {'darks': test_paths}}))
    out = tmp_path / 'report.json'
    evaluate_v2.main(['--cache', small_cache['cache'], '--model', lgbm_model, '--out', str(out),
                      '--exclude_paths', str(legacy_export)])
    assert json.loads(out.read_text())['models'][lgbm_model]['all']['n'] == 9 - 4


def test_evaluate_ensemble(small_cache, cnn_model, lgbm_model, tmp_path):
    import evaluate_v2
    out = tmp_path / 'report.json'
    evaluate_v2.main(['--cache', small_cache['cache'], '--model', lgbm_model,
                      '--ensemble', cnn_model, f'{lgbm_model}=3', '--out', str(out)])
    models = json.loads(out.read_text())['models']
    name = next(n for n in models if n.startswith('ensemble('))
    assert models[name]['members'] == {cnn_model: 0.25, lgbm_model: 0.75}
    assert models[name]['all']['n'] == 9 and 'run_accuracy' in models[name]['all']
    assert lgbm_model in models


def test_ensemble_file_in_classifier_service(small_cache, cnn_model, lgbm_model, tmp_path):
    import evaluate_v2
    from inference_runner import ClassifierService
    evaluate_v2.main(['--cache', small_cache['cache'], '--ensemble', cnn_model, f'{lgbm_model}=3',
                      '--out', str(tmp_path / 'report.json'), '--save_calibration'])
    path = os.path.join(os.path.dirname(cnn_model), 'cnn+lgbmx3_ensemble.json')
    config = json.load(open(path))
    assert [m['model'] for m in config['members']] == ['cnn.keras', 'lgbm.txt']
    assert [m['weight'] for m in config['members']] == [0.25, 0.75]

    paths = [os.path.join(small_cache['archive'], r['path']) for r in small_cache['records'][:4]]
    service = ClassifierService(path)  # temperature and thresholds from the ensemble file
    assert service.temperature == config['temperature']
    assert service.thresholds.tolist() == pytest.approx([config['thresholds'][c] for c in service.class_names])
    payload = service.predict_paths(paths, batch_size=3)
    assert payload['meta']['model_kind'] == 'ensemble' and len(payload['results']) == 4

    members = [ClassifierService(m, temperature=1.0) for m in (cnn_model, lgbm_model)]
    probs = [np.array([r['_probs'] for r in m.predict_paths(paths, batch_size=3)['results']]) for m in members]
    expected = evaluate_v2.apply_temperature(0.25 * probs[0] + 0.75 * probs[1], config['temperature'])
    actual = np.array([r['_probs'] for r in payload['results']])
    assert np.allclose(actual, expected, atol=1e-5)


def test_single_model_uses_saved_calibration(small_cache, lgbm_model, tmp_path):
    import evaluate_v2
    from inference_runner import ClassifierService
    assert ClassifierService(lgbm_model).temperature == 1.0  # no calibration saved yet
    evaluate_v2.main(['--cache', small_cache['cache'], '--model', lgbm_model, '--out', str(tmp_path / 'r.json'),
                      '--save_calibration'])
    stem = os.path.splitext(lgbm_model)[0]
    calibration = json.load(open(stem + '_calibration.json'))
    thresholds = json.load(open(stem + '_thresholds.json'))
    service = ClassifierService(lgbm_model)
    assert service.temperature == calibration['temperature']
    assert service.thresholds.tolist() == pytest.approx([thresholds[c] for c in service.class_names])
    assert ClassifierService(lgbm_model, temperature=2.0).temperature == 2.0  # explicit value wins


def test_ensemble_rejects_mismatching_members(small_cache, cnn_model, lgbm_model, tmp_path):
    from inference_runner import ClassifierService
    from predictors import ENSEMBLE_FORMAT
    schema_path = os.path.splitext(lgbm_model)[0] + '_features.json'
    schema = json.load(open(schema_path))
    schema['image_size'] = 64
    json.dump(schema, open(schema_path, 'w'))
    path = tmp_path / 'models' / 'bad_ensemble.json'
    path.write_text(json.dumps({'format': ENSEMBLE_FORMAT,
                                'members': [{'model': 'cnn.keras'}, {'model': 'lgbm.txt'}]}))
    with pytest.raises(ValueError, match='image size'):
        ClassifierService(str(path))


def test_other_json_is_rejected(tmp_path):
    from inference_runner import ClassifierService
    report = tmp_path / 'eval_v2b_ensemble.json'
    report.write_text(json.dumps({'cache': 'x', 'models': {}}))
    with pytest.raises(ValueError, match='not an ensemble file'):
        ClassifierService(str(report))


def test_parse_member():
    import evaluate_v2
    assert evaluate_v2.parse_member('models/a.keras') == ('models/a.keras', 1.0)
    assert evaluate_v2.parse_member('models/b.txt=0.6') == ('models/b.txt', 0.6)


def test_top_class_gating():
    from inference_runner import apply_thresholds
    import evaluate_v2
    classes = ['deep_sky', 'bias', 'flats']
    proba = np.array([[0.55, 0.002, 0.448],   # top class misses its threshold -> abstain, not bias
                      [0.90, 0.05, 0.05],
                      [0.10, 0.05, 0.85]])
    thresholds = {'deep_sky': 0.6, 'bias': 0.001, 'flats': 0.9}
    thr = np.array([thresholds[c] for c in classes])
    pred, abstained = apply_thresholds(proba, thr, True, 3, top_class_only=True)
    assert pred.tolist() == [3, 0, 3] and abstained.tolist() == [True, False, True]
    assert evaluate_v2.gate(proba, thresholds, classes).tolist() == [-1, 0, -1]
    legacy, _ = apply_thresholds(proba, thr, True, 3)  # legacy rule unchanged
    assert legacy.tolist() == [1, 0, 1]


def test_thresholds_fitted_on_top_class_files():
    import evaluate_v2
    classes = ['deep_sky', 'bias']
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.55, 0.45], [0.2, 0.8]])
    labels = np.array([0, 0, 1, 1])
    thresholds = evaluate_v2.fit_thresholds(proba, labels, classes, target_precision=0.95)
    assert thresholds['deep_sky'] == pytest.approx(0.6)  # 0.55 (a bias file) is excluded
    assert thresholds['bias'] == pytest.approx(0.8)
