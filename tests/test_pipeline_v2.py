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
        assert {'accuracy', 'exposure_accuracy', 'ece', 'coverage', 'by_instrument', 'per_class'} <= set(result['all'])
        thresholds = json.load(open(os.path.splitext(path)[0] + '_thresholds.json'))
        assert set(thresholds) == set(features.CLASSES)  # flat {class: threshold}, readable by the runner
