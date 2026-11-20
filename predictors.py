"""
Loading and prediction for v2 models (feature schema 2). A v2 model is recognised by its schema file
<model stem>_features.json (written by train.py / train_tabular.py); kind 'cnn' (Keras) or 'lightgbm'.

An ensemble is a JSON file (format ENSEMBLE_FORMAT, written by evaluate_v2.py --save_calibration) listing
v2 member models (paths relative to the file) with weights, plus temperature and thresholds fitted on val.
It predicts the weighted mean of the member probabilities.
"""
import json
import os

import numpy as np

import features

ENSEMBLE_FORMAT = 'ost-classifier-ensemble/1'


def schema_path(model_path):
    return os.path.splitext(model_path)[0] + '_features.json'


def load_schema(model_path):
    """Schema of a v2 model, None for legacy models (no schema file)."""
    path = schema_path(model_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class V2Predictor:
    def __init__(self, model_path, schema, tta=False):
        if schema.get('schema_version') != features.SCHEMA_VERSION or schema.get('feature_names') != features.FEATURE_NAMES:
            raise ValueError(
                f'{model_path} was trained with feature schema {schema.get("schema_version")} '
                f'({len(schema.get("feature_names", []))} features); this code computes schema '
                f'{features.SCHEMA_VERSION} ({len(features.FEATURE_NAMES)} features)'
            )
        self.schema = schema
        self.kind = schema['kind']
        self.classes = schema['classes']
        self.image_size = schema['image_size']
        self.tta = tta
        if self.kind == 'cnn':
            from tensorflow import keras
            self.model = keras.models.load_model(model_path, compile=False)
        elif self.kind == 'lightgbm':
            import lightgbm
            self.model = lightgbm.Booster(model_file=model_path)
        else:
            raise ValueError(f'Unknown model kind {self.kind!r}')

    @property
    def needs_images(self):
        return self.kind == 'cnn'

    def predict(self, images, vectors):
        """Class probabilities; images: float [n, S, S, 1] in [0, 1] (ignored for lightgbm)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.kind == 'lightgbm':
            return np.asarray(self.model.predict(vectors), dtype=np.float32).reshape(len(vectors), -1)
        inputs = [images]
        if self.tta:
            inputs += [images[:, ::-1], images[:, :, ::-1]]
        probs = [self.model.predict({'image': x, 'features': vectors}, verbose=0) for x in inputs]
        return np.mean(probs, axis=0).astype(np.float32)


def load_ensemble(path):
    """Configuration of an ensemble file, None for anything else (models, other JSON files)."""
    if not str(path).lower().endswith('.json') or not os.path.exists(path):
        return None
    with open(path) as f:
        config = json.load(f)
    return config if isinstance(config, dict) and config.get('format') == ENSEMBLE_FORMAT else None


class EnsemblePredictor:
    """Weighted mean of the class probabilities of v2 models with the same classes and image size."""

    kind = 'ensemble'

    def __init__(self, config_path, config, tta=False):
        base = os.path.dirname(os.path.abspath(config_path))
        self.members = []
        for member in config['members']:
            path = os.path.join(base, member['model'])
            schema = load_schema(path)
            if schema is None:
                raise ValueError(f'Ensemble member {path} has no schema file (v2 models only)')
            self.members.append(V2Predictor(path, schema, tta=tta))
        first = self.members[0]
        for predictor in self.members[1:]:
            if predictor.classes != first.classes or predictor.image_size != first.image_size:
                raise ValueError(f'Ensemble members of {config_path} differ in classes or image size')
        if config.get('classes') not in (None, first.classes):
            raise ValueError(f'Classes of {config_path} differ from those of its members')
        weights = np.array([float(member.get('weight', 1.0)) for member in config['members']], dtype=np.float64)
        self.weights = weights / weights.sum()
        self.schema = first.schema
        self.classes = first.classes
        self.image_size = first.image_size

    @property
    def needs_images(self):
        return any(predictor.needs_images for predictor in self.members)

    def predict(self, images, vectors):
        probs = [w * predictor.predict(images, vectors) for w, predictor in zip(self.weights, self.members)]
        return np.sum(probs, axis=0).astype(np.float32)


def load_predictor(model_path, tta=False):
    """
    (predictor, calibration) for an ensemble file or a v2 model; (None, None) for legacy models.
    calibration: {'temperature', 'thresholds'} fitted on val (evaluate_v2.py --save_calibration); entries
    are missing if they were not saved.
    """
    config = load_ensemble(model_path)
    if config is not None:
        calibration = {key: config[key] for key in ('temperature', 'thresholds') if key in config}
        return EnsemblePredictor(model_path, config, tta=tta), calibration
    schema = load_schema(model_path)
    if schema is None:
        if str(model_path).lower().endswith('.json'):
            raise ValueError(
                f'{model_path} is not an ensemble file (expected "format": "{ENSEMBLE_FORMAT}"; written by '
                'evaluate_v2.py --save_calibration as <stem>+<stem>_ensemble.json, not the --out report)'
            )
        return None, None
    stem = os.path.splitext(model_path)[0]
    calibration = {}
    if os.path.exists(stem + '_calibration.json'):
        with open(stem + '_calibration.json') as f:
            calibration['temperature'] = json.load(f)['temperature']
    if os.path.exists(stem + '_thresholds.json'):
        with open(stem + '_thresholds.json') as f:
            calibration['thresholds'] = json.load(f)
    return V2Predictor(model_path, schema, tta=tta), calibration
