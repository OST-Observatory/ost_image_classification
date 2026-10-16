"""
Loading and prediction for v2 models (feature schema 2). A v2 model is recognised by its schema file
<model stem>_features.json (written by train.py / train_tabular.py); kind 'cnn' (Keras) or 'lightgbm'.
"""
import json
import os

import numpy as np

import features


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
