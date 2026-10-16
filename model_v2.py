"""
CNN for feature schema v2: image branch + feature branch. Feature standardisation is part of the
model (Normalization layer with the training mean/variance), so inference needs only raw features.
"""
import numpy as np


def build_model(image_size, feature_mean, feature_variance, num_classes, base_filters=32, separable=False):
    from tensorflow import keras
    from tensorflow.keras import layers

    image_in = keras.Input((image_size, image_size, 1), name='image')
    feature_in = keras.Input((len(feature_mean),), name='features')

    x = image_in
    for i, filters in enumerate([base_filters, base_filters * 2, base_filters * 4, base_filters * 8]):
        conv = layers.SeparableConv2D if separable and i > 0 else layers.Conv2D
        x = conv(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    f = layers.Normalization(mean=feature_mean, variance=feature_variance, name='feature_scaling')(feature_in)
    f = layers.Dense(64, activation='relu')(f)
    f = layers.Dropout(0.2)(f)
    f = layers.Dense(64, activation='relu')(f)

    z = layers.Concatenate()([x, f])
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(num_classes, activation='softmax', name='class')(z)
    return keras.Model({'image': image_in, 'features': feature_in}, out)


def feature_moments(matrix):
    """Mean and variance per feature; near-constant features get variance 1 (no blow-up of noise)."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    variance = np.where(std < 1e-3 * (np.abs(mean) + 1.0), 1.0, std ** 2)
    return mean.astype(np.float32), variance.astype(np.float32)
