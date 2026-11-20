"""Feature pipeline v2 on synthetic frames."""
import numpy as np
import pytest
from PIL import Image

import features


def _feats(path, metadata=None):
    image, vector = features.extract(path, metadata=metadata, image_size=64)
    return image, dict(zip(features.FEATURE_NAMES, vector))


def test_vector_matches_schema(frames):
    image, vector = features.extract(frames['dark'], image_size=64)
    assert image.shape == (64, 64) and image.dtype == np.uint8
    assert vector.shape == (len(features.FEATURE_NAMES),)
    assert np.isfinite(vector).all()
    assert len(set(features.FEATURE_NAMES)) == len(features.FEATURE_NAMES)


def test_stars_separate_lights_from_darks(frames):
    _, dark = _feats(frames['dark'])
    _, light = _feats(frames['light'])
    assert light['log_star_count'] > dark['log_star_count'] + 1.0


def test_bands_separate_spectra_from_darks(frames):
    _, dark = _feats(frames['dark'])
    _, spectrum = _feats(frames['spectrum'])
    assert spectrum['band_contrast_major'] > dark['band_contrast_major'] + 1.0
    assert spectrum['band_horizontal'] == 1.0


def test_saturation_from_bzero(frames):
    _, saturated = _feats(frames['saturated'])
    _, flat = _feats(frames['flat'])
    assert saturated['saturation_known'] == 1.0
    assert saturated['saturated_fraction'] > 0.99
    assert flat['saturated_fraction'] == 0.0
    assert 0.4 < flat['median_level'] < 0.6


def test_header_and_metadata(frames):
    _, dark = _feats(frames['dark'])
    assert dark['imagetyp_dark'] == 1.0 and dark['imagetyp_light'] == 0.0
    assert dark['instrument_qhy600'] == 1.0 and dark['metadata_missing'] == 1.0
    _, flat = _feats(frames['flat'], metadata={'instrument': 'SBIG ST-8', 'plate_solved': True})
    assert (flat['binning_x'], flat['binning_y']) == (2.0, 2.0)
    assert flat['instrument_st8'] == 1.0 and flat['plate_solved'] == 1.0 and flat['metadata_missing'] == 0.0
    _, spectrum = _feats(frames['spectrum'])
    assert spectrum['instrument_qhy268'] == 1.0


@pytest.mark.parametrize('value,expected', [
    ('Dark Frame', 'dark'), ('DARKFLAT', 'dark'), ('Flat Field', 'flat'), ('Bias Frame', 'bias'),
    ('Light Frame', 'light'), ('Object', 'light'), ('ThAr', 'wave'), ('', 'none'), ('whatever', 'other'),
])
def test_normalize_imagetyp(value, expected):
    assert features.normalize_imagetyp(value) == expected


def test_tiff_and_rgb(tmp_path, rng):
    gray = (rng.random((120, 160)) * 65535).astype(np.uint16)
    Image.fromarray(gray).save(tmp_path / 'g.tif')
    rgb = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / 'c.jpg')
    for name in ('g.tif', 'c.jpg'):
        image, vector = features.extract(str(tmp_path / name), image_size=32)
        assert image.shape == (32, 32) and np.isfinite(vector).all()


def test_unreadable_file(tmp_path):
    bad = tmp_path / 'bad.fits'
    bad.write_bytes(b'not a fits file')
    with pytest.raises(Exception):
        features.extract(str(bad))


@pytest.mark.parametrize('shape', [(1148, 1500), (2105, 3140), (7, 9)])
def test_block_mean_crops_rows_not_divisible_by_factor(shape):
    data = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    factor = 3 if shape[0] < 2000 else 2
    reduced = features._block_mean(data, factor)
    h, w = shape[0] // factor, shape[1] // factor
    assert reduced.shape == (h, w)
    expected = data[:h * factor, :w * factor].reshape(h, factor, w, factor).mean(axis=(1, 3))
    assert np.allclose(reduced, expected)


def test_large_frame_with_odd_height(tmp_path):
    from astropy.io import fits
    path = tmp_path / 'qhy.fit'
    fits.writeto(path, np.random.default_rng(0).integers(0, 4000, (1148, 4788)).astype(np.uint16))
    image, vector = features.extract(str(path), image_size=32)
    assert image.shape == (32, 32) and len(vector) == len(features.FEATURE_NAMES)
