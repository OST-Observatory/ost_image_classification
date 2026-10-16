"""Synthetic frames for the v2 pipeline tests (small, written with astropy)."""
import json
import os
import sys

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHAPE = (300, 400)


def make_frame(kind, rng):
    """Raw ADU frame (uint16 range) of a given kind."""
    data = rng.normal(1000, 15, SHAPE)
    if kind == 'light':
        yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
        for _ in range(40):
            y, x = rng.integers(10, SHAPE[0] - 10), rng.integers(10, SHAPE[1] - 10)
            data += 8000 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 4.0)
    elif kind == 'spectrum':
        data[140:160, :] += 6000  # one bright horizontal band (long-slit spectrum)
    elif kind == 'saturated':
        data[:] = 65535
    elif kind == 'flat':
        data += 30000
    return np.clip(data, 0, 65535)


def write_fits(path, kind, rng, header=None):
    hdu = fits.PrimaryHDU(make_frame(kind, rng).astype(np.uint16))  # stored as int16 + BZERO 32768
    for key, value in (header or {}).items():
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)
    return path


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def frames(tmp_path, rng):
    """Paths of one synthetic FITS file per kind."""
    headers = {
        'dark': {'IMAGETYP': 'Dark Frame', 'EXPTIME': 300.0, 'INSTRUME': 'QHY600M'},
        'light': {'IMAGETYP': 'Light Frame', 'EXPTIME': 60.0, 'FILTER': 'V', 'INSTRUME': 'QHY600M'},
        'spectrum': {'IMAGETYP': 'LIGHT', 'EXPTIME': 600.0, 'OBJECT': 'Vega', 'INSTRUME': 'QHY268M'},
        'saturated': {'EXPTIME': 150.0},
        'flat': {'IMAGETYP': 'FLAT', 'EXPTIME': 2.0, 'XBINNING': 2, 'YBINNING': 2},
    }
    return {kind: write_fits(str(tmp_path / f'{kind}.fits'), kind, rng, header) for kind, header in headers.items()}


def write_export(path, records):
    with open(path, 'w') as f:
        json.dump({'format': 'ost-classification-export/v2', 'params': {}, 'files': records}, f)
    return str(path)


@pytest.fixture
def small_cache(tmp_path, rng):
    """Cache (image size 32) with darks, lights and spectra in train/val/test."""
    import build_cache

    archive = tmp_path / 'archive'
    archive.mkdir()
    records = []
    plan = [('darks', 'dark'), ('deep_sky', 'light'), ('spectrum_dados', 'spectrum')]
    for split, count in (('train', 6), ('val', 3), ('test', 3)):
        for cls, kind in plan:
            for i in range(count):
                name = f'{split}_{cls}_{i}.fits'
                write_fits(str(archive / name), kind, rng, {'EXPTIME': 10.0})
                records.append({
                    'path': name, 'class': cls, 'split': split, 'hard': i == 0, 'run': i,
                    'instrument': 'QHY 600M', 'year': 2024, 'plate_solved': kind == 'light',
                    'run_spectroscopy': kind == 'spectrum',
                })
    export = write_export(tmp_path / 'export.json', records)
    cache = str(tmp_path / 'cache')
    build_cache.build(export, str(archive), cache, image_size=32, workers=2, shard_size=10)
    return {'cache': cache, 'archive': str(archive), 'export': export, 'records': records}
