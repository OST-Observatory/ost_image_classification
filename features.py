"""
Feature pipeline v2 (schema version 2), shared by cache building, training and inference.

``extract(path, metadata, image_size)`` returns the model image (uint8, robustly stretched, square) and
the feature vector in ``FEATURE_NAMES`` order. Compared to the legacy pipeline (data_loader.py):
- FITS data is read unscaled via memmap and block-averaged, so large frames need little memory
- percentile clipping + asinh stretch instead of global min/max (hot pixels no longer flatten the image)
- new features: saturation, star count (light vs. dark), row/column band profile (spectra vs. darks),
  IMAGETYP/OBJECT keywords and archive metadata (instrument, plate solved, run spectroscopy)
- no shortcut features: file size, TIFF flag and the constant-valued legacy features are gone
"""
import math
import re

import numpy as np
from astropy.io import fits
from PIL import Image
from scipy import ndimage, signal

SCHEMA_VERSION = 2

# Class order (same as the legacy data_loader.FITSDataLoader.classes)
CLASSES = [
    'bias', 'darks', 'flats', 'deep_sky', 'flat_dados', 'spectrum_dados', 'wavelength_calibration_dados',
    'flat_baches', 'spectrum_baches', 'wavelength_calibration_baches', 'einsteinturm',
]

# Exposure type (archive codes) and spectrograph of each class
CLASS_EXPOSURE = {
    'bias': 'BI', 'darks': 'DA', 'flats': 'FL', 'deep_sky': 'LI', 'flat_dados': 'FL', 'spectrum_dados': 'LI',
    'wavelength_calibration_dados': 'WA', 'flat_baches': 'FL', 'spectrum_baches': 'LI',
    'wavelength_calibration_baches': 'WA', 'einsteinturm': 'LI',
}
CLASS_SPECTROGRAPH = {
    'flat_dados': 'D', 'spectrum_dados': 'D', 'wavelength_calibration_dados': 'D', 'flat_baches': 'B',
    'spectrum_baches': 'B', 'wavelength_calibration_baches': 'B', 'einsteinturm': 'E',
}
EXPOSURE_TYPES = ['BI', 'DA', 'FL', 'LI', 'WA']

IMAGETYP_CATEGORIES = ('bias', 'dark', 'flat', 'light', 'wave', 'other', 'none')
OBJECT_KEYWORDS = ('flat', 'dark', 'bias', 'arc', 'thar', 'neon', 'lamp', 'sky')
INSTRUMENT_PATTERNS = (
    ('qhy600', r'qhy\s*600'),
    ('qhy268', r'qhy\s*268'),
    ('stf8300', r'stf-?8300'),
    ('st8', r'st-?8\b'),
    ('st7', r'st-?7\b'),
    ('sti', r'st-?i\b'),
    ('asi2600', r'asi\s*2600'),
    ('asi294', r'asi\s*294'),
    ('apogee', r'apogee'),
)
INSTRUMENT_CATEGORIES = tuple(name for name, _ in INSTRUMENT_PATTERNS) + ('other', 'unknown')

STAT_NAMES = (
    'log_mean', 'log_median', 'log_std', 'log_mad', 'log_p01', 'log_p99', 'log_max',
    'contrast', 'saturated_fraction', 'median_level', 'saturation_known',
)
STRUCTURE_NAMES = (
    'log_star_count', 'band_contrast_major', 'band_contrast_minor', 'band_count', 'band_fraction',
    'band_horizontal',
)
HEADER_NAMES = (
    'log_exptime', 'exptime_zero', 'binning_x', 'binning_y', 'ccd_temp', 'gain', 'filter_present',
    'log_width', 'log_height',
)
METADATA_NAMES = ('plate_solved', 'run_spectroscopy', 'metadata_missing')

FEATURE_NAMES = (
    list(STAT_NAMES) + list(STRUCTURE_NAMES) + list(HEADER_NAMES)
    + [f'imagetyp_{c}' for c in IMAGETYP_CATEGORIES]
    + [f'object_{k}' for k in OBJECT_KEYWORDS]
    + [f'instrument_{c}' for c in INSTRUMENT_CATEGORIES]
    + list(METADATA_NAMES)
)

STATS_MAX_PIXELS = 1_000_000
ASINH_SOFTENING = 10.0


class FrameError(ValueError):
    """The file contains no usable image."""


# --- Reading ---------------------------------------------------------------------------------------

def _slog(value):
    value = float(value)
    return math.copysign(math.log1p(abs(value)), value) if math.isfinite(value) else 0.0


def _number(header, *keys, default=0.0):
    for key in keys:
        value = header.get(key)
        if value is None or value == '':
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            match = re.search(r'-?\d+(\.\d+)?', str(value))
            if match:
                return float(match.group(0))
    return default


def _binning(header):
    x = _number(header, 'XBINNING', 'XBIN', 'BINX', default=0)
    y = _number(header, 'YBINNING', 'YBIN', 'BINY', default=0)
    if not x or not y:
        match = re.match(r'\s*(\d+)\s*[xX]\s*(\d+)', str(header.get('BINNING', '')))
        if match:
            x, y = float(match.group(1)), float(match.group(2))
    return x or 1.0, y or 1.0


def _block_mean(data, factor, scale=1.0, zero=0.0):
    """Block-average a 2D array (memmap friendly: processed in row chunks) and apply BSCALE/BZERO."""
    h = data.shape[0] // factor * factor
    w = data.shape[1] // factor * factor
    out = np.empty((h // factor, w // factor), dtype=np.float32)
    rows = max(1, 2048 // factor) * factor
    for start in range(0, h, rows):
        block = np.asarray(data[start:start + rows, :w], dtype=np.float32)
        n = block.shape[0] // factor
        out[start // factor:start // factor + n] = block.reshape(n, factor, w // factor, factor).mean(axis=(1, 3))
    return out * scale + zero


def _saturation_level(header, bitpix, bzero):
    level = _number(header, 'SATURATE', 'DATAMAX', default=0)
    if level > 0:
        return level
    if bitpix == 8:
        return 255.0
    if bitpix == 16:
        return 65535.0 if bzero >= 32768 else 32767.0
    if bitpix == 32:
        return 4294967295.0 if bzero >= 2 ** 31 else 2147483647.0
    return None  # floating point data: unknown


def _luminance(array):
    if array.ndim == 3:
        # (planes, h, w) for FITS cubes, (h, w, channels) for RGB images
        axis = 0 if array.shape[0] <= 4 else -1
        return np.asarray(array, dtype=np.float32).mean(axis=axis)
    return array


def read_frame(path, image_size):
    """
    Read a FITS/TIFF/JPG frame: (reduced float image ~2*image_size, stats sample, header dict,
    saturation level or None, original (height, width)).
    """
    lower = str(path).lower()
    if lower.endswith(('.fit', '.fits', '.fts')):
        with fits.open(path, memmap=True, ignore_missing_end=True, do_not_scale_image_data=True) as hdul:
            hdu = next((h for h in hdul if h.data is not None and h.data.ndim >= 2), None)
            if hdu is None:
                raise FrameError('no image data')
            header = {k: v for k, v in hdu.header.items() if k}
            scale = float(hdu.header.get('BSCALE', 1.0) or 1.0)
            zero = float(hdu.header.get('BZERO', 0.0) or 0.0)
            data = _luminance(hdu.data)
            return _reduce(data, image_size, scale, zero, header,
                           _saturation_level(hdu.header, hdu.header.get('BITPIX'), zero))
    with Image.open(path) as img:
        bits = {'1': 1, 'L': 8, 'P': 8, 'RGB': 8, 'RGBA': 8, 'I;16': 16, 'I;16B': 16, 'I': 32}.get(img.mode)
        data = _luminance(np.asarray(img))
    level = float(2 ** bits - 1) if bits and bits < 32 else None
    return _reduce(data, image_size, 1.0, 0.0, {}, level)


def _reduce(data, image_size, scale, zero, header, level):
    h, w = data.shape[-2:]
    if h < 2 or w < 2:
        raise FrameError(f'image too small: {data.shape}')
    factor = max(1, min(h, w) // (2 * image_size))
    reduced = _block_mean(data, factor, scale, zero)
    step = max(1, int(math.sqrt(h * w / STATS_MAX_PIXELS)))
    sample = np.asarray(data[::step, ::step], dtype=np.float32) * scale + zero
    reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    return reduced, sample, header, level, (h, w)


# --- Image for the CNN -----------------------------------------------------------------------------

def stretch(reduced, sample, image_size):
    """Percentile clip (0.5/99.5) + asinh stretch, resized to image_size x image_size, as uint8."""
    lo, hi = np.percentile(sample, [0.5, 99.5])
    if not np.isfinite(hi - lo) or hi - lo < 1e-6:
        scaled = np.zeros_like(reduced)
    else:
        scaled = np.clip((reduced - lo) / (hi - lo), 0.0, 1.0)
        scaled = np.arcsinh(ASINH_SOFTENING * scaled) / np.arcsinh(ASINH_SOFTENING)
    resized = Image.fromarray(scaled.astype(np.float32), mode='F').resize((image_size, image_size), Image.BILINEAR)
    return np.clip(np.asarray(resized) * 255.0 + 0.5, 0, 255).astype(np.uint8)


# --- Features --------------------------------------------------------------------------------------

def image_statistics(sample, level):
    p01, median, p99 = np.percentile(sample, [1, 50, 99])
    mad = float(np.median(np.abs(sample - median)))
    stats = {
        'log_mean': _slog(sample.mean()), 'log_median': _slog(median), 'log_std': _slog(sample.std()),
        'log_mad': _slog(mad), 'log_p01': _slog(p01), 'log_p99': _slog(p99), 'log_max': _slog(sample.max()),
        'contrast': _slog((p99 - median) / (1.4826 * mad + 1e-3)),
        'saturated_fraction': 0.0, 'median_level': 0.0, 'saturation_known': 0.0,
    }
    if level:
        stats['saturated_fraction'] = float(np.mean(sample >= 0.98 * level))
        stats['median_level'] = float(np.clip(median / level, -1.0, 2.0))
        stats['saturation_known'] = 1.0
    return stats


def structure_features(reduced):
    """Star count (lights) and band profile (spectra: bright rows or columns in an otherwise dark frame)."""
    small = reduced
    factor = max(1, min(small.shape) // 512)
    if factor > 1:
        small = _block_mean(small, factor)
    background = float(np.median(small))
    sigma = 1.4826 * float(np.median(np.abs(small - background))) + 1e-6
    peaks = (small == ndimage.maximum_filter(small, size=5)) & (small > background + 5 * sigma)
    star_count = int(peaks.sum())

    def profile(axis):
        prof = np.median(small, axis=axis)
        base = float(np.median(prof))
        spread = 1.4826 * float(np.median(np.abs(prof - base))) + 1e-6
        contrast = (float(prof.max()) - base) / sigma
        found, _ = signal.find_peaks(prof, prominence=max(3 * spread, 3 * sigma / math.sqrt(prof.size)))
        fraction = float(np.mean(prof > base + 3 * sigma / math.sqrt(prof.size)))
        return contrast, len(found), fraction

    rows = profile(axis=1)  # one value per row: horizontal bands
    cols = profile(axis=0)
    horizontal = rows[0] >= cols[0]
    major, minor = (rows, cols) if horizontal else (cols, rows)
    return {
        'log_star_count': math.log1p(star_count),
        'band_contrast_major': _slog(major[0]),
        'band_contrast_minor': _slog(minor[0]),
        'band_count': math.log1p(major[1]),
        'band_fraction': major[2],
        'band_horizontal': 1.0 if horizontal else 0.0,
    }


def normalize_imagetyp(value):
    text = str(value or '').strip().lower()
    if not text:
        return 'none'
    if 'dark' in text:  # also dark flats
        return 'dark'
    if 'bias' in text or 'zero' in text:
        return 'bias'
    if 'flat' in text:
        return 'flat'
    if any(k in text for k in ('arc', 'comp', 'lamp', 'thar', 'wave', 'calib')):
        return 'wave'
    if any(k in text for k in ('light', 'object', 'science')):
        return 'light'
    return 'other'


def instrument_category(name):
    text = str(name or '').strip().lower()
    if text in ('', '-', 'uk', 'unknown'):
        return 'unknown'
    for category, pattern in INSTRUMENT_PATTERNS:
        if re.search(pattern, text):
            return category
    return 'other'


def header_features(header, shape, metadata):
    exptime = _number(header, 'EXPTIME', 'EXPOSURE', 'EXPOSURE_TIME', default=-1)
    binning_x, binning_y = _binning(header)
    feats = {
        'log_exptime': math.log1p(max(exptime, 0.0)),
        'exptime_zero': 1.0 if 0 <= exptime < 1e-3 else 0.0,
        'binning_x': binning_x, 'binning_y': binning_y,
        'ccd_temp': _number(header, 'CCD-TEMP', 'CCDTEMP', 'SET-TEMP', default=0.0),
        'gain': _slog(_number(header, 'GAIN', 'EGAIN', default=0.0)),
        'filter_present': 1.0 if str(header.get('FILTER', '') or '').strip() else 0.0,
        'log_width': math.log1p(shape[1]), 'log_height': math.log1p(shape[0]),
    }
    imagetyp = normalize_imagetyp(header.get('IMAGETYP'))
    for category in IMAGETYP_CATEGORIES:
        feats[f'imagetyp_{category}'] = 1.0 if category == imagetyp else 0.0
    obj = str(header.get('OBJECT', '') or '').lower()
    for keyword in OBJECT_KEYWORDS:
        feats[f'object_{keyword}'] = 1.0 if keyword in obj else 0.0

    metadata = metadata or {}
    instrument = metadata.get('instrument') or header.get('INSTRUME', '')
    category = instrument_category(instrument)
    for name in INSTRUMENT_CATEGORIES:
        feats[f'instrument_{name}'] = 1.0 if name == category else 0.0
    feats['plate_solved'] = 1.0 if metadata.get('plate_solved') else 0.0
    feats['run_spectroscopy'] = 1.0 if metadata.get('run_spectroscopy') else 0.0
    feats['metadata_missing'] = 0.0 if metadata else 1.0
    return feats


def extract(path, metadata=None, image_size=448):
    """(image uint8 [image_size, image_size], features float32 [len(FEATURE_NAMES)]) for one file."""
    reduced, sample, header, level, shape = read_frame(path, image_size)
    feats = {}
    feats.update(image_statistics(sample, level))
    feats.update(structure_features(reduced))
    feats.update(header_features(header, shape, metadata))
    vector = np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)
    return stretch(reduced, sample, image_size), np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def schema(image_size, kind, classes=None, **extra):
    """Model schema stored next to a v2 model as <model stem>_features.json."""
    return {
        'schema_version': SCHEMA_VERSION, 'kind': kind, 'image_size': image_size,
        'classes': list(classes or CLASSES), 'feature_names': list(FEATURE_NAMES), **extra,
    }
