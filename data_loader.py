import numpy as np
from astropy.io import fits
import os
from typing import Tuple, Dict
import tensorflow as tf
from pathlib import Path
from scipy import stats
from PIL import Image

class FITSDataLoader:
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.classes = {
            'bias': 0,
            'darks': 1,
            'flats': 2,
            'deep_sky': 3,
            'flat_dados': 4,
            'spectrum_dados': 5,
            'wavelength_calibration_dados': 6,
            'flat_baches': 7,
            'spectrum_baches': 8,
            'wavelength_calibration_baches': 9,
            'einsteinturm': 10,
        }
        # Define all possible file extensions
        self.supported_extensions = {
            '.fit', '.FIT', '.fits', '.FITS',  # FITS extensions
            '.tif', '.TIF', '.tiff', '.TIFF'   # TIFF extensions
        }
        
        # Define expected ranges for different image types
        self.expected_ranges = {
            'bias': {
                'mean': (0, 1000),
                'std': (0, 100),
                'exposure_time': (0, 0.1)
            },
            'darks': {
                'mean': (0, 1000),
                'std': (0, 100),
                'exposure_time': (0.1, 3600)
            },
            'flats': {
                'mean': (1000, 50000),
                'std': (100, 1000),
                'exposure_time': (0.1, 10)
            },
            'deep_sky': {
                'mean': (0, 50000),
                'std': (0, 1000),
                'exposure_time': (10, 3600)
            }
        }
    
    def load_image(self, file_path: str) -> np.ndarray:
        """Load an image file (FITS or TIFF) and return the image data."""
        try:
            # Resolve any symbolic links
            resolved_path = os.path.realpath(file_path)
            
            # Check if file exists and is accessible
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f"File not found: {resolved_path}")
            
            if not os.access(resolved_path, os.R_OK):
                raise PermissionError(f"No read access to file: {resolved_path}")
            
            # Get file extension
            _, ext = os.path.splitext(resolved_path)
            ext = ext.lower()
            
            # Load based on file type
            if ext in {'.fit', '.fits'}:
                with fits.open(resolved_path) as hdul:
                    data = hdul[0].data
            elif ext in {'.tif', '.tiff'}:
                with Image.open(resolved_path) as img:
                    data = np.array(img)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            return data
        except Exception as e:
            print(f"Error loading image file {file_path}: {str(e)}")
            raise
    
    def calculate_image_statistics(self, image: np.ndarray) -> Dict:
        """Calculate statistical features from the image."""
        # Basic statistics
        mean = np.mean(image)
        std = np.std(image)
        median = np.median(image)
        min_val = np.min(image)
        max_val = np.max(image)
        
        # Histogram statistics
        hist, bins = np.histogram(image, bins=50)
        hist_norm = hist / np.sum(hist)
        
        # Noise estimation (using median absolute deviation)
        mad = stats.median_abs_deviation(image, axis=None)
        
        return {
            'mean': float(mean),
            'std': float(std),
            'median': float(median),
            'min': float(min_val),
            'max': float(max_val),
            'mad': float(mad),
            'hist_mean': float(np.mean(hist_norm)),
            'hist_std': float(np.std(hist_norm)),
            'hist_skew': float(stats.skew(hist_norm)),
            'hist_kurtosis': float(stats.kurtosis(hist_norm))
        }
    
    def calculate_header_confidence(self, header_values: Dict, image_stats: Dict) -> Dict:
        """Calculate confidence scores for header values based on image statistics."""
        confidences = {}
        
        # Check exposure time
        exp_time = header_values.get('exposure_time', 0)
        if exp_time > 0:
            # Compare with image statistics
            mean = image_stats['mean']
            std = image_stats['std']
            
            # Calculate expected ranges based on exposure time
            expected_mean = exp_time * 100  # Rough estimate
            expected_std = np.sqrt(exp_time * 100)  # Poisson noise
            
            # Calculate confidence
            mean_diff = abs(mean - expected_mean) / expected_mean
            std_diff = abs(std - expected_std) / expected_std
            
            confidences['exposure_time'] = 1.0 - min(1.0, (mean_diff + std_diff) / 2)
        else:
            confidences['exposure_time'] = 0.0
        
        # Check object type
        object_type = header_values.get('object', '').lower()
        if object_type:
            # Compare with image statistics
            mean = image_stats['mean']
            std = image_stats['std']
            
            # Check if statistics match expected ranges for the object type
            if object_type in self.expected_ranges:
                expected = self.expected_ranges[object_type]
                mean_in_range = expected['mean'][0] <= mean <= expected['mean'][1]
                std_in_range = expected['std'][0] <= std <= expected['std'][1]
                exp_in_range = expected['exposure_time'][0] <= exp_time <= expected['exposure_time'][1]
                
                confidences['object'] = (mean_in_range + std_in_range + exp_in_range) / 3
            else:
                confidences['object'] = 0.5  # Unknown object type
        else:
            confidences['object'] = 0.0
        
        # Check observation mode
        obsmode = header_values.get('obsmode', '').lower()
        if obsmode:
            # Basic checks based on common modes
            if 'bias' in obsmode and mean < 1000:
                confidences['obsmode'] = 0.9
            elif 'dark' in obsmode and 0.1 <= exp_time <= 3600:
                confidences['obsmode'] = 0.9
            elif 'flat' in obsmode and mean > 1000:
                confidences['obsmode'] = 0.9
            else:
                confidences['obsmode'] = 0.5
        else:
            confidences['obsmode'] = 0.0
        
        # Check filter
        filter_name = header_values.get('filter', '').lower()
        if filter_name:
            # Basic check: flats should have filter, darks and bias usually don't
            if ('flat' in obsmode or 'flat' in object_type) and filter_name:
                confidences['filter'] = 0.9
            elif ('dark' in obsmode or 'bias' in obsmode) and not filter_name:
                confidences['filter'] = 0.9
            else:
                confidences['filter'] = 0.5
        else:
            confidences['filter'] = 0.5
        
        return confidences
    
    def extract_header_features(self, file_path: str, image_stats: Dict) -> Tuple[Dict, Dict]:
        """Extract relevant features from image metadata and calculate their confidence."""
        try:
            # Define all possible features with default values
            features = {
                # Basic image statistics
                'mean': float(image_stats.get('mean', 0)),
                'std': float(image_stats.get('std', 0)),
                'median': float(image_stats.get('median', 0)),
                'min': float(image_stats.get('min', 0)),
                'max': float(image_stats.get('max', 0)),
                'mad': float(image_stats.get('mad', 0)),
                
                # Histogram statistics
                'hist_mean': float(image_stats.get('hist_mean', 0)),
                'hist_std': float(image_stats.get('hist_std', 0)),
                'hist_skew': float(image_stats.get('hist_skew', 0)),
                'hist_kurtosis': float(image_stats.get('hist_kurtosis', 0)),
                
                # Header/metadata features
                'exposure_time': 0.0,
                'gain': 0.0,
                'temperature': 0.0,
                'binning_x': 1,
                'binning_y': 1,
                'airmass': 0.0,
                'moon_phase': 0.0,
                'seeing': 0.0,
                
                # Additional features
                'is_tiff': 0.0,  # 0 for FITS, 1 for TIFF
                'file_size': float(os.path.getsize(file_path)),
                'width': float(image_stats.get('width', 0)),
                'height': float(image_stats.get('height', 0))
            }
            
            # Get file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext in {'.fit', '.fits'}:
                # FITS file - extract from header
                with fits.open(file_path) as hdul:
                    header = hdul[0].header
                    # Update features with FITS header values
                    features.update({
                        'exposure_time': float(header.get('EXPTIME', 0)),
                        'gain': float(header.get('GAIN', 0)),
                        'temperature': float(header.get('CCD-TEMP', 0)),
                        'airmass': float(header.get('AIRMASS', 0)),
                        'moon_phase': float(header.get('MOONPHAS', 0)),
                        'seeing': float(header.get('SEEING', 0)),
                        'is_tiff': 0.0
                    })
                    
                    # Handle binning
                    binning = header.get('BINNING', '1x1')
                    try:
                        x_bin, y_bin = map(int, binning.split('x'))
                        features['binning_x'] = x_bin
                        features['binning_y'] = y_bin
                    except:
                        features['binning_x'] = 1
                        features['binning_y'] = 1
            else:
                # TIFF file
                features['is_tiff'] = 1.0
                try:
                    with Image.open(file_path) as img:
                        features['width'] = float(img.width)
                        features['height'] = float(img.height)
                except:
                    pass
            
            # Calculate confidence scores
            confidences = self.calculate_header_confidence(features, image_stats)
            
            return features, confidences
        except Exception as e:
            print(f"Error extracting header features from {file_path}: {str(e)}")
            # Return default features
            return {
                'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
                'mad': 0.0, 'hist_mean': 0.0, 'hist_std': 0.0, 'hist_skew': 0.0,
                'hist_kurtosis': 0.0, 'exposure_time': 0.0, 'gain': 0.0,
                'temperature': 0.0, 'binning_x': 1, 'binning_y': 1, 'airmass': 0.0,
                'moon_phase': 0.0, 'seeing': 0.0, 'is_tiff': 0.0,
                'file_size': 0.0, 'width': 0.0, 'height': 0.0
            }, {
                'exposure_time': 0, 'object': 0, 'obsmode': 0, 'filter': 0
            }
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Preprocess the image data and extract features.
        Returns both the preprocessed image and extracted features."""
        try:
            # Calculate image statistics
            image_stats = self.calculate_image_statistics(image)
            
            # Handle different input dimensions
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            elif len(image.shape) == 3:
                if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                    image = np.transpose(image, (1, 2, 0))
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
            
            # Normalize the image
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            
            if image.shape[-1] > 3:
                image = image[..., :3]
            
            # Add batch dimension for tf.image.resize
            image = np.expand_dims(image, axis=0)
            image = tf.image.resize(image, self.target_size)
            image = image[0]
            
            return image, image_stats
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            print(f"Image shape: {image.shape}")
            raise
    
    def get_all_image_files(self, directory: str) -> list:
        """Get all image files (FITS and TIFF) in a directory."""
        image_files = []
        try:
            # Resolve the directory path
            resolved_dir = os.path.realpath(directory)
            
            # Walk through the directory
            for root, _, files in os.walk(resolved_dir):
                for file in files:
                    # Get file extension in lowercase for comparison
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in self.supported_extensions:
                        file_path = os.path.join(root, file)
                        image_files.append(file_path)
        except Exception as e:
            print(f"Error scanning directory {directory}: {str(e)}")
        
        return image_files
    
    def prepare_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the dataset for training or evaluation.
        Returns a tuple of (images, features, confidences, labels)."""
        images = []
        features = []
        confidences = []
        labels = []
        
        total_files = 0
        processed_files = 0
        
        # First count total number of files
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                total_files += len([f for f in os.listdir(class_dir) 
                                  if any(f.lower().endswith(ext) for ext in self.supported_extensions)])
        
        print(f"\nFound image files: {total_files}")
        print("Processing files...")
        
        # Process first file to get feature size
        first_file_processed = False
        expected_feature_size = None
        
        for class_name, class_idx in self.classes.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist. Skipping...")
                continue
                
            print(f"\nProcessing class: {class_name}")
            for filename in os.listdir(class_dir):
                if any(filename.lower().endswith(ext) for ext in self.supported_extensions):
                    file_path = os.path.join(class_dir, filename)
                    try:
                        # Load image and extract features
                        image = self.load_image(file_path)
                        processed_image, image_stats = self.preprocess_image(image)
                        header_features, header_confidences = self.extract_header_features(file_path, image_stats)
                        
                        # Combine all features
                        combined_features = {**header_features, **image_stats}
                        
                        # Print feature names for first file
                        if not first_file_processed:
                            print("\nFeature names:")
                            for name in combined_features.keys():
                                print(f"- {name}")
                            expected_feature_size = len(combined_features)
                            print(f"\nFeature size: {expected_feature_size}")
                            first_file_processed = True
                        
                        # Convert features to numeric values
                        numeric_features = []
                        for value in combined_features.values():
                            if isinstance(value, (int, float)):
                                numeric_features.append(float(value))
                            elif isinstance(value, str):
                                # Convert string to numeric value (e.g., using hash or length)
                                numeric_features.append(float(len(value)))
                            else:
                                # Default value for unknown types
                                numeric_features.append(0.0)
                        
                        # Check feature size consistency
                        if len(numeric_features) != expected_feature_size:
                            raise ValueError(f"Feature size mismatch: expected {expected_feature_size}, got {len(numeric_features)}")
                        
                        # Convert confidences to numeric values
                        numeric_confidences = []
                        for value in header_confidences.values():
                            if isinstance(value, (int, float)):
                                numeric_confidences.append(float(value))
                            else:
                                numeric_confidences.append(0.0)
                        
                        # Convert to numpy arrays
                        feature_array = np.array(numeric_features, dtype=np.float32)
                        confidence_array = np.array(numeric_confidences, dtype=np.float32)
                        
                        images.append(processed_image)
                        features.append(feature_array)
                        confidences.append(confidence_array)
                        labels.append(class_idx)
                        
                        processed_files += 1
                        if processed_files % 10 == 0:  # Status every 10 files
                            print(f"Progress: {processed_files}/{total_files} files processed "
                                  f"({(processed_files/total_files*100):.1f}%)")
                            
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        print(f"\nProcessing completed. Successfully processed {processed_files} of {total_files} files.")
        
        # Convert to numpy arrays
        images = np.array(images)
        features = np.array(features)
        confidences = np.array(confidences)
        labels = np.array(labels)
        
        return images, features, confidences, labels 