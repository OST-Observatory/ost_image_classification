import os
import numpy as np
from data_loader import FITSDataLoader
from model import FITSClassifier
import argparse

def classify_directory(model_path, input_dir, output_file=None):
    """Classify all FITS images in a directory."""
    # Initialize data loader and classifier
    data_loader = FITSDataLoader()
    classifier = FITSClassifier()
    classifier.load(model_path)
    
    # Get all FITS files
    fits_files = data_loader.get_all_fits_files(input_dir)
    
    # Prepare results
    results = []
    
    # Process each file
    for file_path in fits_files:
        try:
            # Load and preprocess image
            image = data_loader.load_fits(file_path)
            image = data_loader.preprocess_image(image)
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = classifier.predict(image)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Get class name
            class_name = list(data_loader.classes.keys())[predicted_class_idx]
            
            # Store result
            results.append({
                'file': file_path,
                'class': class_name,
                'confidence': float(confidence)
            })
            
            print(f"File: {file_path}")
            print(f"Predicted class: {class_name}")
            print(f"Confidence: {confidence:.4f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Save results if output file is specified
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Classify FITS images using trained model')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input', required=True, help='Input directory containing FITS files')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    classify_directory(args.model, args.input, args.output)

if __name__ == "__main__":
    main() 