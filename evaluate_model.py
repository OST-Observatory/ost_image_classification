import os
import numpy as np
from data_loader import FITSDataLoader
from model import FITSClassifier
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def evaluate_model(model_path, data_dir, target_size=(350, 350)):
    # Initialize data loader
    data_loader = FITSDataLoader(target_size)
    
    # Load and prepare dataset
    print("Loading and preparing dataset...")
    images, features, confidences, labels = data_loader.prepare_dataset(data_dir)
    
    # Convert labels to one-hot encoding
    num_classes = len(data_loader.classes)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Load model
    print("Loading model...")
    model = FITSClassifier(input_shape=(*target_size, 1), num_classes=num_classes)
    model.load(model_path)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(images, features, confidences)
    
    # Calculate accuracy
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
    print(f"\nOverall accuracy: {accuracy:.4f}")
    
    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    for class_name, class_idx in data_loader.classes.items():
        class_mask = np.argmax(labels, axis=1) == class_idx
        if np.any(class_mask):
            class_accuracy = np.mean(
                np.argmax(predictions[class_mask], axis=1) == 
                np.argmax(labels[class_mask], axis=1)
            )
            print(f"{class_name}: {class_accuracy:.4f}")
        else:
            print(f"{class_name}: No samples available")

    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(np.argmax(labels, axis=1), np.argmax(predictions, axis=1),
                              target_names=list(data_loader.classes.keys())))
    
    # Plot confusion matrix
    plot_confusion_matrix(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), list(data_loader.classes.keys()))

if __name__ == "__main__":
    model_path = "fits_classifier.h5"
    data_dir = "../image_classification_test_sample"
    evaluate_model(model_path, data_dir) 