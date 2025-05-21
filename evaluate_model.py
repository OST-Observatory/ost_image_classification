import os
import numpy as np
from data_loader import FITSDataLoader
from model import MultiModalClassifier
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix with detailed analysis."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Detailed confusion matrix analysis
    print("\nDetailed Confusion Matrix Analysis:")
    for i, class_name in enumerate(classes):
        true_positives = cm[i, i]
        false_positives = sum(cm[:, i]) - true_positives
        false_negatives = sum(cm[i, :]) - true_positives
        true_negatives = sum(sum(cm)) - (true_positives + false_positives + false_negatives)
        
        print(f"\n{class_name}:")
        print(f"  True Positives: {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  False Negatives: {false_negatives}")
        print(f"  True Negatives: {true_negatives}")
        
        # Find most confused classes
        if false_positives > 0 or false_negatives > 0:
            print("  Most confused with:")
            for j, other_class in enumerate(classes):
                if i != j:
                    if cm[i, j] > 0:  # False negatives
                        print(f"    - Predicted as {other_class}: {cm[i, j]} times")
                    if cm[j, i] > 0:  # False positives
                        print(f"    - Misclassified as {class_name}: {cm[j, i]} times")

def analyze_feature_importance(model, features, labels, feature_names):
    """Analyze feature importance using gradient-based importance."""
    print("\nAnalyzing feature importance...")
    
    # Convert features to tensor
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    confidences_tensor = tf.convert_to_tensor(np.zeros((features.shape[0], 4)), dtype=tf.float32)  # Dummy confidences
    
    # Get gradients of predictions with respect to features
    with tf.GradientTape() as tape:
        tape.watch(features_tensor)
        # Get predictions for the feature branch
        feature_predictions = model.model.get_layer('functional')([features_tensor, confidences_tensor])
    
    # Calculate gradients
    gradients = tape.gradient(feature_predictions, features_tensor)
    
    # Calculate importance as mean absolute gradient
    importance = np.mean(np.abs(gradients.numpy()), axis=0)
    
    # Create DataFrame with results
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

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
    model = MultiModalClassifier(input_shape=(*target_size, 1), num_classes=num_classes)
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
    
    # Plot confusion matrix with detailed analysis
    plot_confusion_matrix(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), 
                         list(data_loader.classes.keys()))
    
    # Analyze feature importance
    feature_names = [f"feature_{i}" for i in range(features.shape[1])]
    analyze_feature_importance(model, features, np.argmax(labels, axis=1), feature_names)

if __name__ == "__main__":
    model_path = "multimodal_classifier.keras"
    data_dir = "../image_classification_test_sample"
    evaluate_model(model_path, data_dir) 