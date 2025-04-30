import os
import numpy as np
from data_loader import FITSDataLoader
from model import MultiModalClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
import logging

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Disable CUDA warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    # Configuration
    data_dir = "../image_classification_training_sample"  # Hier können Sie den Pfad zu Ihren Daten angeben
    target_size = (350, 350)  # Größe, auf die die Bilder skaliert werden
    
    # Initialize data loader
    data_loader = FITSDataLoader(target_size)
    
    # Load and prepare dataset
    print("Loading and preparing dataset...")
    images, features, confidences, labels = data_loader.prepare_dataset(data_dir)
    
    # Convert labels to one-hot encoding
    num_classes = len(data_loader.classes)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    
    # Split dataset
    X_train, X_test, F_train, F_test, C_train, C_test, y_train, y_test = train_test_split(
        images, features, confidences, labels, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    print("Initializing model...")
    model = MultiModalClassifier(input_shape=(*target_size, 1), num_classes=num_classes)
    
    print("Training model...")
    history = model.train(
        X_train, F_train, C_train, y_train,
        validation_data=([X_test, F_test, C_test], y_test),
        epochs=50,
        batch_size=32
    )
    
    # Save model
    print("Saving model...")
    model.save('multimodal_classifier.keras')
    
    # Evaluate model
    print("Evaluating model...")
    predictions = model.predict(X_test, F_test, C_test)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 