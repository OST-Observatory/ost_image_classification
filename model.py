import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import numpy as np

class MultiModalClassifier:
    def __init__(self, input_shape=(256, 256, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        # Track whether model has been compiled with custom settings
        self._is_compiled = False
    
    def _build_feature_network(self):
        """Build the network for processing header and statistical features."""
        # Define input shapes based on the actual number of features
        num_features = 22  # Updated to match actual feature size
        num_confidences = 4  # Updated to match actual confidence size
        
        feature_input = layers.Input(shape=(num_features,))
        confidence_input = layers.Input(shape=(num_confidences,))
        
        # Process features and confidences separately first
        x_features = layers.BatchNormalization()(feature_input)
        x_features = layers.Dense(64, activation='relu')(x_features)
        
        x_conf = layers.Dense(64, activation='sigmoid')(confidence_input)
        
        # Combine features and confidences
        x = layers.Concatenate()([x_features, x_conf])
        
        # Further processing of combined features
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        return models.Model(inputs=[feature_input, confidence_input], outputs=x)
    
    def _build_image_network(self):
        """Build the CNN for processing image data."""
        image_input = layers.Input(shape=self.input_shape)
        
        # First Convolutional Block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Second Convolutional Block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Third Convolutional Block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Global average pooling for stability and parameter efficiency
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        return models.Model(inputs=image_input, outputs=x)
    
    def _build_model(self):
        """Build the combined model."""
        # Create feature and image networks
        feature_network = self._build_feature_network()
        image_network = self._build_image_network()
        
        # Define inputs (shapes aligned with feature network)
        image_input = layers.Input(shape=self.input_shape)
        feature_input = layers.Input(shape=(22,))
        confidence_input = layers.Input(shape=(4,))
        
        # Process inputs through respective networks
        image_features = image_network(image_input)
        feature_features = feature_network([feature_input, confidence_input])
        
        # Concatenate features
        combined = layers.Concatenate()([image_features, feature_features])
        
        # Final classification layers
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model (compile later to allow flexible training strategies)
        model = models.Model(
            inputs=[image_input, feature_input, confidence_input], 
            outputs=output
        )
        
        return model
    
    def compile(self, optimizer=None, loss=None, metrics=None):
        """Compile the underlying Keras model with flexible settings."""
        if optimizer is None:
            optimizer = 'adam'
        if loss is None:
            loss = 'categorical_crossentropy'
        if metrics is None:
            metrics = ['accuracy']
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._is_compiled = True

    def train(self, train_data, train_features, train_confidences, train_labels, 
              validation_data=None, epochs=50, batch_size=32, class_weight=None,
              optimizer=None, loss=None, metrics=None, use_plateau: bool | None = None):
        # Compile on-demand if custom optimizer/loss/metrics are provided or if not compiled yet
        if optimizer is not None or loss is not None or metrics is not None or not self._is_compiled:
            self.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        # Automatically disable ReduceLROnPlateau when a LR schedule is used
        lr_obj = getattr(self.model.optimizer, 'learning_rate', None)
        plateau_allowed = not isinstance(lr_obj, LearningRateSchedule)
        if use_plateau is not None:
            plateau_allowed = bool(use_plateau)
        if plateau_allowed:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6
                )
            )
        
        history = self.model.fit(
            [train_data, train_features, train_confidences],
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight
        )
        
        return history
    
    def predict(self, images, features, confidences):
        return self.model.predict([images, features, confidences])
    
    def save(self, path):
        # Save without optimizer to avoid serializing custom LR schedules
        self.model.save(path, include_optimizer=False)
    
    def load(self, path):
        # For inference/evaluation, compilation is not needed and can fail
        # with custom (non-registered) losses. Load without compiling.
        self.model = models.load_model(path, compile=False)