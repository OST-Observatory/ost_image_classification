import os
import math
import numpy as np
from data_loader import FITSDataLoader
from model import MultiModalClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import logging
import argparse

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Disable CUDA warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def focal_loss(alpha_vector, gamma=2.0):
    """Create a focal loss function with class-specific alpha.
    alpha_vector: 1D tensor of shape [num_classes]
    """
    alpha_vector = tf.constant(alpha_vector, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # standard categorical crossentropy per-sample
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce)
        # alpha selection per sample (y_true is one-hot)
        alpha = tf.reduce_sum(y_true * alpha_vector, axis=1)
        focal = alpha * tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(focal)

    return loss_fn


def build_optimizer_with_cosine(initial_lr=1e-3, first_decay_epochs=5, steps_per_epoch=100):
    total_first_decay_steps = max(1, int(first_decay_epochs * steps_per_epoch))
    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=total_first_decay_steps,
        t_mul=2.0,
        m_mul=0.8,
        alpha=1e-4,
    )
    return tf.keras.optimizers.Adam(learning_rate=schedule)


def main():
    parser = argparse.ArgumentParser(description="Train multimodal classifier with accuracy-focused, RAM-neutral strategies")
    parser.add_argument("--data_dir", default="../image_classification_training_sample", help="Pfad zu Trainingsdaten")
    parser.add_argument("--target_w", type=int, default=448, help="Bildbreite")
    parser.add_argument("--target_h", type=int, default=448, help="Bildhöhe")
    parser.add_argument("--epochs", type=int, default=50, help="Anzahl Epochen")
    parser.add_argument("--batch_size", type=int, default=24, help="Batchgröße")
    parser.add_argument("--loss_strategy", choices=["none", "class_weights", "focal_loss"], default="class_weights", help="Strategie zur Behandlung von Klassenungleichgewicht")
    parser.add_argument("--label_smoothing", type=float, default=0.08, help="Label Smoothing für CE (bei focal_loss ignoriert)")
    # Class-weight smoothing/clipping controls
    parser.add_argument("--cw_beta", type=float, default=0.3, help="Potenz für Glättung der Klassen-Gewichte (0 = keine Wirkung, 1 = original)")
    parser.add_argument("--cw_cap", type=float, default=4.0, help="Maximalfaktor relativ zur Median-Gewicht (Cap = cap * Median)")
    parser.add_argument("--initial_lr", type=float, default=1e-3, help="Initiale Lernrate (Basis bei batch_ref)")
    parser.add_argument("--lr_schedule", choices=["cosine", "constant"], default="constant", help="LR-Strategie")
    parser.add_argument("--batch_ref", type=int, default=32, help="Referenz-Batchgröße für LR-Skalierung")
    parser.add_argument("--model_out", default="multimodal_classifier.keras", help="Speicherpfad für Modell")
    parser.add_argument("--augment", action="store_true", help="Aktiviere On-the-fly Augmentation (nur Training)")
    parser.add_argument("--aug_strength", type=float, default=0.5, help="Augmentierungsstärke [0..1] (steuert Helligkeit/Kontrast/Shift)")
    parser.add_argument("--aug_conservative", type=str, default=None, help="Komma-separierte Klassen mit konservativer Augmentation (z. B. 'darks,flats')")

    args = parser.parse_args()
    # Configuration
    data_dir = args.data_dir
    target_size = (args.target_h, args.target_w)
    
    # Initialize data loader
    data_loader = FITSDataLoader(target_size)
    # Parse conservative augmentation classes to indices
    conservative_indices = []
    if args.augment:
        if args.aug_conservative:
            names = [n.strip() for n in str(args.aug_conservative).split(',') if n.strip()]
            for n in names:
                if n in data_loader.classes:
                    conservative_indices.append(int(data_loader.classes[n]))
        else:
            # Default to common conservative classes if present
            for n in ["darks", "flats"]:
                if n in data_loader.classes:
                    conservative_indices.append(int(data_loader.classes[n]))
    
    # Load and prepare dataset
    print("Loading and preparing dataset...")
    images, features, confidences, labels = data_loader.prepare_dataset(data_dir)
    
    # Convert labels to one-hot encoding
    num_classes = len(data_loader.classes)
    labels_int = labels.astype(np.int32)
    
    # Split dataset
    X_train, X_test, F_train, F_test, C_train, C_test, y_train_int, y_test_int = train_test_split(
        images, features, confidences, labels_int, test_size=0.2, random_state=42, stratify=labels_int
    )
    # One-hot erst NACH stratified split
    y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test_int, num_classes=num_classes)

    # Feature-Standardisierung (z-score) auf Trainingssplit fitten und anwenden
    feat_mean = np.mean(F_train, axis=0)
    feat_std = np.std(F_train, axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    F_train = (F_train - feat_mean) / feat_std
    F_test = (F_test - feat_mean) / feat_std
    
    # Initialize and train model
    print("Initializing model...")
    model = MultiModalClassifier(input_shape=(*target_size, 1), num_classes=num_classes)
    
    print("Training model...")
    # Optimizer mit CosineDecayRestarts
    steps_per_epoch = max(1, math.ceil(len(X_train) / max(1, args.batch_size)))
    # Scale learning rate with batch size
    lr_scaled = args.initial_lr * (args.batch_size / max(1, args.batch_ref))
    if args.lr_schedule == "cosine":
        optimizer = build_optimizer_with_cosine(initial_lr=lr_scaled,
                                                first_decay_epochs=5,
                                                steps_per_epoch=steps_per_epoch)
    else:
        # Stable baseline optimizer
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_scaled, weight_decay=1e-4)

    # Loss nach Strategie
    # Build tf.data datasets with cache/prefetch. Include sample weights for class_weights strategy.
    def make_dataset(X, F, C, Y, *, batch_size, shuffle=False, cache=True, sample_weights=None, augment=False, conservative_idxs=None):
        if sample_weights is not None:
            ds = tf.data.Dataset.from_tensor_slices(((X, F, C), Y, sample_weights))
        else:
            ds = tf.data.Dataset.from_tensor_slices(((X, F, C), Y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(X), 10000))
        if cache:
            ds = ds.cache()
        if augment:
            # Use Python float inside map fns to avoid symbolic Tensor in Python bool contexts
            aug_s_val = float(args.aug_strength)
            # per-class conservative list from CLI (fallback to [-1] if empty)
            conservative_classes = tf.constant(conservative_idxs if (conservative_idxs and len(conservative_idxs) > 0) else [-1], dtype=tf.int64)

            def _apply_aug(img, y_onehot):
                # Class index from one-hot
                cls_idx = tf.argmax(y_onehot, axis=-1, output_type=tf.int64)
                is_conservative = tf.reduce_any(tf.equal(tf.expand_dims(cls_idx, 0), conservative_classes))
                # Common flips
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)
                # Conditional stronger augs for non-conservative classes
                def strong_aug():
                    # Random rot90
                    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
                    x = tf.image.rot90(img, k)
                    # Small translation via roll (up to ~3% image size scaled by aug_strength)
                    shape = tf.shape(x)
                    h = tf.cast(shape[0], tf.float32)
                    w = tf.cast(shape[1], tf.float32)
                    max_shift = tf.cast(tf.math.maximum(1.0, tf.round(0.03 * tf.maximum(h, w) * aug_s_val)), tf.int32)
                    dx = tf.random.uniform([], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32)
                    dy = tf.random.uniform([], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32)
                    x = tf.roll(x, shift=[dy, dx], axis=[0, 1])
                    # Brightness/contrast
                    x = tf.image.random_brightness(x, max_delta=0.10 * aug_s_val)
                    lower = max(0.5, 1.0 - 0.10 * aug_s_val)
                    upper = 1.0 + 0.10 * aug_s_val
                    x = tf.image.random_contrast(x, lower=float(lower), upper=float(upper))
                    return x
                def light_aug():
                    x = tf.image.random_brightness(img, max_delta=0.05 * aug_s_val)
                    return x
                img = tf.cond(is_conservative, light_aug, strong_aug)
                img = tf.clip_by_value(img, 0.0, 1.0)
                return img

            def _aug_no_w(inputs, y):
                img, feat, conf = inputs
                img = _apply_aug(img, y)
                return (img, feat, conf), y
            def _aug_with_w(inputs, y, w):
                img, feat, conf = inputs
                img = _apply_aug(img, y)
                return (img, feat, conf), y, w
            if sample_weights is not None:
                ds = ds.map(_aug_with_w, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                ds = ds.map(_aug_no_w, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # Choose loss
    use_plateau = (args.lr_schedule != "cosine")
    if args.loss_strategy == "none":
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        sw_train = None
    elif args.loss_strategy == "class_weights":
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        # Class weights aus Trainingslabels (int)
        cls_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_train_int)
        # Glättung per Potenz (w_i := w_i^beta) und Clipping relativ zur Median
        beta = max(0.0, float(args.cw_beta))
        cap_factor = float(args.cw_cap)
        smoothed = np.asarray(cls_weights, dtype=np.float32)
        if beta != 1.0 and beta != 0.0:
            smoothed = np.power(smoothed, beta)
        if cap_factor > 0:
            med = np.median(smoothed)
            max_allowed = cap_factor * med
            smoothed = np.minimum(smoothed, max_allowed)
        mean_orig = float(np.mean(cls_weights))
        mean_new = float(np.mean(smoothed))
        scale = mean_orig / max(1e-8, mean_new)
        smoothed *= scale
        # Build per-sample weights from class weights
        sw_train = smoothed[y_train_int]
    else:
        # focal loss mit alpha vektor (inverse Häufigkeit)
        class_counts = np.bincount(y_train_int, minlength=num_classes).astype(np.float32)
        inv_freq = 1.0 / np.maximum(class_counts, 1.0)
        alpha_vector = inv_freq / np.mean(inv_freq)
        loss = focal_loss(alpha_vector=alpha_vector, gamma=2.0)
        sw_train = None

    train_ds = make_dataset(X_train, F_train, C_train, y_train, batch_size=args.batch_size, shuffle=True, cache=True, sample_weights=sw_train, augment=bool(args.augment), conservative_idxs=conservative_indices)
    val_ds = make_dataset(X_test, F_test, C_test, y_test, batch_size=args.batch_size, shuffle=False, cache=True, sample_weights=None, augment=False, conservative_idxs=[])

    # Compile and fit using dataset API
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    if use_plateau:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6))
    history = model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the model
    print("Saving model...")
    model_path = args.model_out
    model.save(model_path)
    
    print("\nTraining completed. Model saved to:", model_path)
    # Save feature scaler for evaluation/classification consistency
    scaler_path = os.path.splitext(model_path)[0] + "_feat_scaler.npz"
    np.savez(scaler_path, mean=feat_mean, std=feat_std)
    print("Saved feature scaler to:", scaler_path)
    
    # Evaluate model
    print("Evaluating model...")
    predictions = model.predict(X_test, F_test, C_test)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 