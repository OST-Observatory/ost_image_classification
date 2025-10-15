import os
import json
import numpy as np
from data_loader import FITSDataLoader
from model import MultiModalClassifier
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib
matplotlib.use("Agg")  # Headless Backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

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


def parse_thresholds_arg(thresholds_arg, num_classes):
    """Parse thresholds argument: scalar, comma-separated list, or JSON file path."""
    if thresholds_arg is None:
        return None
    # Try float scalar
    try:
        scalar = float(thresholds_arg)
        return np.full(num_classes, scalar, dtype=np.float32)
    except ValueError:
        pass
    # Try comma-separated list
    if "," in thresholds_arg:
        parts = [p.strip() for p in thresholds_arg.split(",")]
        arr = np.array([float(p) for p in parts], dtype=np.float32)
        if arr.size != num_classes:
            raise ValueError(f"Threshold list length {arr.size} does not match num_classes {num_classes}")
        return arr
    # Try JSON file
    if os.path.exists(thresholds_arg):
        with open(thresholds_arg, 'r') as f:
            data = json.load(f)
        # Accept list or dict mapping class names to threshold
        if isinstance(data, list):
            arr = np.array([float(x) for x in data], dtype=np.float32)
            if arr.size != num_classes:
                raise ValueError(f"Threshold list length {arr.size} does not match num_classes {num_classes}")
            return arr
        elif isinstance(data, dict):
            # sort by class index inferred from provided mapping (name->idx may be order of FITSDataLoader)
            return data  # handle later when class order is known
    raise ValueError("Could not parse --thresholds. Provide scalar, comma list, or JSON path.")


def compute_pr_curves_per_class(y_true_int, y_proba, class_names, out_dir="."):
    """Compute PR curves and F1 for each class and save plots. Return dict of suggestions by max F1."""
    os.makedirs(out_dir, exist_ok=True)
    num_classes = y_proba.shape[1]
    suggestions = {}
    for c in range(num_classes):
        y_true_bin = (y_true_int == c).astype(np.int32)
        precision, recall, thresholds = precision_recall_curve(y_true_bin, y_proba[:, c])
        # thresholds length = len(precision)-1
        # compute F1 for each threshold-aligned point
        eps = 1e-8
        f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], eps)
        if f1.size > 0:
            best_idx = int(np.argmax(f1))
            best_thr = float(thresholds[best_idx])
            suggestions[class_names[c]] = {
                'best_f1_threshold': best_thr,
                'best_f1': float(f1[best_idx]),
                'precision_at_best': float(precision[best_idx]),
                'recall_at_best': float(recall[best_idx]),
            }
        # Plot PR curve
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"PR {class_names[c]}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall: {class_names[c]}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'pr_curve_{class_names[c]}.png'))
        plt.close()
    return suggestions


def apply_thresholds_to_predictions(proba, thresholds, strategy="gate", abstain=False, unknown_id=None):
    """Apply per-class thresholds to probability predictions.
    strategy 'gate': choose highest class that passes its threshold; if none, fallback to argmax.
    """
    num_samples, num_classes = proba.shape
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape((1, num_classes))
    # mask of classes that pass
    pass_mask = proba >= thresholds
    # pick highest prob among passing classes
    top_idx = np.argmax(proba, axis=1)
    gated_idx = []
    for i in range(num_samples):
        passing = np.where(pass_mask[i])[0]
        if passing.size > 0:
            # among passing, pick argmax
            j = passing[np.argmax(proba[i, passing])]
            gated_idx.append(int(j))
        else:
            if abstain:
                # Unknown-ID defaults to last index if not provided
                fallback_unknown = num_classes if unknown_id is None else int(unknown_id)
                gated_idx.append(fallback_unknown)
            else:
                gated_idx.append(int(top_idx[i]))
    return np.array(gated_idx, dtype=np.int32)

def evaluate_model(model_path, data_dir, target_size=(350, 350), thresholds=None, auto_threshold=None, target_precision=None, pr_out_dir=None, save_thresholds_path=None, abstain_unknown=False, unknown_id=None, tta=False):
    # Initialize data loader
    data_loader = FITSDataLoader(target_size)
    
    # Load and prepare dataset
    print("Loading and preparing dataset...")
    images, features, confidences, labels = data_loader.prepare_dataset(data_dir)
    
    # Convert labels to one-hot encoding
    num_classes = len(data_loader.classes)
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    labels_int = np.asarray(labels, dtype=np.int32)
    
    # Load model
    print("Loading model...")
    model = MultiModalClassifier(input_shape=(*target_size, 1), num_classes=num_classes)
    model.load(model_path)
    
    # Apply same feature standardization as in training if available
    import os
    scaler_path = os.path.splitext(model_path)[0] + "_feat_scaler.npz"
    if os.path.exists(scaler_path):
        scaler = np.load(scaler_path)
        feat_mean = scaler.get('mean')
        feat_std = scaler.get('std')
        if feat_mean is not None and feat_std is not None:
            feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
            features = (features - feat_mean) / feat_std

    # Make predictions (optionally with Test-Time Augmentation by flip averaging)
    print("Making predictions...")
    def _predict(images_np):
        return np.asarray(model.predict(images_np, features, confidences), dtype=np.float32)
    if not tta:
        proba = _predict(images)
    else:
        # Horizontal and vertical flips; average probabilities
        proba_list = []
        proba_list.append(_predict(images))
        proba_list.append(_predict(images[:, ::-1, :, :]))  # H flip
        proba_list.append(_predict(images[:, :, ::-1, :]))  # V flip
        proba = np.mean(np.stack(proba_list, axis=0), axis=0)

    # Handle thresholds argument that may be a string (comma list or JSON path) or a dict
    class_names = list(data_loader.classes.keys())
    if isinstance(thresholds, str):
        thresholds = parse_thresholds_arg(thresholds, num_classes)
    if isinstance(thresholds, dict):
        thr_arr = np.zeros(num_classes, dtype=np.float32)
        for name, idx in data_loader.classes.items():
            thr_arr[idx] = float(thresholds.get(name, 0.5))
        thresholds = thr_arr

    # Auto thresholds via PR curves
    suggested = None
    if auto_threshold in {"max_f1", "target_precision"} or pr_out_dir is not None:
        suggested = compute_pr_curves_per_class(labels_int, proba, class_names, out_dir=(pr_out_dir or '.'))
        if auto_threshold == "max_f1" and suggested:
            thresholds = np.array([suggested[name]['best_f1_threshold'] for name in class_names], dtype=np.float32)
        elif auto_threshold == "target_precision" and suggested and target_precision is not None:
            # recompute thresholds meeting target precision per class
            thrs = []
            for c, name in enumerate(class_names):
                y_true_bin = (labels_int == c).astype(np.int32)
                precision, recall, thr = precision_recall_curve(y_true_bin, proba[:, c])
                # find first threshold where precision >= target, highest recall among those
                candidates = np.where(precision[:-1] >= float(target_precision))[0]
                if candidates.size > 0:
                    best_idx = int(candidates[np.argmax(recall[:-1][candidates])])
                    thrs.append(float(thr[best_idx]))
                else:
                    # fallback to max F1
                    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-8)
                    best_idx = int(np.argmax(f1)) if f1.size > 0 else 0
                    thrs.append(float(thr[best_idx]) if thr.size > 0 else 0.5)
            thresholds = np.array(thrs, dtype=np.float32)

    # Save suggested thresholds if requested
    if save_thresholds_path and suggested is not None:
        with open(save_thresholds_path, 'w') as f:
            json.dump(suggested, f, indent=2)
        print(f"Saved threshold suggestions to: {save_thresholds_path}")
    
    # Calculate accuracy with/without thresholds
    y_pred_argmax = np.argmax(proba, axis=1)
    y_true_argmax = np.argmax(labels_onehot, axis=1)
    base_accuracy = np.mean(y_pred_argmax == y_true_argmax)
    print(f"\nOverall accuracy (argmax): {base_accuracy:.4f}")
    if thresholds is not None:
        y_pred_thresh = apply_thresholds_to_predictions(proba, thresholds, strategy="gate", abstain=abstain_unknown, unknown_id=unknown_id)
        # Coverage & accuracy@coverage when abstaining
        if abstain_unknown:
            unk_id = (num_classes if unknown_id is None else int(unknown_id))
            covered_mask = (y_pred_thresh != unk_id)
            coverage = float(np.mean(covered_mask))
            acc_at_cov = float(np.mean(y_pred_thresh[covered_mask] == y_true_argmax[covered_mask])) if np.any(covered_mask) else 0.0
            # Strict: count unknown as incorrect
            strict_accuracy = float(np.mean(y_pred_thresh == y_true_argmax))
            print(f"Overall coverage: {coverage:.4f}")
            print(f"Accuracy@coverage: {acc_at_cov:.4f}")
            print(f"Overall accuracy (unknown counted incorrect): {strict_accuracy:.4f}")
        else:
            thr_accuracy = np.mean(y_pred_thresh == y_true_argmax)
            print(f"Overall accuracy (thresholded): {thr_accuracy:.4f}")
        y_pred_for_report = y_pred_thresh
    else:
        y_pred_for_report = y_pred_argmax
    
    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    for class_name, class_idx in data_loader.classes.items():
        class_mask = (y_true_argmax == class_idx)
        if np.any(class_mask):
            class_accuracy = np.mean(y_pred_for_report[class_mask] == y_true_argmax[class_mask])
            print(f"{class_name}: {class_accuracy:.4f}")
        else:
            print(f"{class_name}: No samples available")
    if thresholds is not None and abstain_unknown:
        print("unknown (abstained): reported separately via coverage/accuracy@coverage")

    # Calculate metrics
    print("\nClassification Report:")
    # If abstaining, exclude unknown from the report to avoid shape mismatch unless we extend class list
    if thresholds is not None and abstain_unknown:
        unk_id = (num_classes if unknown_id is None else int(unknown_id))
        mask = (y_pred_for_report != unk_id)
        print(classification_report(y_true_argmax[mask], y_pred_for_report[mask],
                              target_names=list(data_loader.classes.keys())))
    else:
        print(classification_report(y_true_argmax, y_pred_for_report,
                              target_names=list(data_loader.classes.keys())))
    
    # Plot confusion matrix with detailed analysis
    # Confusion matrix: if abstaining, add an "unknown" column for visualization
    if thresholds is not None and abstain_unknown:
        unk_id = (num_classes if unknown_id is None else int(unknown_id))
        # Map unknown to a new label index for plotting convenience
        y_pred_plot = np.copy(y_pred_for_report)
        y_pred_plot[y_pred_plot == unk_id] = num_classes  # last index
        plot_classes = list(data_loader.classes.keys()) + ["unknown"]
        plot_confusion_matrix(y_true_argmax, y_pred_plot, plot_classes)
    else:
        plot_confusion_matrix(y_true_argmax, y_pred_for_report, 
                         list(data_loader.classes.keys()))
    
    # Analyze feature importance
    feature_names = [f"feature_{i}" for i in range(features.shape[1])]
    analyze_feature_importance(model, features, y_true_argmax, feature_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multimodal classifier with optional per-class thresholds and PR analysis")
    parser.add_argument("--model_path", default="multimodal_classifier.keras")
    parser.add_argument("--data_dir", default="../image_classification_test_sample")
    parser.add_argument("--target_w", type=int, default=350)
    parser.add_argument("--target_h", type=int, default=350)
    parser.add_argument("--thresholds", type=str, default=None, help="Scalar, comma list, or JSON path (list or {class: thr})")
    parser.add_argument("--auto_threshold", choices=[None, "max_f1", "target_precision"], default=None)
    parser.add_argument("--target_precision", type=float, default=None, help="Target precision for auto_threshold=target_precision")
    parser.add_argument("--pr_out_dir", type=str, default=None, help="Directory to save per-class PR curves")
    parser.add_argument("--save_thresholds", type=str, default=None, help="Path to save suggested thresholds JSON")
    parser.add_argument("--abstain_unknown", action="store_true", help="Return unknown when no class passes threshold; report coverage and accuracy@coverage")
    parser.add_argument("--unknown_id", type=int, default=None, help="Custom integer id for unknown (default: num_classes)")
    parser.add_argument("--tta", action="store_true", help="Enable simple TTA (flip averaging) during evaluation")
    args = parser.parse_args()

    # Build target size
    tsize = (args.target_h, args.target_w)

    # Parse thresholds (may be np.ndarray, dict, or None)
    thresholds = args.thresholds if args.thresholds is not None else None

    evaluate_model(
        args.model_path,
        args.data_dir,
        target_size=tsize,
        thresholds=thresholds,
        auto_threshold=args.auto_threshold,
        target_precision=args.target_precision,
        pr_out_dir=args.pr_out_dir,
        save_thresholds_path=args.save_thresholds,
        abstain_unknown=args.abstain_unknown,
        unknown_id=args.unknown_id,
        tta=args.tta,
    )