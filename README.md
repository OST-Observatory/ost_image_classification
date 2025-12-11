# OST Image Classification - Project Overview and Usage

This repository implements a multimodal image classifier for astronomical data. The model ingests an image (FITS/TIFF) and auxiliary header/statistical features, and outputs a class among several astrophotography categories.

## Contents
- Current architecture and files
- Installation
- Training workflow
- Evaluation workflow (with per-class thresholds and PR curves)
- Class imbalance handling (class weights and focal loss)
- Recommended settings and thresholds
- Tips for memory-constrained environments

## Current Architecture and Files
- `data_loader.py`: Loads FITS/TIFF files, extracts basic image statistics and header-derived features, performs normalization and resizing to the configured target size.
- `model.py`: Defines a multimodal Keras model with:
  - CNN image branch with GlobalAveragePooling2D (GAP)
  - Dense feature branch for metadata and confidence inputs
  - Concatenation and classification head with softmax output
- `main.py`: Training entry point. Includes:
  - Stratified train/validation split
  - Feature standardization fit on train and applied to validation/test
  - Choice of loss strategy: none, class weights, or focal loss
  - Optional cosine LR schedule
  - Class-weight smoothing and clipping options
- `evaluate_model.py`: Evaluation entry point. Includes:
  - Batch evaluation with saved feature scaler
  - Optional per-class decision thresholds
  - Automatic threshold suggestions via per-class Precision-Recall (PR) curves (max-F1 or target precision)
  - Confusion matrix and detailed per-class metrics
  - Feature-importance probe (gradient-based on the feature branch)

## Installation
Use Python 3.9+ recommended. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Workflow
Recommended training (class weights with smoothing/clipping, warmup+cosine LR, tf.data cache/prefetch, on-the-fly augmentations, MixUp, balanced sampling):
```bash
python main.py \
  --data_dir ../image_classification_training_sample \
  --target_w 448 --target_h 448 \
  --epochs 30 --batch_size 24 \
  --loss_strategy class_weights --cw_beta 0.3 --cw_cap 4.0 \
  --label_smoothing 0.08 \
  --initial_lr 1e-3 --batch_ref 32 \
  --lr_schedule cosine --warmup_epochs 5 \
  --augment --aug_strength 0.3 \
  --aug_conservative 'flat_dados,darks,flats,spectrum_dados,wavelength_calibration_dados,wavelength_calibration_baches' \
  --randaug_n 2 --randaug_m 5 \
  --mixup_alpha 0.1 \
  --balanced_sampling --balance_cap 4.0 \
  --model_out multimodal_classifier_mixup_warmup_randaug.keras
```
Notes:
- `--cw_beta` applies power smoothing to class weights (w := w^beta). Use 0.2–0.5 for moderate smoothing.
- `--cw_cap` clips weights to cap × median(weight) to avoid excessive minority up-weighting.
- Warmup + cosine LR: `--lr_schedule cosine --warmup_epochs 5` typically stabilizes early training.
- On-the-fly augmentation can be tuned via `--aug_strength`, and conservative classes are passed via `--aug_conservative`.
- RandAugment-light adds small extra ops for non-conservative classes: `--randaug_n`, `--randaug_m`.
- MixUp can improve robustness at class boundaries: `--mixup_alpha 0.1–0.2`.
- Balanced sampling increases minority coverage without double-weighting (per-sample weights are disabled when balanced sampling is on).
- Set `--loss_strategy focal_loss` to enable focal loss with automatic class-specific alpha if desired; in our tests class weights performed better overall.

## Evaluation Workflow
Basic evaluation:
```bash
python evaluate_model.py \
  --model_path multimodal_classifier_mixup_warmup_randaug.keras \
  --data_dir ../image_classification_test_sample \
  --target_w 448 --target_h 448
```

With auto-thresholding (per-class max-F1), TTA and PR curves:
```bash
python evaluate_model.py \
  --model_path multimodal_classifier_mixup_warmup_randaug.keras \
  --data_dir ../image_classification_test_sample \
  --target_w 448 --target_h 448 \
  --tta --fit_temperature \
  --auto_threshold max_f1 \
  --pr_out_dir pr_curves \
  --save_thresholds thresholds_suggested.json
```

Using explicit per-class thresholds (comma-separated list; order follows `data_loader.classes`):
```bash
python evaluate_model.py \
  --model_path multimodal_classifier_mixup_warmup_randaug.keras \
  --data_dir ../image_classification_test_sample \
  --target_w 448 --target_h 448 \
  --thresholds 0.904,0.951,0.759,0.04,0.297,0.938,0.95,0.038,0.77,0.775,0.973 \
  --pr_out_dir pr_curves_final2 \
  --save_thresholds thresholds_final2.json
```

You can also pass `--thresholds` as a JSON path with a list of thresholds or a dict mapping class names to thresholds.
We ship ready-to-use profiles:
- Production precision-focused: `thresholds_prod_precision.json`
- Production recall-focused: `thresholds_prod_recall.json`

Temperature scaling:
- Fit temperature on-the-fly: `--fit_temperature`
- Or set a fixed temperature (e.g., from previous fit): `--temperature 0.7`

## Thresholds: What they are and how to use them
- What they do: Per-class thresholds do not change trained weights; they only change the decision rule at inference/evaluation. They shift the precision–recall trade-off per class and can increase reported accuracy and calibration without retraining.
- Where to tune: Always select thresholds on a validation split, then apply to test/production. Avoid tuning on the test set.
- How to choose:
  - Auto max-F1: `--auto_threshold max_f1` suggests the threshold with highest F1 per class.
  - Target precision: `--auto_threshold target_precision --target_precision 0.95` picks the highest recall threshold that meets the precision target.
  - Inspect PR curves: Use `--pr_out_dir` to save per-class PR curves for manual review.
- How to apply:
  - Save suggestions via `--save_thresholds` and reuse via `--thresholds` (scalar, comma list, or JSON of list/dict).
  - In production, load the validated thresholds and apply them before converting probabilities to class labels.
- Optional “abstain”: If no class exceeds its threshold, treat the sample as “unknown” instead of forcing argmax. This reduces false positives in sensitive pipelines.
- Maintenance: Recompute PR curves and refresh thresholds when data distribution drifts (e.g., new devices/nights). Probability calibration (e.g., temperature scaling on validation) can further stabilize thresholding.

### Abstain/Unknown usage
- Enable abstaining when no class passes its threshold:
```bash
python evaluate_model.py \
  --model_path multimodal_classifier_class_weights.keras \
  --data_dir ../image_classification_test_sample \
  --thresholds 0.904,0.951,0.759,0.04,0.297,0.938,0.95,0.038,0.77,0.775,0.973 \
  --abstain_unknown \
  --pr_out_dir pr_curves_with_unknown \
  --save_thresholds thresholds_with_unknown.json
```
- Unknown ID: Defaults to `num_classes` (appends an extra column in the confusion matrix). You can override:
```bash
python evaluate_model.py ... --abstain_unknown --unknown_id 99
```
- Reporting: The evaluator prints coverage (fraction not abstained) and accuracy@coverage. When abstaining, the classification report excludes the unknowns; the confusion matrix includes an extra "unknown" column.

## Class Imbalance Handling
- Class Weights (recommended): Enabled via `--loss_strategy class_weights`. Combine with label smoothing (0.05–0.10) for robust calibration.
- Focal Loss: `--loss_strategy focal_loss`. Works best with tuned gamma (1.0–1.5) and smoothed alpha; in our tests, class weights performed better overall.

## Recommended Settings and Thresholds
From recent experiments:
- Class weights with smoothing/clipping: `--cw_beta 0.3 --cw_cap 4.0`
- Label smoothing: `--label_smoothing 0.08`
- Per-class thresholds: prefer JSON profiles (precision vs recall) shipped in repo; re-fit via PR curves when data changes.
- Observed performance (thresholded): overall accuracy ≈ 0.96 with strong precision across classes and high recall for major classes. Key trade-offs:
  - Slight recall reduction for `darks` and `spectrum_baches` compared to argmax
  - Substantially improved calibration for `deep_sky` and `wavelength_calibration_*`

## Tips for Memory-Constrained Environments (18 GB RAM)
- Prefer streaming pipelines (`tf.data`) over building full in-memory arrays.
- Start with 224×224 input resolution, batch size ~8.
- Use on-the-fly augmentation within the dataset pipeline.
- Keep inputs/features as `float16` where feasible; model weights as `float32` on CPU.
- Replace Flatten with GAP (already implemented) and moderate filter sizes.
- Always evaluate in batches; avoid full-dataset inference in memory.

## Known Next Steps
- Optional: class-specific threshold search targeting precision per class (automate per-class targets)
- Optional: lighter CNN backbone (depthwise/separable convolutions) or small transfer backbone if GPU becomes available
- Optional: K-fold validation for more stable operating points (compute-intensive)

## Production usage
- Model and scaler
  - Use the saved Keras model (`*.keras`) and the matching feature scaler (`*_feat_scaler.npz`).
  - Always resize to the trained target size (e.g., 448×448) and normalize images consistently with the training pipeline.
- Thresholds and temperature
  - Fix temperature to 0.7 (as validated) or refit periodically on fresh validation data.
  - Choose a threshold profile:
    - Precision-focused: `thresholds_prod_precision.json`
    - Recall-focused: `thresholds_prod_recall.json`
- Abstain/Unknown
  - Enable abstain in sensitive pipelines: return “unknown” if no class exceeds its threshold. This reduces false positives at a small coverage cost.
- TTA
  - Keep TTA off by default for latency-sensitive paths; enable `--tta` if you can afford the extra compute for a small accuracy boost.
- Monitoring and refresh
  - Log: coverage, accuracy@coverage (if abstaining), per-class precision/recall, and NLL.
  - Track input data drift; when drift is detected or on a schedule (e.g., monthly), refit temperature and regenerate threshold suggestions via PR curves.
  - Version artifacts: model, scaler, thresholds, temperature.

Example batch inference (production-like):
```bash
python evaluate_model.py \
  --model_path multimodal_classifier_mixup_warmup_randaug.keras \
  --data_dir ../image_classification_test_sample \
  --target_w 448 --target_h 448 \
  --temperature 0.7 \
  --thresholds thresholds_prod_precision.json \
  --abstain_unknown
```

### CLI tool for filesystem classification
- Classify all supported images below a directory and save a JSON report:
```bash
python inference_runner.py filesystem \
  --input_dir /data/images \
  --output_json results.json \
  --model_path multimodal_classifier_mixup_warmup_randaug_2.keras \
  --thresholds thresholds_prod_precision.json \
  --temperature 0.7 \
  --tta \
  --abstain_unknown \
  --defect_log defects.json
```
JSON schema:
- `meta`: timestamp, model_path, thresholds_path, temperature, tta, abstain_unknown, unknown_id, batch_size, coverage, num_files, classes, elapsed_sec
- `results[]`: path, class, class_id, score, abstained, (optional `_probs` if `--include_probs`)
- `defects[]`: per-file issues detected during loading/preprocessing: `{ path, issue, message }`
  - Loader hardening:
    - FITS opened with `memmap=True, ignore_missing_end=True` (truncated files warned und geloggt)
    - NaN/Inf werden zu 0.0 gesetzt; degenerierte Bilder (nahezu konstant) werden markiert

### Django integration (examples)
- Import `ClassifierService` from `inference_runner.py` and call `predict_paths()` with your list of file paths.
- Management command outline:
```python
# yourapp/management/commands/classify_new_files.py
from django.core.management.base import BaseCommand
from inference_runner import ClassifierService

class Command(BaseCommand):
    def handle(self, *args, **options):
        svc = ClassifierService(
            model_path=\"multimodal_classifier_mixup_warmup_randaug_2.keras\",
            thresholds_path=\"thresholds_prod_precision.json\",
            temperature=0.7,
            tta=True,
            abstain_unknown=True,
        )
        paths = [...]  # collect from your DB
        payload = svc.predict_paths(paths, batch_size=32)
        # persist payload[\"results\"] to your DB
```
- Celery task outline:
```python
# yourapp/tasks.py
from celery import shared_task
from inference_runner import ClassifierService

_svc = None
def get_svc():
    global _svc
    if _svc is None:
        _svc = ClassifierService(
            model_path=\"multimodal_classifier_mixup_warmup_randaug_2.keras\",
            thresholds_path=\"thresholds_prod_precision.json\",
            temperature=0.7, tta=True, abstain_unknown=True
        )
    return _svc

@shared_task
def classify_paths(paths):
    svc = get_svc()
    return svc.predict_paths(paths, batch_size=32)
```

---
If you make changes to class mappings in `data_loader.py`, ensure that the threshold list ordering and saved scaler (`*_feat_scaler.npz`) remain consistent across training and evaluation.
