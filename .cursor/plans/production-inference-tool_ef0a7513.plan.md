---
name: production-inference-tool
overview: "Add a production-ready inference tool with two modes: a CLI to classify directories recursively and a library API that a Django management command/Celery task can call. Use the validated operating point (temperature=0.7, thresholds_prod_precision.json, abstain enabled)."
todos:
  - id: add-script
    content: Create inference_runner.py with ClassifierService and CLI (filesystem mode)
    status: completed
  - id: lib-api
    content: Expose predict_paths API using current preprocessing, temperature and thresholds
    status: completed
  - id: json-schema
    content: Define and document output JSON schema and meta block
    status: completed
  - id: django-mgmt
    content: Provide Django management command snippet to call the library API
    status: completed
  - id: celery-task
    content: Provide Celery task snippet for classifying new files
    status: completed
  - id: docs-update
    content: Update README with CLI usage and Django integration instructions
    status: completed
---

# Production Inference Tool

## What we’ll build

- A new script `inference_runner.py` providing:
- CLI “filesystem mode”: classify all supported images in a directory (recursive), write a JSON report.
- Library API: load model/scaler/thresholds once and classify given file paths in batches (to be used by a Django management command/Celery task).
- Sample Django integration snippets (management command and Celery task) that call the library API.

## Key details

- Defaults: temperature=0.7, thresholds from `thresholds_prod_precision.json`, `--abstain_unknown` on.
- Supported files: FITS (`.fit/.fits`) and TIFF (`.tif/.tiff`), same preprocessing as `data_loader.py` (resize to 448×448, normalization, feature extraction, scaler application).
- TTA optional (`--tta`); batch inference with configurable `--batch_size`.
- Output JSON: one record per file with fields: `path`, `class`, `score`, `abstained`, `probs` (per-class), `timestamp`, plus `meta` section with config (model path, thresholds file, temperature, tta, batch size, coverage if abstain used).

## Files to add

- [inference_runner.py](./inference_runner.py):
- `ClassifierService` class: loads model + scaler + thresholds; `predict_paths(paths, batch_size=32, tta=True, temperature=0.7, abstain=True)`; returns structured results.
- CLI (argparse):
- `filesystem` subcommand: `--input_dir`, `--output_json`, `--batch_size`, `--model_path`, `--thresholds`, `--temperature`, `--tta`, `--abstain_unknown`.
- `dry-run`/`--include-probs` flags.
- Docs: extend README “Production usage” with commands and Django integration instructions.

## Django integration (snippets provided)

- Management command example (`yourapp/management/commands/classify_new_files.py`):
- Collect new/changed files from your Archive models, call `ClassifierService.predict_paths`, persist results.
- Celery task example (`yourapp/tasks.py`):
- `@shared_task` that accepts file IDs/paths, batches them, calls the service, writes results back.

## Optimizations & safety

- Preload model and scaler once per process.
- Chunked file scanning; configurable worker count optional (start with single-process because TensorFlow is multi-threaded).
- Robust error handling per-file; include `error` field in JSON for failed files.
- Atomic JSON write (write temp then rename);
- Deterministic ordering and idempotency: compute and include file size + mtime hash in results.

## CLI examples

- Filesystem mode:
- `python inference_runner.py filesystem --input_dir /data/images --output_json results.json --model_path multimodal_classifier_mixup_warmup_randaug_2.keras --thresholds thresholds_prod_precision.json --temperature 0.7 --tta --abstain_unknown`

## Sample Django snippets

- Management command: import `ClassifierService` from this repo (installable path or vendored module), pass list of file paths; map results back to your DB models.
- Celery task: same service, called asynchronously for new files; ensure model is loaded once per worker.