# OST Image Classification – TODO

Created: 2026-10-13

Context: too many archive files needed a manual exposure type. Archive-side tasks (combination rule, review queue, header mapping, `exposure_type_report`) are tracked in `ost_data_archive/TODO.md`.

**Production findings (2026-10-13, 119,535 manual labels):**
- Old SBIG data (2008–2019, no `IMAGETYP`): when the model commits to a class it is right in 93 % → it is the only source of truth there
- New QHY/ZWO data (2021+): the header is right in ~96 %, the model is wrong or abstains on ~21 % of files with a correct header → since the new archive rule (`header_first`) the model only fills header gaps there and flags disagreements
- Main confusion: light ↔ dark in both directions (`LI → DA` 7,831, `DA → LI` 7,691), then `LI → FL` 1,607
- Confidence is poorly calibrated: ≥ 0.99 only 81 % correct, 0.90–0.97 57 % (measured on the manually labelled files)
- Abstaining: 870 real bias frames rejected (bias threshold 0.990 in `thresholds_prod_precision.json`)
- Spectrograph (2,790 manual corrections): `einsteinturm → deep_sky` 1,268 (most likely TIFF files – `is_tiff` + only 23 Einsteinturm training files), `spectrum_dados → deep_sky` 339 (QHY600M imaging camera), `spectrum_baches → darks` 330, `darks → wavelength_calibration_dados` 215, `darks → spectrum_dados` 104 (new DADOS setup with QHY268M in 2026)
- The reported test accuracy (≈0.96) does not reflect production: `cnn_training_data.json` only contains header/ML agreement cases ≥ 2021

---

## Open

### Pipeline v2 (implemented 2026-10-16, see README "Pipeline v2")

Implemented as separate modules; the legacy pipeline stays unchanged for comparison with the production model.
Covers the former step 1 (hard test set, per instrument/year, calibration on val) and step 2 items 0-7 and 10:
run-based splits from the archive export v2, hard cases, star count, band profile, saturation, percentile +
asinh stretch, IMAGETYP/OBJECT/instrument/metadata features, no confidence branch and no `is_tiff`/`file_size`
shortcuts, log-scaled statistics, LightGBM baseline, explicit feature schema checked at inference.
Measured peak memory (synthetic, 448 px): cache build 0.3 GB, CNN training batch 16 3.3 GB.

Next (a12, see README for the commands):
1. [ ] Export v2 on the archive server, build the cache on a12
2. [ ] Train `train_tabular.py` and `train.py`; run `evaluate_v2.py` against the legacy model
3. [x] Decide: CNN, LightGBM or hybrid. Evaluation 2026-11-20 (v2b, full test split, 2730 files, 73 runs): ensemble CNN + LightGBM 1:1 best (run-acc 0.948, LI->DA 4, DA->LI 7, ECE 0.022; CNN 0.920, LightGBM 0.934). The models fail on different runs (CNN: Skyris H-alpha sun, old SBIG darks; LightGBM: moon 2018-07-06). Ensemble files are written by `evaluate_v2.py --save_calibration` and loaded by `ClassifierService`
4. [ ] Underexposed arc frames (run 2024-12-12, DADOS Ar-Ne 1 s, QHY 268M, IMAGETYP "Light Frame"): LightGBM and the ensemble call 97-100 of 113 darks, the CNN 17. Check after the next export (more such runs in training?); candidate feature: line contrast of the band profile relative to the noise
4a. [x] Bugs found by the archive comparison (`classify_exposure_types --compare`, 2026-11-20): (1) `_block_mean` failed for frames whose height is not a multiple of the reduction factor (most large-sensor frames, e.g. QHY 600M/268M): 51 % of a random archive sample unreadable, and the same files are missing from cache, training and evaluation (no QHY 600M in the evaluation at all). (2) Threshold gating took the most probable class among all classes passing their threshold, so classes with tiny thresholds (bias 0.001) won with tiny probabilities (lights -> bias/arc/flat); now top class or abstain for v2 models, thresholds fitted on the files where the class is top. Legacy model keeps its rule
4b. [ ] New export, new cache directory (existing shards do not retry failed files), retrain, re-evaluate; check `errors.jsonl` of the old cache for the number of lost files
5. [ ] Bias has no files in the test split of the v2b export (2026-04-10 relabelled from darks to bias); check `missing_classes` of the next export
6. [ ] Optional: input size 256 vs. 448 and `--separable` for faster CPU training; JPG/CR2 support only if needed; pretrained backbone if a GPU becomes available

### Smaller bugs

- [x] `predict_paths` returns only `score` (top probability) when abstaining, not the top class → return `top_class` too, so the archive can show a suggestion in the review queue
- [x] (v2) Binning is only parsed from `BINNING` in `NxM` format (`data_loader.py` ~L264); most cameras write `XBINNING`/`YBINNING`
- [x] (v2, branch removed) `calculate_header_confidence`: `expected_mean = exp_time * 100` is not a meaningful model for any camera – drop with step 2, item 3
- [x] `evaluate_model.py` ~L51 prints `Misclassified as {class_name}` instead of naming the source class (`other_class`), which makes the "Most confused with" lists misleading
- [ ] README install URLs still point to `your-git-server` placeholders (archive pins `github.com/OST-Observatory/ost_image_classification`)
- [x] Legacy fixes: `--save_thresholds` writes the flat `{class: threshold}` format of the chosen method; evaluation default size 448; `main.py` shuffles after `cache()` and applies MixUp only to training data; stale `classify_images.py` removed
- [ ] Legacy, intentionally unchanged (the production model depends on them): FITS `width`/`height` always 0, all features 0 when the header cannot be read, `hist_mean` scaled by ~1e6, silent pad/truncate of feature vectors in the legacy inference path
