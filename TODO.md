# OST Image Classification – TODO

Created: 2026-10-13

Context: too many archive files end up with exposure type `UK` and need manual correction. Archive-side tasks (header mapping, combination rule, review queue, `exposure_type_report`) are tracked in `ost_data_archive/TODO.md`. Prioritise using the production output of `python manage.py exposure_type_report`.

---

## Open

### Step 1 – evaluation that reflects production

1. **Hard evaluation set:** evaluate on files with manual labels where header and ML disagreed (exported by the archive). `cnn_training_data.json` only contains header/ML agreement cases, so the reported ≈0.96 accuracy is optimistic.
2. **Report per instrument / year**, not only per class – drift between cameras is a likely failure mode.
3. **Main confusion today:** `darks` → `deep_sky` (93) and `darks` → unknown (41) in `evaluation_results.txt`; check whether better features (step 2) fix this before touching the architecture.

### Step 2 – model improvements

1. **Fix the dead confidence branch:** `calculate_header_confidence` (`data_loader.py` ~L159–204) reads `object`, `obsmode` and `filter`, but `extract_header_features` never sets them → 3 of 4 confidence inputs are constant (0 / 0.5). Either feed real header values or remove the branch.
2. **Use the informative header keywords as features:** normalised `IMAGETYP` (one-hot), `OBJECT` keyword tokens (`flat`, `dark`, `bias`, `arc`, …), `FILTER` present, instrument/camera, `log(exptime)`; optionally metadata passed in by the archive (`plate_solved`, run `spectroscopy`). Extend `ClassifierService.predict_paths` to accept per-file metadata.
3. **Robust image scaling:** replace global min–max (a single hot pixel/cosmic ray squashes the image) with percentile clipping (e.g. 0.5/99.5) + asinh or zscale; consider a second channel (log stretch).
4. **Log-transform heavy-tailed statistics** (`mean`, `std`, `max`, `mad`, `file_size`) before standardisation.
5. **Tabular baseline:** gradient boosting (LightGBM/XGBoost) on header + statistics features for BI/DA/FL/LI; compare against the CNN on the hard set. Possible hybrid: tabular model for the exposure type, CNN for spectrograph (DADOS/BACHES/Einsteinturm) and wavelength calibration.
6. **Pretrained backbone** (e.g. small EfficientNet/ConvNeXt) instead of the 3-layer CNN, if a GPU is available.
7. **Feedback loop:** include manually corrected files in the next training export; retrain + refit thresholds/temperature on each refresh.
8. **Remove hard-coded input sizes** in `model.py` (`num_features = 22`, `num_confidences = 4`) – derive from the data/scaler so new features don't silently break loading.

### Smaller bugs

- [ ] `predict_paths` returns only `score` (top probability) when abstaining, not the top class → return `top_class` too, so the archive can show a suggestion in the review queue
- [ ] Binning is only parsed from `BINNING` in `NxM` format (`data_loader.py` ~L264); most cameras write `XBINNING`/`YBINNING`
- [ ] `calculate_header_confidence`: `expected_mean = exp_time * 100` is not a meaningful model for any camera – drop with item 2.1
- [ ] `evaluate_model.py` ~L51 prints `Misclassified as {class_name}` instead of naming the source class (`other_class`), which makes the "Most confused with" lists misleading
- [ ] README install URLs still point to `your-git-server` placeholders (archive pins `github.com/OST-Observatory/ost_image_classification`)
