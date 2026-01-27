import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from data_loader import FITSDataLoader
from model import MultiModalClassifier


SUPPORTED_EXTS = {'.fit', '.fits', '.tif', '.tiff', '.FIT', '.FITS', '.TIF', '.TIFF'}


def softmax_with_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-8
    t = max(temperature, eps)
    # Temperature scaling in probability space (p ** (1/T), renormalize)
    q = np.power(np.clip(probs, eps, 1.0), 1.0 / t)
    q /= np.sum(q, axis=1, keepdims=True)
    return q


def apply_thresholds(proba: np.ndarray,
                     thresholds: np.ndarray,
                     abstain: bool,
                     unknown_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (pred_labels, abstained_mask)
    """
    pass_mask = proba >= thresholds.reshape((1, -1))
    top_idx = np.argmax(proba, axis=1)
    pred = []
    abstained = []
    for i in range(proba.shape[0]):
        passing = np.where(pass_mask[i])[0]
        if passing.size > 0:
            j = passing[np.argmax(proba[i, passing])]
            pred.append(int(j))
            abstained.append(False)
        else:
            if abstain:
                pred.append(int(unknown_id))
                abstained.append(True)
            else:
                pred.append(int(top_idx[i]))
                abstained.append(False)
    return np.array(pred, dtype=np.int32), np.array(abstained, dtype=bool)


class ClassifierService:
    def __init__(self,
                 model_path: str,
                 thresholds_path: Optional[str] = None,
                 temperature: float = 0.7,
                 tta: bool = False,
                 abstain_unknown: bool = True,
                 unknown_id: Optional[int] = None,
                 target_size: Tuple[int, int] = (448, 448)) -> None:
        self.model_path = model_path
        self.thresholds_path = thresholds_path
        self.temperature = float(temperature)
        self.tta = bool(tta)
        self.abstain_unknown = bool(abstain_unknown)
        self.unknown_id_custom = unknown_id
        self.target_size = target_size

        # Loader (reuses training preprocessing)
        self.loader = FITSDataLoader(target_size=self.target_size)
        self.classes = self.loader.classes  # name -> idx
        self.class_names = list(self.classes.keys())
        self.num_classes = len(self.classes)
        self.unknown_id = self.num_classes if self.unknown_id_custom is None else int(self.unknown_id_custom)

        # Model
        self.model = MultiModalClassifier(input_shape=(*self.target_size, 1), num_classes=self.num_classes)
        self.model.load(self.model_path)

        # Feature scaler
        base, _ = os.path.splitext(self.model_path)
        scaler_path = base + "_feat_scaler.npz"
        self.feat_mean = None
        self.feat_std = None
        if os.path.exists(scaler_path):
            scaler = np.load(scaler_path)
            self.feat_mean = scaler.get('mean')
            self.feat_std = scaler.get('std')
            if self.feat_mean is not None and self.feat_std is not None:
                self.feat_std = np.where(self.feat_std < 1e-6, 1.0, self.feat_std)

        # Thresholds
        self.thresholds = np.full(self.num_classes, 0.5, dtype=np.float32)
        if thresholds_path:
            with open(thresholds_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                thr = np.zeros(self.num_classes, dtype=np.float32)
                for name, idx in self.classes.items():
                    thr[idx] = float(data.get(name, 0.5))
                self.thresholds = thr
            elif isinstance(data, list):
                arr = np.array([float(x) for x in data], dtype=np.float32)
                if arr.size == self.num_classes:
                    self.thresholds = arr

    def _predict_batch(self, images_b: np.ndarray, features_b: np.ndarray, conf_b: np.ndarray) -> np.ndarray:
        if not self.tta:
            return np.asarray(self.model.predict(images_b, features_b, conf_b), dtype=np.float32)
        proba_list = []
        proba_list.append(np.asarray(self.model.predict(images_b, features_b, conf_b), dtype=np.float32))
        proba_list.append(np.asarray(self.model.predict(images_b[:, ::-1, :, :], features_b, conf_b), dtype=np.float32))
        proba_list.append(np.asarray(self.model.predict(images_b[:, :, ::-1, :], features_b, conf_b), dtype=np.float32))
        return np.mean(np.stack(proba_list, axis=0), axis=0)

    def _prepare_items(self, paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        images = []
        feats = []
        confs = []
        ok_paths = []
        errors = []
        defects: List[Dict[str, Any]] = []   # only truly unreadable files (exceptions)
        warnings_list: List[Dict[str, Any]] = []  # non-fatal issues (read warnings, degenerate image)
        target_feat_len: Optional[int] = None
        target_conf_len: Optional[int] = None
        for p in paths:
            try:
                img = self.loader.load_image(p)
                # log any loader warnings (e.g., truncated FITS)
                for msg in getattr(self.loader, "last_read_warnings", []) or []:
                    warnings_list.append({"path": p, "issue": "read_warning", "message": msg})
                proc_img, img_stats = self.loader.preprocess_image(img)
                header_features, header_conf = self.loader.extract_header_features(p, img_stats)
                combined = {**header_features, **img_stats}
                numeric_features = []
                for v in combined.values():
                    if isinstance(v, (int, float)):
                        numeric_features.append(float(v))
                    elif isinstance(v, str):
                        numeric_features.append(float(len(v)))
                    else:
                        numeric_features.append(0.0)
                numeric_confs = []
                for v in header_conf.values():
                    if isinstance(v, (int, float)):
                        numeric_confs.append(float(v))
                    else:
                        numeric_confs.append(0.0)
                feat_arr = np.array(numeric_features, dtype=np.float32)
                conf_arr = np.array(numeric_confs, dtype=np.float32)
                # Normalize vector lengths across files (pad/truncate)
                if target_feat_len is None:
                    target_feat_len = int(feat_arr.shape[0])
                if target_conf_len is None:
                    target_conf_len = int(conf_arr.shape[0])
                if feat_arr.shape[0] != target_feat_len:
                    if feat_arr.shape[0] > target_feat_len:
                        feat_arr = feat_arr[:target_feat_len]
                    else:
                        feat_arr = np.pad(feat_arr, (0, target_feat_len - feat_arr.shape[0]), mode='constant')
                if conf_arr.shape[0] != target_conf_len:
                    if conf_arr.shape[0] > target_conf_len:
                        conf_arr = conf_arr[:target_conf_len]
                    else:
                        conf_arr = np.pad(conf_arr, (0, target_conf_len - conf_arr.shape[0]), mode='constant')
                if self.feat_mean is not None and self.feat_std is not None:
                    # guard shape mismatch by slicing to min length
                    m = min(len(feat_arr), len(self.feat_mean))
                    feat_arr[:m] = (feat_arr[:m] - self.feat_mean[:m]) / self.feat_std[:m]
                # Ensure shapes
                if proc_img.ndim == 2:
                    proc_img = np.expand_dims(proc_img, -1)
                elif proc_img.ndim == 3:
                    # Force single channel for consistency with model input
                    if proc_img.shape[-1] != 1:
                        proc_img = proc_img[..., :1]
                # flag degenerate images (near-constant)
                if not np.isfinite(proc_img).all() or float(np.std(proc_img)) < 1e-6:
                    warnings_list.append({"path": p, "issue": "degenerate_image", "message": "non-finite or near-constant image after preprocessing"})
                images.append(proc_img.astype(np.float32))
                feats.append(feat_arr)
                confs.append(conf_arr)
                ok_paths.append(p)
            except Exception as e:
                errors.append(f"{p}: {e}")
                defects.append({"path": p, "issue": "exception", "message": str(e)})
        if not images:
            return (np.zeros((0, *self.target_size, 1), dtype=np.float32),
                    np.zeros((0, len(feats[0]) if feats else 0), dtype=np.float32),
                    np.zeros((0, len(confs[0]) if confs else 0), dtype=np.float32),
                    ok_paths,
                    errors,
                    defects,
                    warnings_list)
        images_np = np.stack(images, axis=0)
        feats_np = np.stack(feats, axis=0)
        confs_np = np.stack(confs, axis=0)
        return images_np, feats_np, confs_np, ok_paths, errors, defects, warnings_list

    def predict_paths(self,
                      paths: List[str],
                      batch_size: int = 32) -> Dict[str, Any]:
        # Chunk inputs
        results = []
        total = len(paths)
        start_time = time.time()
        defects_all: List[Dict[str, Any]] = []
        warnings_all: List[Dict[str, Any]] = []
        for i in range(0, total, batch_size):
            chunk = paths[i:i + batch_size]
            images_b, feats_b, conf_b, ok_paths, errors, defects, warns = self._prepare_items(chunk)
            defects_all.extend(defects)
            warnings_all.extend(warns)
            for err in errors:
                results.append({"path": err.split(":")[0], "error": err})
            if images_b.shape[0] == 0:
                continue
            proba = self._predict_batch(images_b, feats_b, conf_b)
            # Temperature scaling
            proba = softmax_with_temperature(probs=proba, temperature=self.temperature)
            # Threshold gating
            pred_labels, abstained = apply_thresholds(proba, self.thresholds, self.abstain_unknown, self.unknown_id)
            for j, pth in enumerate(ok_paths):
                top_idx = int(np.argmax(proba[j]))
                score = float(proba[j, top_idx])
                record = {
                    "path": pth,
                    "class": self.class_names[pred_labels[j]] if pred_labels[j] < self.num_classes else "unknown",
                    "class_id": int(pred_labels[j]),
                    "score": score,
                    "abstained": bool(abstained[j]),
                }
                # include per-class probabilities only if requested by caller (handled in CLI)
                record["_probs"] = proba[j].tolist()
                results.append(record)
        elapsed = time.time() - start_time
        # Coverage stats if abstaining
        if self.abstain_unknown:
            covered = [r for r in results if not r.get("abstained")]
            coverage = (len(covered) / len(results)) if results else 0.0
        else:
            coverage = 1.0
        meta = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_path": self.model_path,
            "thresholds_path": self.thresholds_path,
            "temperature": self.temperature,
            "tta": self.tta,
            "abstain_unknown": self.abstain_unknown,
            "unknown_id": self.unknown_id,
            "batch_size": batch_size,
            "coverage": coverage,
            "num_files": len(results),
            "classes": self.class_names,
            "elapsed_sec": elapsed,
        }
        return {"meta": meta, "results": results, "defects": defects_all, "warnings": warnings_all}


def scan_directory_recursive(input_dir: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(os.path.realpath(input_dir)):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXTS:
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="Production inference runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fs = sub.add_parser("filesystem", help="Classify all files in a directory (recursive) and save JSON report")
    fs.add_argument("--input_dir", required=True)
    fs.add_argument("--output_json", required=True)
    fs.add_argument("--model_path", required=True)
    fs.add_argument("--thresholds", default="thresholds_prod_precision.json")
    fs.add_argument("--temperature", type=float, default=0.7)
    fs.add_argument("--tta", action="store_true")
    fs.add_argument("--abstain_unknown", action="store_true")
    fs.add_argument("--unknown_id", type=int, default=None)
    fs.add_argument("--batch_size", type=int, default=32)
    fs.add_argument("--include_probs", action="store_true", help="Include per-class probabilities in the JSON")
    fs.add_argument("--defect_log", type=str, default=None, help="Optional path to write a separate defect log JSON (exceptions only)")
    fs.add_argument("--warn_log", type=str, default=None, help="Optional path to write a separate warnings log JSON (non-fatal issues)")

    args = parser.parse_args()

    if args.cmd == "filesystem":
        svc = ClassifierService(
            model_path=args.model_path,
            thresholds_path=args.thresholds,
            temperature=args.temperature,
            tta=bool(args.tta),
            abstain_unknown=bool(args.abstain_unknown),
            unknown_id=args.unknown_id,
            target_size=(448, 448),
        )
        paths = scan_directory_recursive(args.input_dir)
        payload = svc.predict_paths(paths, batch_size=args.batch_size)
        # Optionally drop per-class probs
        if not args.include_probs:
            for r in payload["results"]:
                if "_probs" in r:
                    r.pop("_probs", None)
        # Atomic write
        tmp_path = args.output_json + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, args.output_json)
        print(f"Wrote {len(payload['results'])} records to {args.output_json}")
        # Optional defect log
        if args.defect_log:
            tmp_def = args.defect_log + ".tmp"
            with open(tmp_def, "w") as f:
                json.dump(payload.get("defects", []), f, indent=2)
            os.replace(tmp_def, args.defect_log)
            print(f"Wrote {len(payload.get('defects', []))} defect entries to {args.defect_log}")
        # Optional warnings log
        if args.warn_log:
            tmp_warn = args.warn_log + ".tmp"
            with open(tmp_warn, "w") as f:
                json.dump(payload.get("warnings", []), f, indent=2)
            os.replace(tmp_warn, args.warn_log)
            print(f"Wrote {len(payload.get('warnings', []))} warning entries to {args.warn_log}")


if __name__ == "__main__":
    main()


