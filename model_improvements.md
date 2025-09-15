# Modell-Verbesserungen für Astronomische Bildklassifikation

## TODO-Liste (Priorisiert für 18 GB RAM)

### 🔥 HOHE PRIORITÄT (RAM-schonend, großer Impact)

1. **Streaming-Datenpipeline statt Voll-Ladevorgang** (Generator/`tf.data.Dataset`)
   - Implementierung: `data_pipeline.py` bzw. Anpassung von `data_loader.py` (Generator-Modus)
   - Ziel: Keine vollständigen `numpy`-Arrays aller Bilder im RAM halten
   - Maßnahmen: `FITS` mit `memmap=True`, Batches on-the-fly erzeugen, `prefetch` verwenden

2. **Kleinere Eingangsauflösung** (z. B. `224x224` statt `350x350`)
   - Implementierung: Parameter in `main.py` und `data_loader.py`
   - Ziel: Reduziert RAM- und Rechenbedarf signifikant

3. **On-the-fly Augmentation im `tf.data`-Pipeline (ohne Duplikate im Speicher)**
   - Implementierung: Map/augment in Pipeline, nicht via Dataset-Verdopplung
   - Ziel: Keine aufgeblähten Arrays; Augmentation nur beim Durchlauf

4. **Modell vereinfachen (Flatten → GlobalAveragePooling2D, weniger Filter)**
   - Implementierung: `model.py`
   - Ziel: Deutlich weniger Aktivierungsspeicher und Parameter

5. **Kleinere Batch-Größe (z. B. 8) und ggf. Gradientenakkumulation**
   - Implementierung: Trainings-Loop/Callback für Akkumulation (optional)
   - Ziel: RAM-Spitzen reduzieren bei ähnlicher effektiver Batch-Größe

### 🟡 MITTLERE PRIORITÄT (Solide Speichergewinne, moderater Aufwand)

6. **Leichtgewichtiges CNN (Depthwise/Separable Convs, MobileNet-ähnlich)**
   - Implementierung: `model.py`
   - Ziel: Geringerer Speicher- und Rechen-Footprint bei guter Genauigkeit

7. **Eingabedatentyp und Features auf `float16` reduzieren**
   - Implementierung: Cast im Loader/Pipeline; Modell ggf. weiter in `float32`
   - Ziel: Halbierung des Eingabe-Speichers; Vorsicht bei CPU-Leistung

8. **Batched Evaluation/Prediction** (keine Voll-Dataset-Inferenz im RAM)
   - Implementierung: `evaluate_model.py` in Batches evaluieren
   - Ziel: Stabil bei großen Testsets

### 🟢 NIEDRIGE PRIORITÄT (Mehr Aufwand, optional)

9. **TFRecords + sequentielles Streaming**
   - Implementierung: Exporter + `tf.data.TFRecordDataset`
   - Ziel: Skalierbar für sehr große Datensätze

10. **Hyperparameter-Optimierung (Optuna) mit RAM-Grenzen**
    - Implementierung: `hyperparameter_optimizer.py` (kleine Batches, EarlyStop, wenige Trials parallel)
    - Ziel: Tuning unter 18 GB stabil durchführen

11. **Ensembles/Transformer/Transfer Learning (ressourcenbewusst)**
    - Implementierung: Später, wenn Pipeline stabil ist; ggf. kleinere Backbones
    - Ziel: Genauigkeit steigern, Kosten beachten

---

## Maßnahmen zur Genauigkeitssteigerung (RAM-neutral)

### 🔥 Hohe Priorität (geringer Aufwand, guter Gewinn)

1. **Stratified Split + Reproduzierbarkeit**
   - Implementierung: `train_test_split(..., stratify=labels_int)` in `main.py`
   - Erwarteter Gewinn: +1–3% stabilere Genauigkeit; verlässlichere Evaluation
   - Aufwand: sehr gering

2. **Klassengewichte ODER Focal Loss (klassen-spezifisches alpha)**
   - Implementierung: Class Weights via `compute_class_weight` oder `focal_loss(alpha_vec, gamma)` in `model.py`
   - Erwarteter Gewinn: +3–8% für unterrepräsentierte/verwechselte Klassen
   - Aufwand: gering
   - Hinweis: Nicht gleichzeitig verwenden; zuerst Class Weights testen, dann ggf. Focal Loss

3. **Label Smoothing (0.05–0.1) + besseres LR-Scheduling**
   - Implementierung: `CategoricalCrossentropy(label_smoothing=0.05)`, `CosineDecayRestarts` oder `AdamW` mit `weight_decay`
   - Erwarteter Gewinn: +1–4% allgemein bessere Generalisierung
   - Aufwand: gering

4. **Gezielte On-the-fly Augmentation pro Klasse (ohne Duplikate)**
   - Implementierung: leichte Rotationen/Crops/Rauschen für `deep_sky`; konservativere Augmentation für `darks`/`spectrum_dados`
   - Erwarteter Gewinn: +2–6% (Recall-Boost für `deep_sky`, Präzision bei überprädizierten Klassen)
   - Aufwand: gering

5. **GlobalAveragePooling2D statt Flatten**
   - Implementierung: `model.py` (CNN-Head vereinfachen)
   - Erwarteter Gewinn: +1–3% (robustere Merkmalsaggregation), RAM-neutral
   - Aufwand: gering

### 🟡 Mittlere Priorität (moderater Aufwand, solider Gewinn)

6. **Balanced Batch Sampling**
   - Implementierung: in `tf.data` via `sample_from_datasets` oder gewichtetes Sampling je Klasse
   - Erwarteter Gewinn: +2–5%
   - Aufwand: mittel

7. **Confidence-Gating für Header-Features**
   - Implementierung: Features mit `sigmoid(a*conf+b)` gewichten, geringe/zweifelhafte Header schwächer einfließen lassen
   - Erwarteter Gewinn: +1–4% weniger Fehlklassifikation durch unzuverlässige Metadaten
   - Aufwand: mittel

8. **Konsistente Feature-Standardisierung**
   - Implementierung: Skaler nur auf Trainingssplit fitten, bei Val/Test anwenden (statt reinem BN)
   - Erwarteter Gewinn: +1–3%
   - Aufwand: mittel

### 🟢 Niedrige Priorität (geringer Aufwand, kleiner gezielter Gewinn)

9. **Heuristische Post-Processing-Regeln für kritische Verwechslungen**
   - Implementierung: z. B. `dark`-Prädiktionen verwerfen, wenn Bild-Mean/Std ungewöhnlich; einfache Grenzen in `evaluate_model.py`
   - Erwarteter Gewinn: +1–3% Präzision bei betroffenen Klassen
   - Aufwand: gering

---

## Wenn mehr RAM verfügbar: priorisierte Maßnahmen (Aufwand/Gewinn)

### 🔥 Hohe Priorität (größerer Gewinn, moderater Aufwand)

1. **Höhere Eingangsauflösung (z. B. 350–512 px)**
   - Gewinn: +2–6% (mehr Detail, vor allem für `deep_sky`/Spektren)
   - Aufwand: mittel; RAM-Bedarf ↑

2. **MixUp/CutMix + stärkere Augmentation (RandAugment)**
   - Gewinn: +2–6% robustere Generalisierung
   - Aufwand: mittel; Training etwas langsamer, RAM moderat ↑

3. **Größere Batch-Größe (stabilere BN-Statistiken)**
   - Gewinn: +1–3%
   - Aufwand: gering–mittel; RAM ↑

4. **Umfangreicheres Hyperparameter-Tuning (Optuna/RayTune) mit vielen Trials**
   - Gewinn: +3–8%
   - Aufwand: mittel–hoch; parallelisierte Trials → RAM/CPU ↑

### 🟡 Mittlere Priorität (spürbarer Gewinn, höherer Aufwand)

5. **Transfer Learning mit größeren Backbones (ResNet50V2, EfficientNetV2-B0, ViT-Tiny)**
   - Gewinn: +5–15%
   - Aufwand: mittel–hoch; RAM ↑, bevorzugt GPU

6. **K-Fold Training + Model Averaging**
   - Gewinn: +3–8% (robustere Modelle)
   - Aufwand: hoch; RAM/Compute ↑ (k Trainingsläufe)

7. **Ensembles (verschiedene Seeds/Architekturen)**
   - Gewinn: +5–12%
   - Aufwand: hoch; Inferenzkosten ↑, RAM ↑

### 🟢 Niedrige Priorität (strategisch, längerfristig)

8. **TFRecords + aggressive Caching/Prefetch**
   - Gewinn: +1–3% indirekt (mehr Epochen/konstante Pipeline möglich)
   - Aufwand: mittel; RAM/Platten-I/O Management

9. **Self-Supervised Pretraining (SimCLR/BYOL) auf eigenen Daten**
   - Gewinn: +3–10% mit genügend unlabeled Data
   - Aufwand: hoch; RAM/GPU/Compute ↑

10. **Reichere Domänen-Features (PSF/FWHM, Stern-Dichte, spektrale Kennzahlen)**
    - Gewinn: +2–6%
    - Aufwand: mittel–hoch; Feature-Engineering + Validierung

---

## Speicherleitfaden (18 GB RAM)

- **Laden**: `fits.open(path, memmap=True)` verwenden; TIFF nur bei Bedarf vollständig lesen.
- **Pipeline**: `tf.data.Dataset.from_generator` oder `from_tensor_slices(...).map(...).batch(...).prefetch(tf.data.AUTOTUNE)`
- **Augmentation**: ausschließlich on-the-fly in `map`, nicht durch Vervielfachung der Arrays.
- **Auflösung**: Start mit `224x224x1`; nur erhöhen, wenn nötig.
- **Batch-Größe**: 8 (oder 4 bei Engpässen); optional Gradientenakkumulation (Update nur alle N Schritte).
- **Dtype**: Eingaben/Features `float16`, Labels `int32`; Modellgewichtungen weiterhin `float32` (stabiler auf CPU).
- **Modell**: `GlobalAveragePooling2D` statt `Flatten`; Filterprogression konservativ (z. B. 24-48-96).
- **Evaluation**: Immer in Batches vorgehen; keine Gesamtdaten im RAM.

---

## Detaillierte Implementierungen

### RAM-freundliche On-the-fly Augmentation mit `tf.data`

```python
# data_pipeline.py (Skizze)
import tensorflow as tf

def augment_fn(image, features, confidences, label):
    # Beispielhafte, leichte Augmentationen
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.05)
    return image, features, confidences, label

def build_dataset(gen_fn, *, batch_size=8, shuffle=True, cache=False, augment=True):
    ds = tf.data.Dataset.from_generator(
        gen_fn,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float16),
            tf.TensorSpec(shape=(None,), dtype=tf.float16),
            tf.TensorSpec(shape=(4,), dtype=tf.float16),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
    if shuffle:
        ds = ds.shuffle(1024)
    if cache:
        ds = ds.cache()
    # Resize spät (im Pipeline), um I/O zu reduzieren
    def _resize(image, features, confidences, label):
        image = tf.image.resize(image, (224, 224))
        return image, features, confidences, label
    ds = ds.map(_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
```

> Hinweis: Der Generator `gen_fn` liest pro Sample on-demand (FITS `memmap=True`), konvertiert zu `float16` und liefert einzelne Beispiele. Keine Sammel-Arrays im Speicher.

### Modellvereinfachung (Flatten → GlobalAveragePooling2D)

```python
# model.py (Ausschnitt)
from tensorflow.keras import layers

def _build_image_network(self):
    image_input = layers.Input(shape=self.input_shape)
    x = layers.Conv2D(24, 3, activation='relu', padding='same')(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(48, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(96, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)  # statt Flatten

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    return models.Model(inputs=image_input, outputs=x)
```

### 1. Data Augmentation für astronomische Bilder

```python
# Hinweis: Bevorzugt tf.data (siehe oben). Die unten stehende Batch-Verdopplung
# aus der ursprünglichen Skizze sollte bei 18 GB RAM NICHT verwendet werden.
```

### 2. Hyperparameter-Optimierung
- Bitte unter RAM-Grenzen betreiben: kleine Batch-Größen, maximale Epochen begrenzen, EarlyStopping aggressiv setzen, Trials seriell statt parallel.

### 3. Focal Loss Implementation

```python
# focal_loss.py
import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss für unausgewogene Klassen
    gamma: Fokus-Parameter (höher = mehr Fokus auf schwierige Beispiele)
    alpha: Gewichtung für Klassen
    """
    def focal_loss_fn(y_true, y_pred):
        # Categorical crossentropy
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Probability der korrekten Klasse
        pt = tf.exp(-ce)
        
        # Focal Loss
        focal_loss = alpha * tf.pow(1 - pt, gamma) * ce
        
        return tf.reduce_mean(focal_loss)
    
    return focal_loss_fn

# Verwendung in model.py:
# model.compile(
#     optimizer='adam',
#     loss=focal_loss(gamma=2.0, alpha=0.25),
#     metrics=['accuracy']
# )
```

### 4. Erweiterte Bildvorverarbeitung

```python
# enhanced_preprocessing.py
import numpy as np
from skimage import exposure, filters, restoration
from scipy import ndimage

class EnhancedPreprocessor:
    def __init__(self, target_size=(350, 350)):
        self.target_size = target_size
    
    def preprocess_image(self, image):
        """Erweiterte Vorverarbeitung für astronomische Bilder"""
        # 1. Rauschunterdrückung
        image = self._denoise_image(image)
        
        # 2. Kontrastverbesserung
        image = self._enhance_contrast(image)
        
        # 3. Hintergrund-Subtraktion
        image = self._subtract_background(image)
        
        # 4. Normalisierung
        image = self._normalize_image(image)
        
        # 5. Größenanpassung
        image = self._resize_image(image)
        
        return image
    
    def _denoise_image(self, image):
        """Rauschunterdrückung mit verschiedenen Methoden"""
        # Gaussian Filter für hochfrequentes Rauschen
        image = filters.gaussian(image, sigma=0.5)
        
        # Median Filter für Salz-und-Pfeffer-Rauschen
        image = filters.median(image)
        
        return image
    
    def _enhance_contrast(self, image):
        """Kontrastverbesserung"""
        # Histogramm-Equalisierung
        image = exposure.equalize_hist(image)
        
        # Adaptive Histogramm-Equalisierung
        image = exposure.equalize_adapthist(image, clip_limit=0.03)
        
        return image
    
    def _subtract_background(self, image):
        """Hintergrund-Subtraktion"""
        # Rolling Ball Algorithm für Hintergrund-Subtraktion
        background = ndimage.gaussian_filter(image, sigma=50)
        image = image - background
        
        return image
    
    def _normalize_image(self, image):
        """Robuste Normalisierung"""
        # Percentile-basierte Normalisierung
        p5, p95 = np.percentile(image, (5, 95))
        image = (image - p5) / (p95 - p5)
        
        # Clipping auf [0, 1]
        image = np.clip(image, 0, 1)
        
        return image
    
    def _resize_image(self, image):
        """Größenanpassung mit Interpolation"""
        # Bilineare Interpolation für bessere Qualität
        from skimage.transform import resize
        image = resize(image, self.target_size, order=1, preserve_range=True)
        
        return image
```

### 5. Cross-Validation Implementation

```python
# cross_validation.py
from sklearn.model_selection import StratifiedKFold
import numpy as np
import tensorflow as tf

class CrossValidator:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.scores = []
        self.models = []
    
    def cross_validate(self, data_loader, model_builder, epochs=50):
        """Stratified K-Fold Cross-Validation"""
        # Daten laden
        images, features, confidences, labels = data_loader.prepare_dataset("../image_classification_training_sample")
        labels_categorical = tf.keras.utils.to_categorical(labels, num_classes=len(data_loader.classes))
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
            print(f"Training fold {fold + 1}/{self.n_splits}")
            
            # Daten aufteilen
            X_train, X_val = images[train_idx], images[val_idx]
            F_train, F_val = features[train_idx], features[val_idx]
            C_train, C_val = confidences[train_idx], confidences[val_idx]
            y_train, y_val = labels_categorical[train_idx], labels_categorical[val_idx]
            
            # Modell erstellen und trainieren
            model = model_builder()
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.2)
            ]
            
            history = model.fit(
                [X_train, F_train, C_train], y_train,
                validation_data=([X_val, F_val, C_val], y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Beste Validierungsgenauigkeit speichern
            best_score = max(history.history['val_accuracy'])
            fold_scores.append(best_score)
            self.models.append(model)
            
            print(f"Fold {fold + 1} - Best validation accuracy: {best_score:.4f}")
        
        # Ergebnisse zusammenfassen
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"\nCross-Validation Results:")
        print(f"Mean accuracy: {mean_score:.4f} ± {std_score:.4f}")
        print(f"Individual fold scores: {[f'{score:.4f}' for score in fold_scores]}")
        
        return mean_score, std_score, self.models
```

---

## Implementierungsplan (für 18 GB RAM)

### Phase 1 (Woche 1): RAM-First
1. Streaming-Pipeline (Generator/`tf.data`) + `memmap=True`
2. Zielauflösung 224x224 festlegen; Batch-Größe 8
3. On-the-fly Augmentation in `tf.data` integrieren
4. `Flatten` → `GlobalAveragePooling2D`; Filter reduzieren

### Phase 2 (Woche 2): Stabilisierung und Tuning
1. Eingabe/Features auf `float16` casten; Modell auf `float32` belassen
2. Batched Evaluation implementieren
3. Leichtgewichtiges CNN (separable Convs) testen

### Phase 3 (Woche 3+): Genauigkeit unter Budget erhöhen
1. RAM-bewusstes Optuna-Tuning
2. Optional: TFRecords-Pipeline
3. Optional: Kleine Ensembles oder Transfer Learning mit kleinen Backbones

---

## Monitoring und Evaluation
- Unverändert, jedoch Evaluation strikt in Batches und mit Speicherprofiling (z. B. `tracemalloc`/`psutil`).

---

*Letzte Aktualisierung: 2025-08-08*
*Status: RAM-optimierter Plan aktiv* 