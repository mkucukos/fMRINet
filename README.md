# fMRINet

<p align="center">
  <img src="assets/images/model_architecture_table.jpg" width="80%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/framework-TensorFlow-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/framework-PyTorch-red" alt="PyTorch">
</p>

> **PyTorch users:** A PyTorch implementation is available on the [`torch` branch](https://github.com/mkucukos/fMRINet/tree/torch).

A lightweight deep learning framework for classifying cognitive task states from fMRI time series data. The architecture adapts the [EEGNet](https://github.com/vlawhern/arl-eegmodels) design — separating temporal dynamics from spatial mixing via depthwise convolution — to region × time fMRI inputs, achieving ~84% balanced accuracy across six cognitive tasks.

---

## Overview

| Component | File |
|-----------|------|
| Model definitions | `fMRINet/fMRINet.py` |
| Training & evaluation notebook | `fMRINet/fMRINet_8.ipynb` |
| Demo dataset | `fMRINet/toy_dataframe.pkl` |
| Subject splits | `fMRINet/subjs.pickle` |

---

## Project Structure

```
fMRINet/
├── fMRINet/
│   ├── fMRINet_8.ipynb          # Main analysis notebook
│   ├── fMRINet.py               # Model architecture (fmriNet8/16/32)
│   ├── toy_dataframe.pkl        # Demo dataset (subset, for testing)
│   └── subjs.pickle             # Pre-defined subject splits
├── assets/
│   ├── images/
│   │   └── model_architecture_table.jpg
│   └── plots/
│       ├── accuracy_loss.png
│       ├── spatial_filters.png
│       └── temporal_filters.png
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

```bash
# Python 3.10 recommended
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note**: `AdamW` is available in `tensorflow.keras.optimizers` from TF 2.11+. TF 2.10.x users should install `tensorflow-addons` separately.

---

## Datasets

The demo `toy_dataframe.pkl` is a small subset included in this repo for workflow testing. The full `dataframe.pkl` used for the reported results is not publicly distributed — contact the author for access.

Each row in the DataFrame contains:

| Column | Description |
|--------|-------------|
| `Task` | Integer class label (0–5) |
| `Time_Series_Data` | 2D array of shape `(regions, time_points)` |
| `subject` | Subject ID string |
| `session` | Session number |

Supported task labels:

| Code | Task | Label |
|------|------|-------|
| PVT | Psychomotor Vigilance Task | 0 |
| VWM | Visual Working Memory | 1 |
| DOT | Dot Motion Task | 2 |
| MOD | Modular Task | 3 |
| DYN | Dynamic Task | 4 |
| rest | Resting State | 5 |

---

## Pipeline

### 1 — Data Loading

```python
import pandas as pd, pickle
df = pd.read_pickle('toy_dataframe.pkl')   # or dataframe.pkl for full data

with open('subjs.pickle', 'rb') as f:
    subjs = pickle.load(f)

train_df = df[df['subject'].isin(subjs[0:45])]
valid_df = df[df['subject'].isin(subjs[45:])]
```

### 2 — Data Transformation

```python
import numpy as np

train_data = np.dstack(train_df['Time_Series_Data'])
train_data = np.expand_dims(train_data, axis=0)
train_data = np.transpose(train_data, axes=[3, 2, 1, 0])
# Final shape: (batch, regions, time_points, 1)
```

### 3 — Class Balancing

```python
import sklearn.utils.class_weight

train_labels = np.argmax(train_label, axis=1)
unique = np.unique(train_labels)
weights = sklearn.utils.class_weight.compute_class_weight(
    class_weight='balanced', classes=unique, y=train_labels
)
class_weights = dict(enumerate(weights))
```

### 4 — Model Selection

Three variants are available, differing only in the number of temporal filters:

```python
from fMRINet import fmriNet8, fmriNet16, fmriNet32

model = fmriNet8(num_classes=6, input_shape=(214, 277, 1))
# model = fmriNet16(num_classes=6)
# model = fmriNet32(num_classes=6)

# Custom dataset (e.g. 200 regions × 300 time points, 4 tasks)
model = fmriNet8(num_classes=4, input_shape=(200, 300, 1))
model.summary()
```

### 5 — Training

```python
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import AdamW

model.compile(
    loss='categorical_crossentropy',
    optimizer=AdamW(weight_decay=0.0005),
    metrics=['accuracy']
)

checkpointer = ModelCheckpoint('/tmp/checkpoint.h5', save_best_only=True)

def lr_schedule(epoch):
    return 0.001 * np.power(0.5, np.floor(epoch / 200))

scheduler = LearningRateScheduler(lr_schedule, verbose=1)

fittedModel = model.fit(
    train_data, train_label,
    batch_size=64,
    epochs=400,
    validation_data=(valid_data, valid_label),
    callbacks=[checkpointer],
    class_weight=class_weights
)
```

### 6 — Evaluation

```python
from sklearn.metrics import balanced_accuracy_score

model.load_weights('/tmp/checkpoint.h5')
preds = model.predict(valid_data)
balanced_accuracy_score(np.argmax(valid_label, axis=1), np.argmax(preds, axis=1))
```

---

## Model Architecture

### Network Structure

```
Input: (214, 277, 1)
    ↓
Dropout(0.25)
    ↓
Conv2D(8, (1, 60)) — Temporal filtering
    ↓
Permute → DepthwiseConv2D(depth_multiplier=4) — Spatial filtering
    ↓
BatchNorm → ReLU → AveragePooling2D(1, 15)
    ↓
Dropout(0.5) → SeparableConv2D(64, (1, 8)) — Feature extraction
    ↓
BatchNorm → ReLU → AveragePooling2D(1, 4)
    ↓
Flatten → Dense(num_classes) → Softmax
```

### Key Design Choices

| Component | Purpose |
|-----------|---------|
| `ZeroThresholdConstraint` | Enforces sparsity in spatial filters (weights < 0.025 zeroed) |
| Temporal-then-spatial ordering | Separates temporal dynamics from spatial mixing |
| `DepthwiseConv2D` | Efficient per-filter spatial weighting across brain regions |
| Balanced class weights | Compensates for class imbalance at training time |

### Parameters (fmriNet8)

| | Count |
|-|-------|
| Total | 11,558 |
| Trainable | 11,366 |
| Non-trainable | 192 |

---

## Results

Results reported here are from the **full dataset** (`dataframe.pkl`), not the toy subset.

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~83–84% |
| Balanced Accuracy | ~84% |
| Final Validation Loss | ~0.55 |

### Accuracy / Loss

![Accuracy/Loss](assets/plots/accuracy_loss.png)

### Learned Filters

**Temporal filters** — 8 filters capturing temporal dynamics across the fMRI time series:

![Temporal filters](assets/plots/temporal_filters.png)

**Spatial filters** — 32 filters (8 temporal × 4 depth) showing spatiotemporal patterns across brain regions:

![Spatial filters](assets/plots/spatial_filters.png)

---

## Reproducibility

- Subject splits are fixed via `subjs.pickle` — no randomness in train/validation assignment
- Learning rate schedule: halved every 200 epochs starting from 0.001
- Best model checkpoint saved to `/tmp/checkpoint.h5`

---

## Acknowledgments

This project adapts and extends the [EEGNet / EEGModels framework](https://github.com/vlawhern/arl-eegmodels) originally developed by Vernon J. Lawhern and colleagues at the Army Research Laboratory.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
