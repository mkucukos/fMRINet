# fMRI Task State Classification with Deep Learning

A deep learning approach for classifying cognitive task states from fMRI time series data using a custom CNN architecture adapted from EEGNet for neuroimaging data.

## Overview

This framework implements a configurable classification system for fMRI task states.  

- `num_classes`: number of task categories in your dataset  
- `input_shape`: shape of your region × time × channel matrix (default `(214, 277, 1)`)

You can train on **any set of task labels**. For example, our demo dataset includes 6 tasks:

| Task Code | Task Name                  | Label |
|-----------|----------------------------|-------|
| PVT       | Psychomotor Vigilance Task | 0 |
| VWM       | Visual Working Memory      | 1 |
| DOT       | Dot Motion Task            | 2 |
| MOD       | Modular Task               | 3 |
| DYN       | Dynamic Task               | 4 |
| rest      | Resting State              | 5 |

## Sample Dataset Information

- **Input Dimensions**: `(214, 277, 1)` → (brain regions, time points, channels)
- **Architecture**: Custom CNN adapted from EEGNet for fMRI data
- **Performance**: ~84% balanced accuracy on validation set

## Usage Notes

1. **Input requirements**: Data must be structured as `(regions × time points × channels)` arrays.  
   - Default: `(214, 277, 1)`  
   - Preprocessing (e.g., fMRIPrep) should be applied beforehand for artifact removal.  
2. **Subject-based splits**: Training and validation are split by subject IDs to prevent data leakage.  
3. **Modular architecture**: Choose between `fmriNet8`, `fmriNet16`, or `fmriNet32` depending on available compute and task complexity.  
4. **Custom constraints**: `ZeroThresholdConstraint` enforces sparsity in spatial filters.  
5. **Training setup**: Learning rate scheduling (halves every 200 epochs) and checkpointing are built-in.  
6. **Filter interpretation**: Learned temporal and spatial patterns can be visualized for interpretability.  

## Quick Start

### 1. Environment Setup

```bash
# Create and activate a new conda environment
conda create --name tf python=3.8
conda activate tf


# Install dependencies
pip install -r requirements.txt
### TensorFlow & AdamW Note

- For TensorFlow **2.11+**, `AdamW` is included in `tensorflow.keras` (no extra steps).  
- For TensorFlow **2.10.x**, `AdamW` comes from `tensorflow-addons`.  

### 2. Required Files

Ensure these files are in your project directory:
- `fMRINet/toy_dataframe.pkl` - Demo fMRI dataset (for testing)
- `fMRINet/dataframe.pkl` - Full fMRI dataset (contact author for access)
- `fMRINet/subjs.pickle` - Pre-defined subject splits
- `fMRINet/fMRINet_8.ipynb` - Main notebook
- `fMRINet/fMRINet.py` - Model architecture definitions

### 3. Run the Analysis

Execute notebook cells sequentially for complete analysis pipeline.

## 📋 Step-by-Step Workflow

### Step 1: Data Loading and Preprocessing

```python
# Load pickled toy_dataframe and subject splits
df = pd.read_pickle('toy_dataframe.pkl')
with open('subjs.pickle', 'rb') as f:
    subjs = pickle.load(f)

# Split data by subjects (no data leakage)
train_df = df[df['subject'].isin(subjs[0:45])]
valid_df = df[df['subject'].isin(subjs[45:,])]
```

### Step 2: Data Transformation

```python
# Transform data to proper tensor format
train_data = np.dstack(train_df['Time_Series_Data'])
train_data = np.expand_dims(train_data, axis=0)
train_data = np.transpose(train_data, axes=[3, 2, 1, 0])
# Final shape: (batch, regions, time_points, channels)
```

### Step 3: Class Balancing

```python
# Calculate balanced class weights
weights = sklearn.utils.class_weight.compute_class_weight(
    class_weight='balanced', 
    classes=unique_classes, 
    y=train_labels
)
```

### Step 4: Model Architecture Selection

```python
# Choose from three available architectures
from fMRINet import fmriNet8, fmriNet16, fmriNet32

# Default usage (demo dataset)
model = fmriNet8(num_classes=6)
# model = fmriNet16(num_classes=6)  # 16 temporal filters  
# model = fmriNet32(num_classes=6) # 32 temporal filters

# Custom dataset (e.g., 200 regions × 300 time points × 1 channel, 4 tasks)
model = fmriNet8(num_classes=4, input_shape=(200, 300, 1))
model.summary()
```
### Step 5: Training Configuration

```python
# Setup training parameters
model.compile(
    loss='categorical_crossentropy',
    optimizer=AdamW(weight_decay=0.0005),
    metrics=['accuracy']
)

# Callbacks
checkpointer = ModelCheckpoint('/tmp/checkpoint.h5', save_best_only=True)
def lr_schedule(epoch):
    return (0.001 * np.power(0.5, np.floor(epoch/200)))
scheduler = LearningRateScheduler(lr_schedule, verbose=1)
```

### Step 6: Model Training

```python
# Train the model
fittedModel = model.fit(
    train_data, train_label,
    batch_size=64,
    epochs=400, 
    validation_data=(valid_data, valid_label),
    callbacks=[checkpointer],
    class_weight=class_weights
)
```

### Step 7: Evaluation and Visualization

```python
# Load best weights and evaluate
model.load_weights('/tmp/checkpoint.h5')
preds = model.predict(valid_data)
balanced_accuracy = balanced_accuracy_score(
    np.argmax(valid_label, axis=1), 
    np.argmax(preds, axis=1)
)
```

## Model Architecture Details

![Summary & Explanations of fMRINet8](assets/images/model_architecture_table.jpg)


<!--  -->
### Network Structure
```
Input: (214, 277, 1)
    ↓
Dropout(0.25)
    ↓
Conv2D(8, (1,60)) - Temporal Filtering
    ↓
Permute → DepthwiseConv2D - Spatial Processing
    ↓
BatchNorm → ReLU → AveragePooling2D
    ↓
SeparableConv2D(64) - Feature Extraction
    ↓
BatchNorm → ReLU → AveragePooling2D
    ↓
Flatten → Dense(6) → Softmax
```

### Key Components

1. **ZeroThresholdConstraint**: Custom constraint for sparsity
2. **Temporal-Spatial Separation**: First temporal, then spatial processing
3. **DepthwiseConv2D**: Efficient spatial feature learning
4. **Balanced Class Weights**: Handles dataset imbalance

### Model Parameters
- **Total Parameters**: 11,558
- **Trainable Parameters**: 11,366
- **Non-trainable Parameters**: 192

##  Results

- **Validation Accuracy**: ~83-84%
- **Balanced Accuracy**: ~84%
- **Final Validation Loss**: ~0.55

## Accuracy/Loss Visualization

![Accuracy/Loss](assets/plots/accuracy_loss.png)

## Filter Visualization

The notebook includes visualization of learned filters:

- **Temporal Filters**: 8 filters showing temporal patterns across time
- **Spatial Filters**: 32 filters (8×4) showing spatiotemporal patterns across brain regions


**Temporal Filters** are visualized; 

![Temporal filters](assets/plots/temporal_filters.png)

**Spatial Filters** are visualized; 

![Spatial filters](assets/plots/spatial_filters.png)


## Important Note: 
The **Results** section and **Filter Visualization** were generated using the full dataset contained in `dataframe.pkl`.  
For methodological demonstration purposes, we also introduced a smaller `toy_dataframe`, which includes only a limited subset of the data to illustrate the workflow in a simplified manner.  

<<<<<<< HEAD
Please note that the toy dataset was **not** used to produce any of the plots or reported results.  
All final analyses and visualizations were performed exclusively on the complete dataset using the fMRI filter-based CNN architecture.

## Project Structure
=======
## 📁 Repository Structure
>>>>>>> 9f1b7f5 (added license info)

```
fMRI-PROJECT/
├── assets/                          # Project assets (figures & tables)
│   ├── images/                      # High-level tables and static diagrams
│   │   └── model_architecture_table.jpg   # Architecture summary table
│   └── plots/                       # Visualization outputs
│       ├── spatial_filters.png     # Learned spatial filter visualization
│       └── temporal_filters.png    # Learned temporal filter visualization
│
├── fMRINet/                         # Main project directory
│   ├── fMRINet_8.ipynb             # Main analysis notebook
│   ├── fMRINet.py                # Model architecture definitions
│   ├── toy_dataframe.pkl           # Demo dataset (for testing) - [Actual data provided on the side of authors.]
│   └── subjs.pickle                # Subject ID splits for reproducibility
│
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Acknowledgments

This project adapts and extends the [EEGNet/EEGModels framework](https://github.com/vlawhern/arl-eegmodels) originally developed by Vernon J. Lawhern and colleagues at the Army Research Laboratory.  
<<<<<<< HEAD

Their work on CNN architectures for EEG classification provided the foundation for the temporal–spatial convolutional design used here, which we have customized for fMRI task-state classification.



=======
Their work on CNN architectures for EEG classification provided the foundation for the temporal–spatial convolutional design used here, which we have customized for fMRI task-state classification.

License This project is licensed under the MIT License — see the LICENSE
>>>>>>> 9f1b7f5 (added license info)
