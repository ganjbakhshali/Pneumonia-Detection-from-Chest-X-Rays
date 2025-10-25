#  Pneumonia Detection from Chest X-Rays (RSNA Challenge)

This project focuses on **automated pneumonia detection** from **chest X-ray images**, using **deep learning (CNN)** techniques.  
The dataset is from the [**RSNA Pneumonia Detection Challenge**](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/), which includes thousands of labeled DICOM images from real patients.

---

##  Problem Description

Pneumonia is a lung infection that causes inflammation in the air sacs of one or both lungs. Detecting it from X-rays is a common diagnostic approach, but **manual interpretation** can be time-consuming and prone to error.

This project aims to:
- **Classify** chest X-ray images into *Normal (0)* and *Pneumonia (1)*.
- **Build a robust deep learning pipeline** that automates preprocessing, training, validation, and performance evaluation.

---

##  Approach Overview

The main model is based on **ResNet-18**, a well-known CNN architecture.  
We fine-tuned it for binary classification on X-ray data by modifying the first and last layers.

###  Workflow Summary

1. **Data Preparation**
   - DICOM images were read using `pydicom`.
   - Resized to `(224×224)` pixels using OpenCV.
   - Normalized to `[0,1]` range.
   - Split into **train (90%)** and **validation (10%)** sets.
   - Saved as `.npy` arrays for faster access.

2. **Model Architecture**
   - Base: `torchvision.models.resnet18`
   - Modified first convolution: to handle single-channel (grayscale) X-rays.
   - Output layer: 1 neuron for binary classification.
   - Loss: `BCEWithLogitsLoss`
   - Optimizer: `Adam (lr=1e-4)`

3. **Training Framework**
   - Implemented using **PyTorch Lightning** for cleaner training loops.
   - GPU-accelerated training.
   - Checkpointing based on validation accuracy (`val_acc`).
   - TensorBoard for training visualization.

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, and Confusion Matrix.
   - Tested with thresholds `0.5` and `0.25` to show sensitivity trade-offs.

---

##  Results

| Metric | Threshold = 0.5 | Threshold = 0.25 |
|:--|:--:|:--:|
| **Accuracy** | 0.85 | — |
| **Precision** | 0.73 | — |
| **Recall** | 0.52 | — |
| **Confusion Matrix** | `[[1950, 114], [290, 314]]` | `[[1766, 298], [158, 446]]` |

 Lowering the decision threshold increases sensitivity (recall) but decreases precision — useful in medical contexts where **missing a pneumonia case is riskier** than a false alarm.

---

##  Dependencies

| Library | Version |
|:--|:--|
| Python | 3.11 |
| PyTorch | 2.5.1 |
| PyTorch Lightning | 2.5.5 |
| TorchMetrics | 1.8.2 |
| TensorBoard | 2.20.0 |
| OpenCV | 4.10.0 |
| pandas | 2.2.3 |
| tqdm | 4.67.1 |
| pydicom | 2.4.4 |

---

##  Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ganjbakhshali/pneumonia-detection.git
cd pneumonia-detection
conda create -n torch python=3.11
conda activate torch
pip install torch torchvision torchaudio pytorch-lightning torchmetrics
pip install opencv-python pandas tqdm pydicom tensorboard
```
##  Training

Make sure your dataset folder structure looks like this:

```bash
kaggle-pneumonia/
│
├── stage_2_train_labels.csv
├── stage_2_train_images/
│   ├── 0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm
│   ├── ...
└── stage_2_test_images/
```

Checkpoints and logs will be saved under:
```bash
/proc/
└── logs/
```

##  Future Work

* Extend model to object detection (bounding box localization of pneumonia areas).

* Integrate Grad-CAM for visual interpretability.

* Evaluate on external datasets for generalization.*

