# 🧠 Upper-Limb Movement Detection via Inertial Images & Deep Learning

A robust and transferable Human Activity Recognition (HAR) pipeline that transforms raw inertial data into RGB images and uses deep learning (EfficientNet-B0) to detect upper-limb point-to-point (P2P) movements — a critical metric in stroke rehabilitation. Also applicable to daily activity classification.

---

## 📌 Highlights

- 📈 **85–90% Accuracy** across clinical (stroke) and real-world datasets
- 🖼️ **Inertial Image Representation** using XY, XZ, YZ plane-wise projections
- ⚙️ **Transfer Learning** with EfficientNet-B0 for robust feature extraction
- 🔁 **LOSOCV & Stratified Splits** for strong evaluation and generalization
- 💡 **No Data Augmentation** used – baseline results show high potential


---

## 📊 Datasets

### 🧠 Stroke Dataset
- **Subjects**: 18 stroke survivors
- **Activities**: Point-to-Point (P2P), Wrist Rotation (WR), Repetitive (Rep), Hand-to-Mouth (H2M), Discard
- **Challenges**: Small data, class imbalance, impaired motor signals

### 🏃 RealWorld HAR Dataset
- **Subjects**: 7 healthy individuals
- **Activities**: Walking, Running, Sitting, Standing, Climbing
- **Strength**: Balanced and diverse movement signatures

---

## 🔧 Preprocessing Pipeline

| Step             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Filtering**     | Butterworth low-pass filter (5 Hz) to remove sensor noise                   |
| **Segmentation**  | Stroke: Expert-annotated windows; HAR: 2s sliding window (50% overlap)      |
| **Normalization** | Signals scaled to [0,127] to match RGB pixel intensity range                |
| **Image Creation**| XY (Red), XZ (Green), YZ (Blue) trajectory encoding into RGB image (128×128)|

---

## 🧠 Model Architecture

- 🔍 **EfficientNet-B0** as frozen feature extractor (pre-trained on ImageNet)
- 🎯 **Custom Classification Head** with dropout and L2 regularization
- 🔁 **AdaptiveAvgPool2d** for global feature condensation
- 📐 Input Resized from 128x128 → 224x224
- ⚙️ Optimized with Adam + Cosine Annealing Scheduler

---

## 📈 Evaluation Strategy

| Evaluation Type          | Description                                                         |
|--------------------------|---------------------------------------------------------------------|
| **Stratified Splits**     | Balanced folds preserving subject and class distributions           |
| **LOSOCV**                | Leave-One-Subject-Out Cross Validation for user-level generalization|
| **Metrics**               | Accuracy, Confusion Matrix, ROC-AUC curves                         |

---

## ✅ Results Summary

### Stroke Dataset
| Class     | ROC-AUC (LOSOCV) | ROC-AUC (Stratified) |
|-----------|------------------|-----------------------|
| P2P       | 0.89             | 0.90                  |
| WR        | 0.67             | 0.66                  |
| H2M       | 0.76             | 0.73                  |
| Rep       | 0.89             | 0.86                  |
| Discard   | 0.94             | 0.94                  |

### RealWorld HAR Dataset
| Class       | ROC-AUC (LOSOCV) | ROC-AUC (Stratified) |
|-------------|------------------|-----------------------|
| Walking     | 0.88             | 0.90                  |
| Running     | 0.86             | 0.84                  |
| Sitting     | 0.91             | 0.91                  |
| Standing    | 0.81             | 0.77                  |
| ClimbingUp  | 0.85             | 0.85                  |


