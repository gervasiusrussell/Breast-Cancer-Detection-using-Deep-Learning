# 🩺 Automated Detection of Invasive Ductal Carcinoma (IDC)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **A Deep Learning solution to assist pathologists in identifying Invasive Ductal Carcinoma (IDC) in breast histology images with high sensitivity.**

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [The Dataset](#-the-dataset)
- [Methodology & Workflow](#-methodology--workflow)
    - [1. Exploratory Data Analysis](#1-exploratory-data-analysis-eda)
    - [2. Patient-Centric Splitting](#2-patient-centric-splitting-crucial)
    - [3. Model Architecture](#3-model-architecture)
- [Experimental Results](#-experimental-results)
- [Key Findings](#-key-findings)
- [Usage](#-usage)
- [Credits](#-credits)

---

## 🔍 Project Overview

**Invasive Ductal Carcinoma (IDC)** is the most common form of breast cancer. Pathologists currently diagnose IDC by manually inspecting whole-mount tissue slides, a process that is time-consuming and prone to human fatigue.

**Objective:** To build a robust Convolutional Neural Network (CNN) capable of classifying $50 \times 50$ pixel image patches of breast tissue as either:
* **0 (Non-IDC):** Benign/Healthy tissue.
* **1 (IDC):** Malignant cancer tissue.

**Goal:** Prioritize **Recall (Sensitivity)** to ensure the model minimizes False Negatives (missing a cancer case is worse than a false alarm).

---

## 📊 The Dataset

* **Source:** [Breast Histopathology Images](https://www.kaggle.com/paultimothymooney/breast-histopathology-images) (Original curation by Paul Mooney).
* **Size:** 277,524 image patches.
* **Format:** $50 \times 50 \times 3$ (RGB).
* **Class Imbalance:** * Healthy (~72%)
    * Cancer (~28%)

---

## ⚙️ Methodology & Workflow

### 1. Exploratory Data Analysis (EDA)
* **Pixel Intensity:** Analyzed RGB channel distributions. Discovered that cancerous tissue typically exhibits lower mean pixel intensity (darker/purple) due to hypercellularity and nuclear atypia, matching H&E staining biology.
* **Patient Variability:** Identified "Easy" patients (massive solid tumors) vs. "Hard" patients (sparse, tiny cancer spots).

### 2. Patient-Centric Splitting (Crucial)
A naive random split would cause **data leakage**, where patches from the same patient appear in both Train and Test sets. To fix this:
1.  Grouped data by `Patient_ID`.
2.  Calculated "Cancer Severity" per patient.
3.  **Stratified Split:** Divided patients into Train/Val/Test sets ensuring an equal distribution of Low, Medium, and High severity cases in all sets.

### 3. Model Architecture
We experimented with Transfer Learning (VGG16) but found a **Custom Lightweight CNN (CancerNet)** performed best.

**Final Model Structure:**
* **Input:** (50, 50, 3)
* **3x Convolutional Blocks:** Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout.
* **Global Average Pooling (GAP):** Used instead of Flattening to reduce parameters and prevent overfitting.
* **Dense Head:** Fully connected layers with heavy Dropout (0.5).
* **Output:** Sigmoid activation for binary classification.

**Training Strategy:**
* **Class Weights:** Applied to penalize the model heavily for missing the minority class (Cancer).
* **Callbacks:** `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`.

---

## 🏆 Experimental Results

To improve robustness, **Test-Time Augmentation (TTA)** was used during inference. Predictions were averaged across 3 views (Original, Horizontal Flip, Vertical Flip).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **86%** |
| **Recall (Sensitivity)** | **86%** |
| **Precision** | **75%** |
| **F1-Score** | **0.79** |

### Confusion Matrix Interpretation (Test Set)
* **True Positives (Cancer Found):** ~11,651
* **False Negatives (Cancer Missed):** ~1,945
* **False Positives (False Alarm):** ~3,884

> The trade-off results in a higher False Positive rate to ensure a **High Recall**, making this a safe screening tool for pathologists.

---

## 💡 Key Findings

1.  **Custom vs. Pre-trained:** A custom CNN trained from scratch outperformed VGG16 for this specific low-resolution task ($50 \times 50$ px).
2.  **TTA is powerful:** Averaging predictions across flipped versions of the image smoothed out noise and improved generalization.
3.  **Context Matters:** Reconstructing the whole-slide images confirmed that the model effectively delineates tumor boundaries, even though it only sees small patches.

---

## 🚀 Usage

### Prerequisites
* Python 3.x
* TensorFlow / Keras
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-Learn

### Running the Project
1.  Clone the repository.
2.  Download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/breast-histopathology-images) and place it in the root directory.
3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook BreastCancerDetection.ipynb
    ```

---

## 👤 Acknowledgement

https://www.kaggle.com/code/allunia/breast-cancer?scriptVersionId=66390280

https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/data

https://www.kaggle.com/code/paultimothymooney/predict-idc-in-breast-cancer-histology-images
