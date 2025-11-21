Breast Cancer Detection using Deep Learning (IDC)

📌 Executive Summary

The goal of this project was to develop a robust Deep Learning solution capable of detecting Invasive Ductal Carcinoma (IDC) in histopathology slides. The primary clinical objective was to maximize Sensitivity (Recall) to ensure no cancer cases are missed, while maintaining a high level of Precision to minimize false alarms.

Using a dataset of 277,524 image patches, we developed a Custom Lightweight CNN optimized with Test-Time Augmentation (TTA). This approach outperformed complex transfer learning models (VGG16) and attention mechanisms (CBAM) in terms of efficiency and balance.

Final Test Set Results:

Sensitivity (Recall): 86%

Precision: 75%

False Negatives: Minimized for patient safety.

📂 Dataset

Source: Breast Histopathology Images (Kaggle)

Content: 277,524 patches of size 50x50 extracted from 162 whole-mount slide images.

Classes:

0: Non-IDC (Healthy/Benign) - ~72%

1: IDC (Cancerous) - ~28%

⚙️ Methodology

1. Data Preprocessing

Stratified Patient Split: Instead of a random split, we split data by Patient ID to prevent "Data Leakage" (ensuring the model doesn't memorize a patient's specific tissue texture).

Class Balancing: Utilized sklearn.utils.class_weight to compute weights, forcing the model to pay ~2.5x more attention to the minority Cancer class during training.

Streaming Pipeline: Implemented tf.data.Dataset with CPU-pinning to stream batches to the GPU, preventing VRAM crashes on large datasets.

2. Model Architecture (The Winner: Custom CNN)

We experimented with multiple architectures:

Custom CNN (Baseline): 3 Conv Blocks + Flatten + Dense.

CancerNet V2: Deeper (6 Conv) + Global Average Pooling + ELU.

Attention Model: Custom CNN + CBAM (Channel/Spatial Attention).

Transfer Learning: VGG16 (Fine-tuned).

Verdict: The Standard Custom CNN proved to be the most effective. Complex models like VGG16 provided marginally better Recall but were computationally heavy and prone to overfitting noise. V2 with Global Average Pooling failed to detect small tumor features due to signal dilution.

3. Optimization: Test-Time Augmentation (TTA)

To reduce False Positives, we implemented a TTA strategy during inference:

Predict on Original Image.

Predict on Horizontally Flipped Image.

Predict on Vertically Flipped Image.

Average the probabilities.

This consensus mechanism reduced False Alarms by ~100 cases in the Test Set without retraining.

📊 Results

The model was evaluated on a held-out Test Set (41,883 images) from patients never seen during training.

Metric

Baseline (Single Pass)

TTA (3-View Average)

Improvement

Recall (Sensitivity)

0.8569

0.8572

+4 cases found

Precision

0.7500

0.7548

+0.48% accuracy

False Alarms (FP)

3,884

3,786

-98 false scares

Missed Cancers (FN)

1,945

1,941

-4 missed tumors

Clinical Interpretation: The system successfully acts as a "Digital Highlighter," flagging 86% of cancerous regions while being correct 3 out of 4 times it raises an alarm.

🖼️ Visual Validation

We reconstructed whole-slide masks to verify the model's spatial understanding.

(Placeholder: Insert the 'Ground Truth vs Prediction' side-by-side image generated in the notebook here)

The Blue regions (AI Predictions) closely follow the Red regions (Ground Truth), proving the model learned biological morphology (tissue density, nuclear atypia) rather than statistical noise.

🚀 How to Run

Prerequisites

Python 3.9+

TensorFlow 2.10 (for Windows Native GPU) or 2.15+ (Linux/WSL2)

NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn

Installation

pip install tensorflow pandas numpy matplotlib seaborn scikit-learn


Usage

Download Data: Place the IDC_regular_ps50_idx5 folder in the project root.

Training: Run the Jupyter Notebook Breast_Cancer_Detection.ipynb.

Inference:

from predictor import CancerPredictor

# Load the TTA-enabled predictor
model = CancerPredictor('models/cancer_model_final.h5')

# Predict on a single image array (50, 50, 3)
result = model.predict(my_image)
print("Cancer Detected" if result == 1 else "Healthy")


📝 Future Improvements

Stain Normalization: Apply Macenko normalization to handle color variations between different labs.

Ensemble Learning: Combine the Custom CNN with an EfficientNetB0 to capture different feature scales.

Deployment: Wrap the model in a Flask/Streamlit API for a web-based diagnostic tool.

🤝 Credits

Dataset: [Paul Timothy Mooney
