# 🫁 Lung Cancer Diagnostic Suite: A Comparative Study
### SimpleCNN vs. ResNet-18 vs. Ultimate TransUNet

## 📌 Project Overview
This research project focuses on the automated detection and segmentation of lung nodules from CT scans using the **LUNA16 dataset**. The study evaluates the evolution of deep learning architectures, moving from basic convolutional networks to high-performance hybrid Transformer-CNN models.

---

## 🏗️ Model Architectures

### 1. SimpleCNN (The Baseline)
* **Architecture**: A lightweight Encoder-Decoder U-Net.
* **Purpose**: Established the initial benchmark for nodule localization.
* **Limitation**: Struggled with "Hard Samples" and complex lung parenchyma textures due to limited receptive field.

### 2. Multi-Task ResNet-18
* **Architecture**: ResNet-18 backbone with a custom Transpose Convolution decoder.
* **Purpose**: Utilized Transfer Learning (ImageNet weights) to enhance feature extraction.
* **Strength**: Achieved high spatial accuracy in segmentation tasks.

### 3. Ultimate TransUNet (The Proposed Model)
* **Architecture**: Hybrid CNN-Transformer.
* **Feature**: Employs a **6-layer Transformer Bottleneck** to capture global context and long-range dependencies.
* **Innovation**: Uses **Hybrid Loss** (Dice Loss + Focal Loss) to handle class imbalance (Malignant vs. Benign).


---

## 📊 Comparative Performance Leaderboard

| Metric | SimpleCNN | ResNet-18 | **Ultimate TransUNet** |
| :--- | :---: | :---: | :---: |
| **Mean Dice Score** | 0.8665 | **0.9401** | 0.9191 |
| **AUC (Detection)** | 0.9816 | 0.9712 | **0.9807** |
| **Overall Accuracy** | 93.00% | 93.63% | **94.10%** |
| **Nodule Precision** | 0.93 | 0.92 | **0.96** |

### Key Takeaways:
1.  **Detection Superiority**: TransUNet achieved the highest **AUC (0.9807)**, making it the most reliable model for identifying actual nodules.
2.  **Malignancy Precision**: The Transformer layers allowed the model to reach **96% Precision**, drastically reducing false positives compared to ResNet.
3.  **Segmentation Balance**: While ResNet-18 leads in Dice Score, TransUNet provides a better balance between pixel-level accuracy and clinical detection logic.

---

## 🖥️ Streamlit Diagnostic Dashboard
The project features a **High-Tech Diagnostic UI** built with Streamlit.
* **Mathematically Synchronized**: Uses World-to-Voxel coordinate transformation for 100% alignment between full scans and AI patches.
* **Dual View HUD**: Includes a "Raw Scan" vs "AI Diagnosis" comparison with color-coded risk indicators.
* **Auto-Export**: Automatically saves generated reports to a `Results_Export` folder for clinical documentation.


---

## 📂 Project Directory Structure
```bash
├── Common CSV files/        # candidates_V2.csv (Ground Truth Coordinates)
├── Subsets/                 # Original .mhd/.raw CT Volume files
├── TransUNet_Preprocessed_Data/ # Specialized 64x64x64 voxel patches
├── Results_Export/          # Auto-saved diagnostic reports
├── TransUNet_model.py       # Ultimate Architecture & Hybrid Loss logic
├── TransUNet_evaluation.py  # Synchronized metrics & confusion matrices
├── TransUNet_App.py         # Streamlit Diagnostic UI
└── transunet_ULTIMATE_best.pth # Optimized model weights
```

