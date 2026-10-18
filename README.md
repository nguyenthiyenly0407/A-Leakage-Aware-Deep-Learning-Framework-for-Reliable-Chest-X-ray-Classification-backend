# A Leakage-Aware Deep Learning Framework for Reliable Chest X-ray Classification 🫁

[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Authors:** Ly Yen Nguyen-Thi, Linh To Tran, Thien B. Nguyen-Tat
> **Institution:** University of Information Technology, VNU-HCM

## Overview:
This repository contains the implementation of a deep learning framework designed for the reliable multi-class classification of Chest X-rays (CXR). Our work addresses two critical issues in medical image analysis: **data leakage** and **model interpretability**.
## Graphical Abstract:
<img width="567" height="769" alt="Graphical abstract" src="https://github.com/user-attachments/assets/61b1fe98-4657-432b-8eb3-36bb3dad7bf4" />


**Key Contributions:**
*   **Leakage-Aware Pipeline:** We implemented a rigorous data deduplication process using MD5 and Perceptual Hashing (pHash) *before* splitting the dataset. This ensures no duplicate or near-duplicate images inflate the evaluation metrics across training, validation, and test sets.
*   **Anatomically Constrained Explainability:** We run a parallel **U-Net** model to extract precise lung masks. These masks are used to constrain the **Grad-CAM++** visual explanations, forcing the model to highlight only pulmonary regions and eliminating spurious background attention.
*   **High Performance:** Utilizing an **EfficientNet-B0** backbone, the classification model achieves **95.57% accuracy** and a **96.74% macro-F1 score** across four categories: COVID-19, Lung Opacity, Viral Pneumonia, and Normal. The U-Net segmentation achieved a **0.9861 Dice coefficient**.
## Architecture:
<img src="https://github.com/user-attachments/assets/7128bade-5a46-4647-9b29-4f8651d7d0a4" style="background-color: white; padding: 10px; border-radius: 5px;" />
## Repository Structure (Backend API)
This repository contains the backend code for deploying the web-based diagnostic system (as described in Section VI of our paper).

```text
├── API/                    # API endpoints for handling requests (upload CXR, get results)
├── Database/               # Database configuration and connection scripts
├── Model/                  # Inference scripts and pre-trained weights (EfficientNet-B0, U-Net)
├── create_table.py         # Script to initialize database schemas for doctor/patient records
├── server.py               # Main application entry point for the backend server
└── README.md
