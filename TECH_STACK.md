# 🛠️ Project Tech Stack

This document outlines the core technologies and architectures used in the **Smart Crop Disease Detector**.

## 🧠 Deep Learning & AI
*   **Core Architecture**: **ResNet-50** (Residual Networks) pretrained on ImageNet.
*   **Framework**: **PyTorch 2.0+** with Torchvision.
*   **Input Guard**: **OpenAI CLIP** (Visual-Text AI) for zero-shot semantic plant verification.
*   **Optimization**: AdamW Optimizer, Cosine Annealing Learning Rate Scheduler.
*   **Hardware**: Specialized support for **NVIDIA CUDA** (GPU) acceleration with automatic CPU fallback.

## 📡 Backend (API)
*   **Language**: Python 3.10+
*   **Framework**: **Flask 3.0+**
*   **Cross-Origin**: Flask-CORS for secure cross-domain frontend communication.
*   **Inference**: Real-time PyTorch state-dict loading and predictive processing.

## 🌐 Frontend (Web UI)
*   **Structure**: Semantic HTML5.
*   **Styling**: Vanilla CSS3 (Custom Dark Theme with Glassmorphism).
*   **Logic**: Modern Vanilla JavaScript (Fetch API for asynchronous predictions).
*   **Responsiveness**: Fully responsive mobile-first design.

## 🛠️ Data & Processing
*   **Image Processing**: **Pillow (PIL)** for high-quality resizing and normalization.
*   **Data Visualization**: **Matplotlib** for training curve generation.
*   **Serialization**: JSON for metadata, class mapping, and training history logs.


