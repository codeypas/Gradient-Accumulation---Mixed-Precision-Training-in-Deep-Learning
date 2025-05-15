# Gradient Accumulation & Mixed Precision Training in Deep Learning

This project predicts weekly sales using a neural network model trained on the WorldMart Weekly Sales dataset. It demonstrates and compares three training techniques: **Standard Precision Training**, **Gradient Accumulation** (using PyTorch), and **Mixed Precision Training** (using TensorFlow/Keras). These approaches help optimize training performance and memory usage, especially on resource-constrained hardware.

## Features

- 📊 Sales prediction using a fully connected neural network
- 🧼 Data preprocessing (handling dates, label encoding, normalization)
- 🧺 Custom PyTorch `Dataset` and `DataLoader` for batch training
- 🔁 Gradient Accumulation to simulate larger batch sizes in PyTorch
- ⚡ Mixed Precision Training in TensorFlow for faster computation
- 📉 Loss visualization using Matplotlib
- 🚀 Supports CUDA-enabled GPU acceleration for faster training (if available)

## Technologies Used

- Python, Pandas, NumPy, Matplotlib
- PyTorch (Standard & Gradient Accumulation)
- TensorFlow (Mixed Precision)
- CUDA-enabled GPU support for PyTorch and TensorFlow

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/codeypas/weekly-sales-prediction.git
   cd weekly-sales-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place `train.csv` in the root directory.

4. Run the desired training script:
   - For Gradient Accumulation (PyTorch):
     ```bash
     python gradient.py
     ```
   - For Mixed Precision (TensorFlow):
     ```bash
     python mixed.py
     ```
   - For Standard Training (PyTorch):
     ```bash
     python standard.py
     ```

> Note: The scripts automatically detect and use CUDA-enabled GPUs if available for faster training.


---

Feel free to ⭐ the repo if you find it helpful!

