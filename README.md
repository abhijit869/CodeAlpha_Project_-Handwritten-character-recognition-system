🧠 Handwritten Character & Digit Recognition
📋 Overview

This project implements a deep learning–based handwritten recognition system capable of identifying both digits (0–9) and characters (A–Z, a–z).
It is trained using EMNIST and custom balanced datasets, optimized for real-world handwriting variation.

Two versions of the model are included:

🧩 handwriting_recognition_model — Standard model trained for general accuracy

🦾 handwriting_robust_balanced_model — Improved model trained with balanced data for thin, slanted, or noisy handwriting

🚀 Features
Auto lode Dataset are use MNIST
✅ Recognizes handwritten digits and alphabets
✅ Trained on EMNIST and custom balanced datasets
✅ Supports JPG/PNG upload prediction
✅ Includes robust preprocessing (thresholding, noise removal, centering)
✅ Evaluates test samples from Excel file (mnist_test.xlsx)
✅ 100% compatible with Google Colab or local Python

🧩 Project Structure
📁 Handwritten_Character_Recognition/
│
├── handwriting_recognition_model/           # Base CNN model script / saved weights
├── handwriting_robust_balanced_model/       # Improved model with balanced EMNIST data
├── mnist_test.xlsx                          # Test dataset for validation and evaluation
├── README.md                                # Project documentation (this file)
└── requirements.txt                         # Python dependencies (optional)

⚙️ Installation & Setup
🧰 Requirements

Python 3.8+

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Seaborn

scikit-learn

Pillow

Pandas

Install all dependencies:

pip install tensorflow tensorflow_datasets opencv-python numpy matplotlib seaborn scikit-learn pillow pandas