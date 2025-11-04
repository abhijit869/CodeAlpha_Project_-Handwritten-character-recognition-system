# CodeAlpha_Project_-Handwritten-character-recognition-system
# 🧠 Handwritten Character Recognition using EMNIST

A **deep learning-based handwritten character recognition system** built with **TensorFlow**, **Keras**, and **OpenCV**.  
This project uses the **EMNIST Balanced dataset** to train a robust Convolutional Neural Network (CNN) capable of recognizing handwritten **letters (A–Z, a–z)** and **digits (0–9)**.  

The implementation is **Google Colab ready**, includes **image preprocessing for custom uploads**, and provides **detailed visualizations** for performance tracking.

---

## 🚀 Features
- ✅ Train on **EMNIST (Balanced)** dataset  
- 🧩 Recognizes **letters and digits**  
- ⚡ Uses **CNN architecture** with dropout regularization  
- 📈 Generates **accuracy/loss graphs**  
- 🖼️ Supports **custom image upload** for prediction  
- 💾 Automatically saves trained model (`.h5` file)  
- 🔍 Includes **confusion matrix** and **classification report**

---

## 🧠 Model Overview
The model is a **Convolutional Neural Network (CNN)** optimized for handwritten data recognition:
- Convolutional layers with ReLU activation  
- MaxPooling for dimensionality reduction  
- Dropout layers for generalization  
- Dense layers for feature mapping  
- Softmax activation for final classification

---

## 🧩 Dataset Information
**Dataset:** [EMNIST Balanced](https://www.nist.gov/itl/products-and-services/emnist-dataset)  
**Classes:** 47 (26 uppercase + 26 lowercase + 10 digits merged)  
**Input Shape:** 28x28 grayscale images  

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/handwritten-character-recognition-emnist.git
cd handwritten-character-recognition-emnist

2️⃣ Install Dependencies

Install all necessary Python packages using:

pip install -r requirements.txt

3️⃣ Run the Project

To train the model:

python handwriting_robust_balanced_model.py


Or to test pre-trained models:

python handwriting_recognition_model.py

Loaded using:
python
import tensorflow_datasets as tfds

📈 Results Visualization

The model training displays:

Accuracy vs Epochs

Loss vs Epochs

Confusion Matrix for evaluation

🧑‍💻 Author

Abhijit Biswas
Deep Learning Developer | AI & Computer Vision Enthusiast

🌐 GitHub: https://github.com/abhijit869/CodeAlpha_Project_-Handwritten-character-recognition-system
emnist = tfds.load('emnist/balanced')

