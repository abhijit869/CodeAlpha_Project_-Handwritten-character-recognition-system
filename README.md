# 🧠 Handwritten Character Recognition (EMNIST) — Colab Ready ✅

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Google Colab](https://img.shields.io/badge/Colab-Ready-yellow?logo=googlecolab)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A **fully working and tested deep learning project** for **handwritten character recognition** using the **EMNIST Balanced dataset**, implemented in **TensorFlow + Keras** and tested end-to-end in **Google Colab**.  

It can **train**, **evaluate**, and **predict** handwritten **letters (A–Z, a–z)** and **digits (0–9)** from uploaded images.  
All image preprocessing, dataset handling, and visualization code works flawlessly in Colab.

---

## 🚀 Key Highlights
- ✅ **100% functional & tested in Google Colab**
- 🧩 **EMNIST Balanced** dataset for letters and digits
- 🧠 **CNN architecture** (Convolutional Neural Network)
- 🖼️ **Image upload + preprocessing** using OpenCV & Pillow
- 📊 **Accuracy, loss & confusion matrix** visualization
- 💾 **Automatic model saving** (`.h5`)
- 🔍 **Real-time prediction** on custom images

---

## 📂 Project Structure

| File | Description |
|------|--------------|
| `handwriting_robust_balanced_model.py` | Main training + testing script |
| `handwriting_recognition_model.h5` | Saved trained model |
| `mnist_test.xlsx` | Example test data / predictions |
| `requirements.txt` | Dependencies list |
| `README.md` | Documentation (this file) |

---

## ⚙️ Setup in Google Colab

### 🔹 Step 1: Clone Repository

```bash
!git clone https://github.com/<your-username>/handwritten-character-recognition-emnist.git
%cd handwritten-character-recognition-emnist
```

### 🔹 Step 2: Install Dependencies

```bash
!pip install -r requirements.txt
```

### 🔹 Step 3: Run Training

```bash
!python handwriting_robust_balanced_model.py
```
This will:
- Load EMNIST Balanced.
- Train the CNN model.
- Save the trained model as `handwriting_recognition_model.h5`.
- Display accuracy/loss graphs.

### 🔹 Step 4: Upload & Test Images

You can upload any `.png` or `.jpg` handwritten character:

```python
from google.colab import files
from tensorflow.keras.models import load_model
import cv2, numpy as np

uploaded = files.upload()
model = load_model('handwriting_recognition_model.h5')

for filename in uploaded.keys():
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    pred = np.argmax(model.predict(img))
    print(f"{filename} → Predicted Class: {pred}")
```

---

## 🧠 Model Architecture

- Conv2D → ReLU → MaxPooling → Dropout
- Flatten → Dense → Dropout → Softmax

**Optimizer:** Adam  
**Loss:** categorical_crossentropy  
**Metrics:** accuracy

---

## 📊 Example Output

```yaml
Epoch 10/10
Accuracy: 97.42%
Validation Accuracy: 96.88%
```

