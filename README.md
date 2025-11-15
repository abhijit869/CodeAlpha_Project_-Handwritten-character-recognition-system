# Handwritten Character Recognition System

**Fast, Robust & Interactive Deep Learning Pipeline for Digits and Letters (MNIST/EMNIST)**

---

## 🚀 Features

- **Train in ~2 minutes (MNIST) or robust mode (EMNIST Balanced)**
- **Accurate recognition** of handwritten digits, uppercase, and lowercase letters
- **Interactive testing** for your own handwritten images (with Google Colab upload or local image path)
- **Real-time visualizations:** training curves, confusion matrix, prediction previews
- **Image preprocessing pipeline:** cropping, resizing, centering (with Center of Mass for robust mode)
- **High performance:** mixed precision, early stopping, model checkpointing
- **Production-ready code** with modular classes (*Config, DataLoader, CNN, Trainer, Tester, Evaluator*)
- **Colab ready** or run locally

---

## 📦 Included Files

- `handwriting_recognition_model.py` – Fast digits-only pipeline (**Python**)
- `handwriting_robust_balanced_model.py` – Robust pipeline for digits + A-Z + a-z (**Python**)
- `mnist_test.csv` – Sample CSV-formatted digit images (**CSV format**) for testing/evaluation

---

## 🛠️ Requirements

See [`requirements.txt`](requirements.txt) for detailed libraries.

### **Programming Language**
- **Python 3.6+**

### **Tools, Libraries & Frameworks**
- **TensorFlow** (incl. Keras and mixed-precision)
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Pillow**
- **scikit-learn**
- **OpenCV**
- *(optional for EMNIST)*: **tensorflow-datasets**
- *(optional for Colab testing)*: **google.colab**

---

## ✨ Quick Start (Google Colab recommended)

> **For MNIST digits only (quick demo):**
```python
!pip install tensorflow numpy matplotlib seaborn pillow scikit-learn opencv-python
python handwriting_recognition_model.py
```

> **For robust character recognition (digits, A-Z, a-z):**
```python
!pip install tensorflow tensorflow-datasets numpy matplotlib seaborn pillow scikit-learn opencv-python
python handwriting_robust_balanced_model.py
```

---

## 🖍️ Interactive Testing

### **Google Colab**
- After model training completes, you'll see prompts to upload and test your own handwritten images using:
    - MS Paint drawings
    - Scanned photos (white background, black pen)
    - PNG/JPG/BMP/etc

### **Local Testing**
- Use the provided `ImageTester` class in each script:
```python
from handwriting_recognition_model import ImageTester
tester = ImageTester(model)
tester.test_single_image('path/to/your_image.png')
```

---

## 🧠 Model Customization

- Change data subset (`Config.USE_DATA_SUBSET` for quick testing)
- Increase epochs for better accuracy (`Config.EPOCHS`)
- Switch between MNIST and EMNIST as needed (`Config.DATASET` in config)

---

## 📈 Training & Evaluation Outputs

- **Accuracy/Loss curves**
- **Confusion matrix** (matplotlib + seaborn)
- **Top 5 predictions per image** (with confidence)
- **Preprocessing visualization pipeline**

---

## 📝 CSV Testing

You can evaluate with ready-to-use csv-formatted digit images (`mnist_test.csv`). Just load as numpy arrays and feed to the model.

---

## 🙏 Credits

- [EMNIST Dataset](https://www.tensorflow.org/datasets/community_catalog/huggingface/emnist)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- Project by [Abhijit869](https://github.com/abhijit869)

---

## 📄 License

**MIT License**  
See [LICENSE](LICENSE) in this repository for full details.

---

## 📣 Get Started!

1. Clone/download the repo
2. Install required packages (`requirements.txt`)
3. Run either script (pick your accuracy/speed tradeoff)
4. Visualize results, test your own handwriting

**Happy coding & recognition! 🎉**

