"""
Handwritten Character Recognition System (Robust Version)
Production-ready implementation for Google Colab
- Uses EMNIST 'balanced' (digits, uppercase, lowercase)
- Uses Data Augmentation for robustness (rotation, zoom, shift)
- Uses Center of Mass (COM) preprocessing for user images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import cv2
import os
import time

# ============================================================================
# CONFIGURATION (Using 'balanced' dataset)
# ============================================================================

class Config:
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    CHANNELS = 1

    BATCH_SIZE = 256
    EPOCHS = 15  # Increased epochs for augmentation & harder dataset
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.1

    USE_MIXED_PRECISION = True
    USE_DATA_SUBSET = False
    SUBSET_SIZE = 10000

    # Use the 'balanced' split for digits, upper, and lower
    DATASET = 'emnist'
    EMNIST_SPLIT = 'balanced'

    MODEL_SAVE_PATH = 'handwriting_robust_balanced_model.h5'

    @staticmethod
    def get_num_classes():
        # 47 classes for 'balanced'
        if Config.DATASET == 'emnist' and Config.EMNIST_SPLIT == 'balanced':
            return 47
        elif Config.DATASET == 'emnist' and Config.EMNIST_SPLIT == 'letters':
            return 26
        else: # MNIST
            return 10

    @staticmethod
    def get_label_map():
        """Returns a map of class index to character label for 'balanced'."""
        if Config.DATASET == 'emnist' and Config.EMNIST_SPLIT == 'balanced':
            # This is the official mapping for the 'balanced' dataset
            return {
                0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: 'f', 11: 'B', 12: 'C', 13: 'D', 14: 'N', 15: 'F', 16: 'A', 17: 'H', 18: 'I',
                19: 'J', 20: 'K', 21: 'L', 22: 't', 23: 'Z', 24: 'O', 25: 'P', 26: 'B', 27: 'R',
                28: 'N', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'M',
                36: 'g', 37: 'b', 38: 'd', 39: 'e', 40: 'A', 41: 'a', 42: 'h', 43: 'n',
                44: 'q', 45: 'r', 46: 't'
            }
        else:
            # Fallback for other datasets
            map_size = Config.get_num_classes()
            return {i: str(i) for i in range(map_size)}

# ============================================================================
# DATA LOADING AND PREPROCESSING (Using 'balanced' logic)
# ============================================================================

class DataLoader:
    @staticmethod
    def load_mnist():
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        return x_train, y_train, x_test, y_test

    @staticmethod
    def load_emnist():
        print(f"Loading EMNIST dataset: {Config.EMNIST_SPLIT}...")
        try:
            import tensorflow_datasets as tfds

            split_name = f'emnist/{Config.EMNIST_SPLIT}'
            ds_train = tfds.load(split_name, split='train', as_supervised=True)
            ds_test = tfds.load(split_name, split='test', as_supervised=True)

            x_train, y_train = [], []
            x_test, y_test = [], []

            # 'letters' split is 1-indexed, 'balanced' is 0-indexed.
            is_letters_split = (Config.EMNIST_SPLIT == 'letters')

            for image, label in ds_train:
                x_train.append(image.numpy())
                y_train.append(label.numpy() - 1 if is_letters_split else label.numpy())

            for image, label in ds_test:
                x_test.append(image.numpy())
                y_test.append(label.numpy() - 1 if is_letters_split else label.numpy())

            x_train = np.array(x_train).squeeze()
            y_train = np.array(y_train)
            x_test = np.array(x_test).squeeze()
            y_test = np.array(y_test)

            return x_train, y_train, x_test, y_test

        except Exception as e:
            print(f"Error loading EMNIST: {e}")
            print("Falling back to MNIST...")
            Config.DATASET = 'mnist'
            return DataLoader.load_mnist()

    @staticmethod
    def preprocess_data(x_train, y_train, x_test, y_test):
        print("Preprocessing data...")

        if Config.USE_DATA_SUBSET:
            print(f"⚡ Using subset of {Config.SUBSET_SIZE} samples for fast training")
            indices = np.random.choice(len(x_train), Config.SUBSET_SIZE, replace=False)
            x_train = x_train[indices]
            y_train = y_train[indices]

            test_indices = np.random.choice(len(x_test), Config.SUBSET_SIZE // 5, replace=False)
            x_test = x_test[test_indices]
            y_test = y_test[test_indices]

        x_train = x_train.reshape(-1, Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.CHANNELS)
        x_test = x_test.reshape(-1, Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.CHANNELS)

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        num_classes = Config.get_num_classes()
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        print(f"Training samples: {x_train.shape[0]}")
        print(f"Test samples: {x_test.shape[0]}")
        print(f"Input shape: {x_train.shape[1:]}")
        print(f"Number of classes: {num_classes}")

        return x_train, y_train, x_test, y_test

# ============================================================================
# MODEL ARCHITECTURE (With Data Augmentation)
# ============================================================================

class HandwritingCNN:
    @staticmethod
    def build_model():
        print("Building optimized CNN model with DATA AUGMENTATION...")

        # --- NEW: Define augmentation layers ---
        data_augmentation = keras.Sequential(
            [
                layers.RandomRotation(0.1, fill_mode='constant', fill_value=0.0), # +/- 10% rotation
                layers.RandomZoom(0.1, fill_mode='constant', fill_value=0.0), # +/- 10% zoom
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value=0.0),
            ],
            name="data_augmentation",
        )

        # --- MODIFIED: Add an Input layer and the augmentation layer ---
        model = models.Sequential([
            # Add explicit Input layer
            layers.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.CHANNELS)),

            # Add the augmentation layers
            # These layers are ONLY active during training
            data_augmentation,

            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            # Output Layer
            layers.Dense(Config.get_num_classes(), activation='softmax')
        ])

        return model

    @staticmethod
    def compile_model(model):
        if Config.USE_MIXED_PRECISION:
            try:
                from tensorflow.keras import mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print("✅ Mixed precision enabled for faster training")
            except:
                print("⚠️  Mixed precision not available, using default precision")

        optimizer = keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\nModel Summary:")
        model.summary()
        print(f"\n⚡ Total parameters: {model.count_params():,}")

        return model

# ============================================================================
# TRAINING PIPELINE (Unchanged)
# ============================================================================

class Trainer:
    @staticmethod
    def get_callbacks():
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,  # Slightly more patience for augmentation
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                Config.MODEL_SAVE_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks

    @staticmethod
    def train(model, x_train, y_train, x_test, y_test):
        print("\nStarting training...")
        print(f" Optimization settings:")
        print(f"   - Batch size: {Config.BATCH_SIZE}")
        print(f"   - Epochs: {Config.EPOCHS}")
        print(f"   - Mixed precision: {Config.USE_MIXED_PRECISION}")
        print(f"   - Data subset: {Config.USE_DATA_SUBSET}")
        print(f"   - Data Augmentation: Enabled")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   - GPU detected: {len(gpus)} device(s) ")
        else:
            print(f"   - Running on CPU (slower)")

        print("\n" + "="*70)

        start_time = time.time()
        history = model.fit(
            x_train, y_train,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS,
            validation_split=Config.VALIDATION_SPLIT,
            callbacks=Trainer.get_callbacks(),
            verbose=1
        )

        end_time = time.time()
        training_time = end_time - start_time

        print("\n" + "="*70)
        print(f"⏱️  Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print("="*70)

        return history

# ============================================================================
# EVALUATION AND VISUALIZATION (Updated for labels)
# ============================================================================

class Evaluator:
    @staticmethod
    def evaluate(model, x_test, y_test):
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy

    @staticmethod
    def plot_training_history(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(model, x_test, y_test):
        print("Plotting confusion matrix... (this may take a moment)")
        predictions = model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred)

        label_map = Config.get_label_map()
        tick_labels = None
        if label_map:
            all_labels = sorted(label_map.keys())
            tick_labels = [label_map.get(i, str(i)) for i in all_labels]

        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=tick_labels, yticklabels=tick_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def show_predictions(model, x_test, y_test, num_samples=10):
        predictions = model.predict(x_test[:num_samples])
        label_map = Config.get_label_map()
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        for i in range(num_samples):
            axes[i].imshow(x_test[i].squeeze(), cmap='gray')
            true_label_num = np.argmax(y_test[i])
            pred_label_num = np.argmax(predictions[i])
            confidence = predictions[i][pred_label_num]

            true_label = label_map.get(true_label_num, str(true_label_num))
            pred_label = label_map.get(pred_label_num, str(pred_label_num))

            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
                                color=color)
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

# ============================================================================
# INFERENCE (Updated for labels)
# ============================================================================

class Predictor:
    def __init__(self, model):
        self.model = model

    def preprocess_image(self, image_path):
        # This function is simple, for *dataset-like* images.
        # The complex logic is in ImageTester
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        if np.mean(img) > 127:
            img = 255 - img
        img = img.astype('float32') / 255.0
        img = img.reshape(1, Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.CHANNELS)
        return img

    def predict(self, image_path):
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img, verbose=0)
        label_map = Config.get_label_map()
        predicted_class_num = np.argmax(prediction)
        confidence = prediction[0][predicted_class_num]
        predicted_class = label_map.get(predicted_class_num, str(predicted_class_num))
        return predicted_class, confidence

    def predict_with_visualization(self, image_path):
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img, verbose=0)
        label_map = Config.get_label_map()
        predicted_class_num = np.argmax(prediction)
        confidence = prediction[0][predicted_class_num]
        top_5_idx = np.argsort(prediction[0])[-5:][::-1]
        top_5_probs = prediction[0][top_5_idx]

        predicted_class_char = label_map.get(predicted_class_num, str(predicted_class_num))
        top_5_chars = [label_map.get(idx, str(idx)) for idx in top_5_idx]

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'Input Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.barh(range(5), top_5_probs)
        plt.yticks(range(5), top_5_chars)
        plt.xlabel('Probability')
        plt.ylabel('Class')
        plt.title(f'Top 5 Predictions\nPredicted: {predicted_class_char} ({confidence:.2%})')
        plt.xlim([0, 1])
        plt.tight_layout()
        plt.show()
        return predicted_class_char, confidence

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("HANDWRITTEN CHARACTER RECOGNITION SYSTEM (ROBUST)")
    print(" (Balanced Dataset + Augmentation + COM Preprocessing)")
    print("="*70)

    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f" Configured {len(gpus)} GPU(s) for optimal performance\n")
    except Exception as e:
        print(f"  GPU configuration: {e}\n")

    if Config.DATASET == 'mnist':
        x_train, y_train, x_test, y_test = DataLoader.load_mnist()
    else:
        x_train, y_train, x_test, y_test = DataLoader.load_emnist()

    x_train, y_train, x_test, y_test = DataLoader.preprocess_data(
        x_train, y_train, x_test, y_test
    )

    model = HandwritingCNN.build_model()
    model = HandwritingCNN.compile_model(model)

    history = Trainer.train(model, x_train, y_train, x_test, y_test)

    test_loss, test_accuracy = Evaluator.evaluate(model, x_test, y_test)

    Evaluator.plot_training_history(history)
    Evaluator.show_predictions(model, x_test, y_test, num_samples=10)
    print("\n" + "="*70)
    print("🎉 ROBUST MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📊 Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"💾 Model saved to: {Config.MODEL_SAVE_PATH}")
    print("="*70)
    print("\n💡 To test your drawings, run:")
    print("   >>> upload_and_test_images(model)")
    print("="*70)

    return model

# ============================================================================
# INTERACTIVE TESTING (With Improved COM Preprocessing)
# ============================================================================

class ImageTester:
    def __init__(self, model):
        self.model = model
        self.predictor = Predictor(model)

    def preprocess_uploaded_image(self, image_path, show_preprocessing=True):
        img_original = cv2.imread(image_path)
        if img_original is None:
            raise ValueError(f"Could not load image from {image_path}")

        if len(img_original.shape) == 3:
            img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_original

        # Invert and threshold
        _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
             img_cropped = img_binary
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_binary.shape[1] - x, w + 2 * padding)
            h = min(img_binary.shape[0] - y, h + 2 * padding)
            img_cropped = img_binary[y:y+h, x:x+w]

        # --- NEW PREPROCESSING LOGIC ---

        # 1. Resize to fit in a 20x20 box
        target_h, target_w = 20, 20
        aspect_ratio = img_cropped.shape[1] / img_cropped.shape[0]

        if aspect_ratio > 1: # Wider
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else: # Taller
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        img_resized = cv2.resize(img_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Create final 28x28 canvas
        img_canvas = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # 3. Calculate Center of Mass (COM)
        M = cv2.moments(img_resized)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = new_w // 2, img_resized // 2

        # 4. Calculate offset to move COM to the center
        x_offset = (Config.IMG_WIDTH // 2) - cX
        y_offset = (Config.IMG_HEIGHT // 2) - cY

        # 5. Paste the image onto the canvas
        y1_paste = max(0, y_offset)
        y2_paste = min(Config.IMG_HEIGHT, y_offset + new_h)
        x1_paste = max(0, x_offset)
        x2_paste = min(Config.IMG_WIDTH, x_offset + new_w)
        y1_crop = max(0, -y_offset)
        y2_crop = min(new_h, Config.IMG_HEIGHT - y_offset)
        x1_crop = max(0, -x_offset)
        x2_crop = min(new_w, Config.IMG_WIDTH - x_offset)
        y_size = min(y2_paste - y1_paste, y2_crop - y1_crop)
        x_size = min(x2_paste - x1_paste, x2_crop - x1_crop)
        img_canvas[y1_paste:y1_paste+y_size, x1_paste:x1_paste+x_size] = \
            img_resized[y1_crop:y1_crop+y_size, x1_crop:x1_crop+x_size]
        img_final = img_canvas
        # --- END NEW LOGIC ---

        img_normalized = img_final.astype('float32') / 255.0
        img_model_input = img_normalized.reshape(1, Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.CHANNELS)

        if show_preprocessing:
            self._show_preprocessing_steps(img_original, img_gray, img_binary,
                                          img_cropped, img_final)

        return img_model_input, img_final

    def _show_preprocessing_steps(self, original, gray, binary, cropped, final):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('2. Grayscale')
        axes[0, 1].axis('off')
        axes[0, 2].imshow(binary, cmap='gray')
        axes[0, 2].set_title('3. Binary (Inverted)')
        axes[0, 2].axis('off')
        axes[1, 0].imshow(cropped, cmap='gray')
        axes[1, 0].set_title('4. Cropped to Character')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(final, cmap='gray')
        axes[1, 1].set_title('5. Resized & COM-Centered (28x28)')
        axes[1, 1].axis('off')
        axes[1, 2].text(0.5, 0.5, 'Ready for\nModel Input',
                        ha='center', va='center', fontsize=16, weight='bold')
        axes[1, 2].axis('off')
        plt.suptitle('Image Preprocessing Pipeline (Improved)', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.show()

    def test_single_image(self, image_path, show_preprocessing=True):
        print(f"\nTesting image: {image_path}")
        print("="*70)
        try:
            img_processed, img_display = self.preprocess_uploaded_image(
                image_path, show_preprocessing
            )
            prediction = self.model.predict(img_processed, verbose=0)
            label_map = Config.get_label_map()
            predicted_class_num = np.argmax(prediction)
            confidence = prediction[0][predicted_class_num]
            top_5_idx = np.argsort(prediction[0])[-5:][::-1]
            top_5_probs = prediction[0][top_5_idx]

            predicted_class_char = label_map.get(predicted_class_num, str(predicted_class_num))
            top_5_chars = [label_map.get(idx, str(idx)) for idx in top_5_idx]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(img_display, cmap='gray')
            axes[0].set_title(f'Processed Image (28x28)', fontsize=14, weight='bold')
            axes[0].axis('off')
            colors = ['green' if i == 0 else 'lightblue' for i in range(5)]
            bars = axes[1].barh(range(5), top_5_probs, color=colors)
            axes[1].set_yticks(range(5))
            axes[1].set_yticklabels(top_5_chars)
            axes[1].set_xlabel('Confidence', fontsize=12)
            axes[1].set_ylabel('Class', fontsize=12)
            axes[1].set_title('Top 5 Predictions', fontsize=14, weight='bold')
            axes[1].set_xlim([0, 1])
            for i, (bar, prob) in enumerate(zip(bars, top_5_probs)):
                axes[1].text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{prob:.1%}', va='center', fontsize=10)
            plt.tight_layout()
            plt.show()

            print(f"\n🎯 PREDICTION RESULTS:")
            print(f"   Predicted Class: {predicted_class_char} (raw index: {predicted_class_num})")
            print(f"   Confidence: {confidence:.2%}")
            print(f"\n📊 Top 5 Predictions:")
            for i, (char, prob) in enumerate(zip(top_5_chars, top_5_probs), 1):
                print(f"   {i}. Class {char}: {prob:.2%}")
            return predicted_class_char, confidence
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def test_multiple_images(self, image_paths):
        print("\n" + "="*70)
        print(f"TESTING {len(image_paths)} IMAGES")
        print("="*70)
        results = []
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {image_path}")
            predicted_class, confidence = self.test_single_image(
                image_path, show_preprocessing=False
            )
            results.append({
                'image': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
        print("\n" + "="*70)
        print("SUMMARY OF ALL PREDICTIONS")
        print("="*70)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['image']}")
            conf_str = f"({result['confidence']:.2%})" if result['confidence'] is not None else "(Error)"
            print(f"   → Predicted: {result['predicted_class']} {conf_str}")
        return results

# ============================================================================
# GOOGLE COLAB FILE UPLOAD INTERFACE
# ============================================================================

def upload_and_test_images(model):
    try:
        from google.colab import files
        import io

        print("="*70)
        print("UPLOAD YOUR HANDWRITTEN IMAGES (Digits, A-Z, a-z)")
        print("="*70)
        print("\nInstructions:")
        print("   1. Click 'Choose Files' button below")
        print("   2. Select your MS Paint drawings or photos")
        print("   3. You can select multiple files")
        print("\n" + "="*70 + "\n")

        uploaded = files.upload()
        if not uploaded:
            print(" No files uploaded!")
            return
        image_paths = []
        for filename, content in uploaded.items():
            with open(filename, 'wb') as f:
                f.write(content)
            image_paths.append(filename)
            print(f" Uploaded: {filename}")
        tester = ImageTester(model)
        if len(image_paths) == 1:
            tester.test_single_image(image_paths[0])
        else:
            tester.test_multiple_images(image_paths)
    except ImportError:
        print(" This function requires Google Colab environment")
        print(" For local testing, use: ImageTester(model).test_single_image('path/to/image.png')")


if __name__ == "__main__":
    # Train model
    model = main()

    print("\n\n" + "🎨"*35)
    print("ROBUST MODEL TRAINING COMPLETE! Ready for testing.")
    print("🎨"*35)

    # Automatically start the uploader
    upload_and_test_images(model)
