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

class Config:
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    CHANNELS = 1

    BATCH_SIZE = 256
    EPOCHS = 5
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.1

    USE_MIXED_PRECISION = True
    USE_DATA_SUBSET = False
    SUBSET_SIZE = 10000

    DATASET = 'mnist'
    MODEL_SAVE_PATH = 'handwriting_recognition_model.h5'

    @staticmethod
    def get_num_classes():
        return 10 if Config.DATASET == 'mnist' else 47

class DataLoader:
    @staticmethod
    def load_mnist():
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        return x_train, y_train, x_test, y_test

    @staticmethod
    def load_emnist():
        print("Loading EMNIST dataset...")
        try:
            import tensorflow_datasets as tfds

            ds_train = tfds.load('emnist/balanced', split='train', as_supervised=True)
            ds_test = tfds.load('emnist/balanced', split='test', as_supervised=True)

            x_train, y_train = [], []
            x_test, y_test = [], []

            for image, label in ds_train:
                x_train.append(image.numpy())
                y_train.append(label.numpy())

            for image, label in ds_test:
                x_test.append(image.numpy())
                y_test.append(label.numpy())

            x_train = np.array(x_train).squeeze()
            y_train = np.array(y_train)
            x_test = np.array(x_test).squeeze()
            y_test = np.array(y_test)

            return x_train, y_train, x_test, y_test

        except Exception as e:
            print(f"Error loading EMNIST: {e}")
            print("Falling back to MNIST...")
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

class HandwritingCNN:
    @staticmethod
    def build_model():
        print("Building optimized CNN model for fast training...")

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu',
                          input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.CHANNELS),
                          padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

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

class Trainer:
    @staticmethod
    def get_callbacks():
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
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
        print(f"   - Batch size: {Config.BATCH_SIZE} (larger = faster)")
        print(f"   - Epochs: {Config.EPOCHS} (reduced for speed)")
        print(f"   - Mixed precision: {Config.USE_MIXED_PRECISION}")
        print(f"   - Data subset: {Config.USE_DATA_SUBSET}")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   - GPU detected: {len(gpus)} device(s) ")
        else:
            print(f"   - Running on CPU (slower)")

        print("\n" + "="*70)

        import time
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
        predictions = model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def show_predictions(model, x_test, y_test, num_samples=10):
        predictions = model.predict(x_test[:num_samples])

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i in range(num_samples):
            axes[i].imshow(x_test[i].squeeze(), cmap='gray')

            true_label = np.argmax(y_test[i])
            pred_label = np.argmax(predictions[i])
            confidence = predictions[i][pred_label]

            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
                                color=color)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

class Predictor:
    def __init__(self, model):
        self.model = model

    def preprocess_image(self, image_path):
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
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        return predicted_class, confidence

    def predict_with_visualization(self, image_path):
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'Input Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        top_5_idx = np.argsort(prediction[0])[-5:][::-1]
        top_5_probs = prediction[0][top_5_idx]
        plt.barh(range(5), top_5_probs)
        plt.yticks(range(5), top_5_idx)
        plt.xlabel('Probability')
        plt.ylabel('Class')
        plt.title(f'Top 5 Predictions\nPredicted: {predicted_class} ({confidence:.2%})')
        plt.xlim([0, 1])

        plt.tight_layout()
        plt.show()

        return predicted_class, confidence

def main():
    print("="*70)
    print("HANDWRITTEN CHARACTER RECOGNITION SYSTEM")
    print(" OPTIMIZED FOR FAST TRAINING (~2 minutes)")
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
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print(f" Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f" Model saved to: {Config.MODEL_SAVE_PATH}")
    print("="*70)
    print("\n To test your drawings, run:")
    print("   >>> quick_test(model)")
    print("="*70)

    return model

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

        _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_binary.shape[1] - x, w + 2 * padding)
            h = min(img_binary.shape[0] - y, h + 2 * padding)

            img_cropped = img_binary[y:y+h, x:x+w]
        else:
            img_cropped = img_binary

        aspect_ratio = img_cropped.shape[1] / img_cropped.shape[0]

        if aspect_ratio > 1:
            new_w = Config.IMG_WIDTH
            new_h = int(Config.IMG_HEIGHT / aspect_ratio)
        else:
            new_h = Config.IMG_HEIGHT
            new_w = int(Config.IMG_WIDTH * aspect_ratio)

        img_resized = cv2.resize(img_cropped, (new_w, new_h))

        img_final = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        y_offset = (Config.IMG_HEIGHT - new_h) // 2
        x_offset = (Config.IMG_WIDTH - new_w) // 2

        img_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

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
        axes[1, 1].set_title('5. Resized & Centered (28x28)')
        axes[1, 1].axis('off')

        axes[1, 2].text(0.5, 0.5, 'Ready for\nModel Input',
                        ha='center', va='center', fontsize=16, weight='bold')
        axes[1, 2].axis('off')

        plt.suptitle('Image Preprocessing Pipeline', fontsize=16, weight='bold')
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
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]

            top_5_idx = np.argsort(prediction[0])[-5:][::-1]
            top_5_probs = prediction[0][top_5_idx]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(img_display, cmap='gray')
            axes[0].set_title(f'Processed Image (28x28)', fontsize=14, weight='bold')
            axes[0].axis('off')

            colors = ['green' if i == 0 else 'lightblue' for i in range(5)]
            bars = axes[1].barh(range(5), top_5_probs, color=colors)
            axes[1].set_yticks(range(5))
            axes[1].set_yticklabels(top_5_idx)
            axes[1].set_xlabel('Confidence', fontsize=12)
            axes[1].set_ylabel('Class', fontsize=12)
            axes[1].set_title('Top 5 Predictions', fontsize=14, weight='bold')
            axes[1].set_xlim([0, 1])

            for i, (bar, prob) in enumerate(zip(bars, top_5_probs)):
                axes[1].text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{prob:.1%}', va='center', fontsize=10)

            plt.tight_layout()
            plt.show()

            print(f"\nPREDICTION RESULTS:")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print(f"\n Top 5 Predictions:")
            for i, (idx, prob) in enumerate(zip(top_5_idx, top_5_probs), 1):
                print(f"   {i}. Class {idx}: {prob:.2%}")

            return predicted_class, confidence

        except Exception as e:
            print(f" Error processing image: {e}")
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
            print(f"   → Predicted: {result['predicted_class']} ({result['confidence']:.2%})")

        return results

def upload_and_test_images(model):
    try:
        from google.colab import files
        import io

        print("="*70)
        print("UPLOAD YOUR HANDWRITTEN IMAGES")
        print("="*70)
        print("\nInstructions:")
        print("   1. Click 'Choose Files' button below")
        print("   2. Select your MS Paint drawings or photos")
        print("   3. You can select multiple files")
        print("   4. Wait for upload to complete")
        print("\n  Image Tips:")
        print("   - Draw digits/characters on WHITE background")
        print("   - Use BLACK pen/pencil")
        print("   - Make it clear and centered")
        print("   - Common formats: PNG, JPG, BMP")

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
    model = main()
    print("MODEL TRAINING COMPLETE! Ready for testing.")
    pradict=upload_and_test_images(model)