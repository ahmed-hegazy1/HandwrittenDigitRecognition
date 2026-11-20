# Handwritten Digit Recognition (CNN)
**CodeAlpha Machine Learning Internship – Task 3**

A deep learning project that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Objective
Identify handwritten digits using image processing and deep learning, as described in the CodeAlpha Task 3 brief. *(Note: This implementation focuses on digits using MNIST and can be extended to alphabetic characters using EMNIST).*

## Dataset
- **Source:** MNIST Handwritten Digits (`tensorflow.keras.datasets.mnist`)
- **Size:** 60,000 training images and 10,000 test images (28×28 grayscale)
- **Preprocessing:** Pixel values normalized to `[0, 1]` and reshaped to `(28, 28, 1)` for CNN input.

## Model Architecture
- `Conv2D(32, (3,3), ReLU)` → `MaxPooling2D(2,2)`
- `Conv2D(64, (3,3), ReLU)` → `MaxPooling2D(2,2)`
- `Flatten` → `Dense(64, ReLU)` → `Dropout(0.3)` → `Dense(10, Softmax)`
- **Training:** 5 epochs, batch size 128, Adam optimizer, `sparse_categorical_crossentropy` loss, 10% validation split.

## Results
- **Test Accuracy:** **98.91%** on 10,000 test images
- **Macro Avg F1-Score:** **0.99** across all 10 digit classes (0–9)
- Includes training/validation accuracy & loss curves, confusion matrix, sample predictions, and an error analysis of misclassified digits.

## Technologies
Python, TensorFlow/Keras, NumPy, Matplotlib, Seaborn, Scikit-learn, Google Colab

## How to Run
1. Open `Handwritten_Digit_Recognition.ipynb` in Google Colab.
2. Run all cells (`Runtime → Run all`). The MNIST dataset downloads automatically.
