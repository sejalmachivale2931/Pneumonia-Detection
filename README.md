# 🩺 Pneumonia Detection using CNN

This project uses Deep Learning to detect Pneumonia from chest X-ray images using a Convolutional Neural Network (CNN).

The model classifies images into:
- Normal
- Pneumonia

---

## 🎯 Objective

Build an AI model that can automatically detect pneumonia from medical images.

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 📂 Dataset Structure

dataset/

train/
NORMAL/
PNEUMONIA/

val/
NORMAL/
PNEUMONIA/

New.jpg

---

## ⚙️ Workflow

1. Load dataset using ImageDataGenerator
2. Perform preprocessing (resize and normalize images)
3. Build CNN model
4. Train model
5. Evaluate performance
6. Predict new image

---

## 📊 Model Performance

Validation Accuracy: 96%

Model successfully detects pneumonia cases.

---

## ▶️ How to Run

Install libraries:

pip install -r requirements.txt

Run project:

python project.py

---

## 📁 Files

project.py → main code

requirements.txt → libraries

training_results.csv → training accuracy data

sample_output.png → accuracy graph

---

## 🚀 Future Scope

Improve accuracy using Transfer Learning models like VGG16 or ResNet.

---

## 👩‍💻 Author

Sejal Machivale
Aspiring Data Scientist
