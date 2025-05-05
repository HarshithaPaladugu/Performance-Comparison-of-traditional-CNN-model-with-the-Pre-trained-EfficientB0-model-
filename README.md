# Performance-Comparison-of-traditional-CNN-model-with-the-Pre-trained-EfficientNetB0-model-
This projects aims to evaluate the performance of traditional CNN and pre-trained EfficientNetB0 model in detecting the health status of chili leaf 


# 🌿 Chilli Plant Leaf Disease Classification using CNN & EfficientNetB0

This repository contains the code and methodology for classifying chilli plant leaf health status using image-based deep learning models. The project compares the performance of a traditional Convolutional Neural Network (CNN) with a pretrained EfficientNetB0 model.

---

## 📌 Objective

To develop and evaluate deep learning models that can accurately classify chilli plant leaves into one of three categories:

* **Healthy**
* **Cercospora Leaf Spot**
* **Wilt Disease**

---

## 📂 Dataset

The dataset used for this study consists of **618 labelled chilli plant leaf images**. The images were divided into:

* **Training Set**
* **Validation Set**
* **Testing Set**

The dataset was preprocessed and augmented to improve generalization during model training.

---

## 🛠️ Methodology

Two models were trained and evaluated:

### 1️⃣ Traditional CNN Model

* 3 Convolutional layers
* MaxPooling and Dropout layers
* Flatten → Dense → Output layers
* Activation: ReLU (hidden), Softmax (output)
* Optimizer: Adam
* Loss: Categorical Crossentropy

### 2️⃣ Pretrained EfficientNetB0 Model

* EfficientNetB0 base loaded with pretrained ImageNet weights
* Custom dense head with Dropout and Softmax output
* Frozen base layers initially, later fine-tuned
* Same optimizer and loss function

---

## 📊 Evaluation Metrics

Model performance was compared using:

* Confusion Matrix
* Accuracy
* Precision
* Recall (Sensitivity)
* F1 Score
* ROC-AUC (optional if implemented)

---

## 📈 Results

| Model              | Accuracy   | Precision | Recall   | F1 Score |
| ------------------ | ---------- | --------- | -------- | -------- |
| CNN                | 61.29%     | Moderate  | Moderate | Moderate |
| **EfficientNetB0** | **95.16%** | **High**  | **High** | **High** |

The EfficientNetB0 model significantly outperformed the traditional CNN model across all metrics.

---





## 📝 Conclusion

This project demonstrates that transfer learning using EfficientNetB0 achieves far superior results compared to a standard CNN model when applied to a limited dataset of plant disease images. It highlights the importance of leveraging pretrained models in precision agriculture.

---

## 🙌 Acknowledgements

* Dataset collected and labelled manually
* EfficientNetB0 pretrained weights from TensorFlow/Keras


