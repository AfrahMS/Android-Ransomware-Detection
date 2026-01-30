# Android Ransomware Detection using Supervised ML Techniques

This repository contains the implementation and experimental setup related to the research paper:

**“Android Ransomware Detection Using Supervised Machine Learning Techniques Based on Traffic Analysis”**
published in *Sensors (MDPI), 2024*.

The project investigates the effectiveness of machine learning (ML) and deep learning (DL) models for detecting Android ransomware using network traffic analysis, addressing key challenges such as class imbalance, feature dimensionality, and robustness.

---

## 📌 Abstract

Android ransomware is one of the most disruptive forms of mobile malware, encrypting user data and demanding ransom payments.
This project applies supervised ML and DL techniques to classify Android network traffic as ransomware or benign, leveraging a large-scale, real-world dataset.

The study evaluates multiple ML, ensemble, and DL models under two experimental settings:

* Using all traffic features
* Using a reduced, optimized feature set

Performance is assessed using standard classification metrics to identify robust and efficient ransomware detection models.

---

## 🧠 Models Implemented

* Decision Tree (DT)
* Support Vector Machine (SVM)
* k-Nearest Neighbors (KNN)
* Ensemble Model (DT + SVM + KNN)
* Feedforward Neural Network (FNN)
* TabNet (Attention-based DL model for tabular data)

Each model is evaluated under:

* Full feature set
* Selected feature subset (top features)

---

## 🛠️ Technologies & Tools

* Python
* Scikit-learn
* TensorFlow / Keras
* PyTorch (TabNet)
* Kaggle Dataset
* Google Colab / Jupyter Notebook

---

## 📊 Dataset

* Dataset: Android Ransomware Detection (CIC, Kaggle)
* Our Dataset link: https://drive.google.com/file/d/1jVctpOJNAg2MJkuWHv-GtDC7i4AIWh3H/view?usp=drive_link
* Total Records: 392,035
* Classes: Benign traffic + 10 ransomware families
* Final Task: Binary classification (Ransomware vs. Benign)
* Features: 85 network traffic features
* Preprocessing Steps:

  * Label binarization
  * Random under-sampling to handle class imbalance
  * Categorical-to-numerical conversion
  * Feature selection (Forward Selection & Feature Importance)

---

## ⚙️ Experimental Setup

Two main experiments were conducted:

1. Experiment 1: Training using all available traffic features
2. Experiment 2: Training using the top selected features only

Dataset split:

* Training: 80%
* Testing: 20%

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score

---

## 📈 Key Findings

* Decision Tree achieved the best overall accuracy (97.24%)
* SVM achieved 100% recall, making it highly effective for ransomware detection
* Feature selection reduced computational complexity without degrading performance
* Ensemble and DL models demonstrated strong robustness on large-scale traffic data
* TabNet showed promising performance for structured traffic features
---

## 📚 Citation

If you use this work, please cite:

> Albin Ahmed, A., Shaahid, A., Alnasser, F., Alfaddagh, S., Binagag, S., & Alqahtani, D.
> *Android Ransomware Detection Using Supervised Machine Learning Techniques Based on Traffic Analysis.*
> Sensors, 2024.

---
