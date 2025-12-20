# 🎙️ Speaker Identification Using MFCC

## 📌 Project Overview

This project implements a **Speaker Identification system** using **Mel-Frequency Cepstral Coefficients (MFCC)** for feature extraction and **Machine Learning classification**.
The system is capable of identifying **which known speaker** is speaking from a given audio sample.

The project is developed as part of an **academic mini-project** for students of **Computing and Data Science (3rd Year)**.

---

## 🎯 Objectives

* Record and preprocess speech audio samples
* Extract MFCC features from speech signals
* Build a labeled dataset for multiple speakers
* Train a machine learning model for speaker identification
* Perform prediction on unseen audio samples

---

## 🧠 Core Concepts Used

* Digital Signal Processing (DSP)
* Speech Feature Extraction
* MFCC (Mel-Frequency Cepstral Coefficients)
* Supervised Machine Learning
* Classification Models

---

## 🛠️ Technologies & Libraries

* **Python 3**
* **Librosa** – Audio processing & MFCC extraction
* **SoundDevice** – Audio recording
* **NumPy** – Numerical operations
* **Pandas** – Dataset handling
* **Scikit-learn** – Machine learning models
* **Joblib** – Model persistence

---

## 📁 Project Structure

```
📦 Speaker-Identification
 ┣ 📜 SpeechRecognition.ipynb
 ┣ 📂 dataset/
 ┃ ┣ 📂 speaker_1/
 ┃ ┣ 📂 speaker_2/
 ┃ ┗ 📂 speaker_n/
 ┣ 📂 models/
 ┃ ┗ 📜 speaker_model.pkl
 ┣ 📜 README.md
```

---

## 🔊 Dataset Creation

* Audio samples are recorded using a microphone.
* Each speaker provides multiple voice samples.
* Samples are stored speaker-wise.
* MFCC features are extracted and averaged per sample.
* Labels are assigned based on speaker identity.

---

## ⚙️ Feature Extraction

* Audio is sampled at a fixed sampling rate.
* Noise-free mono audio is processed.
* **MFCC features** are extracted using Librosa.
* Mean MFCC vectors are used as feature inputs.

---

## 🤖 Model Training

* Extracted MFCC features form the feature matrix.
* Speaker labels are encoded numerically.
* A supervised classification model (e.g., **SVM**) is trained.
* The trained model is saved using `joblib`.

---

## 🔍 Prediction Workflow

1. Record a new speech sample
2. Extract MFCC features
3. Load trained model
4. Predict speaker label
5. Display identified speaker

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install numpy pandas librosa sounddevice scikit-learn joblib
```

### 2️⃣ Open Notebook

```bash
jupyter notebook SpeechRecognition.ipynb
```

### 3️⃣ Execute Cells Sequentially

* Dataset creation
* Feature extraction
* Model training
* Prediction

---

## 📊 Results

* Successfully identifies registered speakers
* Works best in low-noise environments
* Accuracy improves with more training samples

---

## 🚀 Future Enhancements

* Real-time continuous speaker recognition
* Noise-robust feature extraction
* Deep learning (CNN / LSTM)
* Speaker verification (authentication)
* GUI or web interface

---

## 📚 Applications

* Voice-based authentication systems
* Smart assistants
* Attendance systems
* Security systems
* Forensic voice analysis

---

## 📜 License

This project is intended for **educational purposes only**.

---
