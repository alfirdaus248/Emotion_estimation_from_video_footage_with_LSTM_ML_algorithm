# 🧠 Emotion Estimation from Video Footage with LSTM (Reconstruction)

## 📌 Overview

This project reconstructs and analyzes the research paper:

**Emotion estimation from video footage with LSTM**
**Author:** Samer Attrah
🔗 [https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1678984/full](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1678984/full)

The system performs **facial emotion recognition** using:

* MediaPipe **blendshape features**
* LSTM-based classification
* Real-time webcam inference

This implementation reproduces the original pipeline and extends it with improved preprocessing, evaluation, and deployment.

---

## 🎯 Objectives

* Reproduce the original methodology
* Validate reproducibility of results
* Analyze model behavior and limitations
* Demonstrate real-time emotion recognition

---

## 🚀 Features

* 📊 Dataset preprocessing & balancing (FER2013)
* 🎭 Blendshape extraction (27 selected features)
* 🧠 LSTM-based emotion classification
* 📉 Evaluation & error analysis
* 📊 Feature visualization
* 🎥 Real-time webcam demo (plug-and-play)

---

## 📊 Results

| Metric   | Paper   | This Project |
| -------- | ------- | ------------ |
| Accuracy | ~71.99% | **74.81%**   |
| F1-score | ~0.63   | **0.66**     |

The reconstructed model achieves **comparable and slightly improved performance**.

---

## 🎥 Real-Time Demo

Run the webcam demo:

```
python src/demo/webcam_demo.py
```

### Output Classes

* Happy
* Unknown
* Sad

✔ Works immediately after installation (**no `.env` required**)

---

## 🛠️ Installation

```
pip install -r requirements.txt
```

---

## 📂 Dataset

This project uses the FER2013 dataset:

🔗 [https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)

⚠️ Dataset is not included due to size.

---

## 🧠 Model Details

* Input: 27 blendshape features
* Architecture: LSTM
* Classes:

  * 0 → Happy
  * 1 → Unknown
  * 2 → Sad

---

# ⚙️ Environment Configuration

## 🟢 Quick Demo

No configuration required:

```
python src/demo/webcam_demo.py
```

---

## 🔵 Full Pipeline (Training & Preprocessing)

To reproduce the full pipeline, a `.env` file is required.

### 📄 Create `.env`

Copy the example file:

```
cp .env.example .env
```

Or manually create `.env` and paste the following:

```
# ==============================
# DATASET PATH
# ==============================

FER2013_DATASET_PATH="path/to/fer2013.csv"

# ==============================
# PROCESSED DATA FILES
# ==============================

TRAIN_DATASET="data/training_set_full.csv"
VAL_DATASET="data/validation_set_full.csv"
TEST_DATASET="data/test_set_full.csv"
FULL_TEST_SET="data/test_set_full_index.csv"

# ==============================
# MEDIAPIPE MODEL
# ==============================

FACE_LANDMARKER="models/face_landmarker_v2_with_blendshapes.task"

# ==============================
# TRAINED MODEL
# ==============================

SAVED_MODEL_PATH="ckpt/epoch_40-val_loss_0.6317.keras"

# ==============================
# KERAS TUNER OUTPUT
# ==============================

KERAS_TUNER_EXPERIMENTS_DIR="keras_tuner_experiments"
```

---

### 📁 Required Files

Ensure the following files exist:

```
models/face_landmarker_v2_with_blendshapes.task
ckpt/epoch_40-val_loss_0.6317.keras
```

---

## 📁 Project Structure

```
project/
├── src/
├── ckpt/
│   └── epoch_40-val_loss_0.6317.keras
├── models/
│   └── face_landmarker_v2_with_blendshapes.task
├── data/
├── .env.example
├── requirements.txt
├── README.md
```

---

# ▶️ Usage

## Evaluate Model

```
python src/model/evaluation.py
```

---

## Error Analysis

```
python src/utils/error_analysis.py
```

---

## Feature Visualization

```
python src/utils/test_data_visualization.py
```

---

# 🔁 Retraining the Model

## ⚠️ Before Retraining

Delete old checkpoints:

**Windows (CMD):**

```
del /q ckpt\*
```

**PowerShell:**

```
Remove-Item ckpt\* -Force
```

**Linux / macOS:**

```
rm -rf ckpt/*
```

---

## 🔄 Run Full Pipeline

```
python src/data/data_cleaning.py
python src/data/augmenting_and_normalizing.py
python src/mediapipe_tools/blendshapes_dataset.py
python src/data/dataset_indexing.py
python src/model/model_training.py
```

---

## 🔍 Optional Analysis

```
python src/model/evaluation.py
python src/utils/error_analysis.py
python src/utils/test_data_visualization.py
```

---

# 📉 Key Findings

* Dataset imbalance can cause model collapse
* Balancing significantly improves performance
* Blendshape features overlap across classes
* "Sad" is the hardest emotion to classify
* LSTM captures complex feature relationships

---

# ⚠️ Limitations

* Lower performance on "Sad" class
* Feature overlap reduces separability
* Sensitive to lighting and face detection

---

# 🚀 Future Work

* Improve minority class detection
* Use multimodal inputs (audio + video)
* Explore alternative architectures
* Improve robustness in real-world scenarios

---

# 🙏 Acknowledgment

This project is a **reconstruction and extension** of:

**Emotion estimation from video footage with LSTM**
**Samer Attrah**

Original implementation:
🔗 [https://github.com/Samir-atra/Emotion_estimation_from_video_footage_with_LSTM_ML_algorithm/tree/main](https://github.com/Samir-atra/Emotion_estimation_from_video_footage_with_LSTM_ML_algorithm/tree/main)

All core methodology and concepts belong to the original author.

---

# 👨‍💻 Author

M. Hisyam Al Firdaus
UIN Maulana Malik Ibrahim Malang

---

# 🎯 Summary

Reproducible ✔
Improved ✔
Real-time ✔
Deployable ✔
