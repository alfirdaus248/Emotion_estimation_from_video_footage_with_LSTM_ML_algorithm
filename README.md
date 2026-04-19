# 🧠 Emotion Estimation from Video Footage with LSTM (Reconstruction)

## 📌 Overview

This project reconstructs and analyzes the research paper:

**Emotion estimation from video footage with LSTM**
**Author:** Samer Attrah
🔗 [https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1678984/full](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1678984/full)

Original repository:
🔗 [https://github.com/Samir-atra/Emotion_estimation_from_video_footage_with_LSTM_ML_algorithm/tree/main](https://github.com/Samir-atra/Emotion_estimation_from_video_footage_with_LSTM_ML_algorithm/tree/main)

The system performs **facial emotion recognition** using:

* MediaPipe blendshape features
* LSTM-based classification
* Real-time webcam inference

This implementation reproduces the original pipeline and extends it with improved preprocessing, evaluation, hyperparameter tuning, and deployment.

---

## 🎯 Objectives

* Reproduce the original methodology
* Validate reproducibility of results
* Analyze model behavior and limitations
* Demonstrate real-time inference
* Improve performance via hyperparameter tuning

---

## 🚀 Features

* Dataset preprocessing & balancing (FER2013)
* Blendshape extraction (27 selected features)
* LSTM-based emotion classification
* Hyperparameter tuning with Keras Tuner
* Evaluation & error analysis
* Feature visualization
* Real-time webcam demo

---

## 📊 Results

| Metric   | Paper   | This Project |
| -------- | ------- | ------------ |
| Accuracy | ~71.99% | **74.81%**   |
| F1-score | ~0.63   | **0.66**     |

---

## 🧰 Recommended Environment

This project is designed for:

* Python **3.9**
* Conda environment

### Setup

```bash
conda create -n blendfer python=3.9
conda activate blendfer
pip install -r requirements.txt
```

---


## 📂 Dataset

FER2013 dataset:
🔗 [https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)

⚠️ Not included in this repository due to size.

---

## 🎥 Real-Time Demo (Quick Start)

```bash
python src/demo/webcam_demo.py
```

Output:

* Happy
* Unknown
* Sad

✔ No `.env` required

---

## ⚙️ Full Pipeline Configuration

To run preprocessing, training, or tuning, create `.env`:

```
FER2013_DATASET_PATH="path/to/fer2013.csv"

TRAIN_DATASET="data/training_set_full.csv"
VAL_DATASET="data/validation_set_full.csv"
TEST_DATASET="data/test_set_full.csv"
FULL_TEST_SET="data/test_set_full_index.csv"

FACE_LANDMARKER="models/face_landmarker_v2_with_blendshapes.task"

SAVED_MODEL_PATH="ckpt/epoch_40-val_loss_0.6317.keras"

KERAS_TUNER_EXPERIMENTS_DIR="keras_tuner_experiments"
```

---

## 📁 Required Files

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
│   └── best_model.keras
├── models/
│   └── face_landmarker_v2_with_blendshapes.task
├── data/
├── .env.example
├── requirements.txt
├── README.md
```

---

# 🔁 Workflow Overview

```
Data → Preprocessing → Blendshape Extraction
     → (Recommended) Hyperparameter Tuning
     → Final Model Training → Evaluation
```

---

# ⚙️ Recommended Workflow (Important)

There are two ways to use this project:

---

## 🟢 Option 1 — Quick Training (No Tuning)

If you just want a working model quickly:

```bash
python src/data/data_cleaning.py
python src/data/augmenting_and_normalizing.py
python src/mediapipe_tools/blendshapes_dataset.py
python src/data/dataset_indexing.py
python src/model/model_training.py
```

✔ Faster
❌ Not optimized

---

## 🔵 Option 2 — Recommended (With Hyperparameter Tuning)

This gives better performance.

---

### Step 1 — Prepare Data

```bash
python src/data/data_cleaning.py
python src/data/augmenting_and_normalizing.py
python src/mediapipe_tools/blendshapes_dataset.py
python src/data/dataset_indexing.py
```

---

### Step 2 — Run Hyperparameter Tuning

```bash
python src/model/keras_tuner_experimenter.py
```

Results saved in:

```
keras_tuner_experiments/
```

---

### Step 3 — Apply Best Parameters

Open tuning results and update:

```
src/model/model_training.py
```

---

### Step 4 — Train Final Model

```bash
python src/model/model_training.py
```

---

## ⭐ Important Notes

* Keras Tuner performs multiple training runs internally
* Running training before tuning is unnecessary
* Final training is required after tuning

---

## ⏱️ Time Consideration

| Step                 | Time          |
| -------------------- | ------------- |
| Training (no tuning) | ~85 minutes   |
| Tuning               | ~7 hours      |
| Final training       | ~85 minutes   |

---

## ▶️ Additional Usage

### Evaluate model

```bash
python src/model/evaluation.py
```

### Error analysis

```bash
python src/utils/error_analysis.py
```

### Feature visualization

```bash
python src/utils/test_data_visualization.py
```

---

## 📉 Key Findings

* Dataset imbalance can cause model collapse
* Balancing significantly improves performance
* Blendshape features overlap across classes
* "Sad" is hardest to classify
* LSTM captures feature relationships effectively

---

## ⚠️ Limitations

* Lower performance on "Sad"
* Feature overlap reduces separability
* Sensitive to lighting

---

## 🚀 Future Work

* Improve minority class detection
* Use multimodal inputs
* Explore alternative architectures

---

## 🙏 Acknowledgment

This project is a reconstruction of:

**Emotion estimation from video footage with LSTM**
Samer Attrah

Original implementation:
🔗 [https://github.com/Samir-atra/Emotion_estimation_from_video_footage_with_LSTM_ML_algorithm/tree/main](https://github.com/Samir-atra/Emotion_estimation_from_video_footage_with_LSTM_ML_algorithm/tree/main)

---

## 👨‍💻 Author

M. Hisyam Al Firdaus

---