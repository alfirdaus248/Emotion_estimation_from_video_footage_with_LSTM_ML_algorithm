Here is your **final clean README** with everything included properly and no formatting issues:

---

# 🧠 Emotion Estimation from Video Footage with LSTM (Reconstruction)

## 📌 Overview

This project reconstructs and analyzes the research paper:

**Emotion estimation from video footage with LSTM**
**Author:** Samer Attrah
Paper: [https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1678984/full](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1678984/full)

Original repository:
[https://github.com/Samir-atra/Emotion_estimation_from_video_footage_with_LSTM_ML_algorithm/tree/main](https://github.com/Samir-atra/Emotion_estimation_from_video_footage_with_LSTM_ML_algorithm/tree/main)

The system performs facial emotion recognition using:

* MediaPipe blendshape features
* LSTM-based classification
* Real-time webcam inference

This implementation reproduces the original pipeline and extends it with improved preprocessing, evaluation, hyperparameter tuning, and deployment.

---

## 🎯 Objectives

* Reproduce the original methodology
* Validate reproducibility of results
* Analyze model behavior and limitations
* Demonstrate real-time emotion recognition
* Improve performance via hyperparameter tuning

---

## 🚀 Features

* Dataset preprocessing and balancing (FER2013)
* Blendshape extraction (27 selected features)
* LSTM-based emotion classification
* Hyperparameter tuning with Keras Tuner
* Evaluation and error analysis
* Feature visualization
* Real-time webcam demo

---

## 📊 Results

| Metric   | Paper   | This Project |
| -------- | ------- | ------------ |
| Accuracy | ~71.99% | **74.81%**   |
| F1-score | ~0.63   | **0.66**     |

---

## 🧰 Recommended Environment (IMPORTANT)

This project is designed for:

* Python **3.9**
* Conda environment

### Create environment

```bash
conda create -n blendfer python=3.9
conda activate blendfer
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 📂 Dataset

FER2013 dataset:
[https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)

⚠️ Not included in this repository due to size.

---

## 🧠 Model Details

* Input: 27 blendshape features
* Model: LSTM
* Classes:

  * 0 → Happy
  * 1 → Unknown
  * 2 → Sad

---

## 🎥 Real-Time Demo

Run:

```bash
python src/demo/webcam_demo.py
```

Output classes:

* Happy
* Unknown
* Sad

Works immediately after installation (no `.env` required).

---

### Full Pipeline (Requires `.env`)

Create `.env` in the root directory:

```env
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

Ensure:

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

## ▶️ Usage

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

## 🔁 Full Pipeline (If you want to re-train the model)

```bash
python src/data/data_cleaning.py
python src/data/augmenting_and_normalizing.py
python src/mediapipe_tools/blendshapes_dataset.py
python src/data/dataset_indexing.py
python src/model/model_training.py
```

---

## ⚙️ Hyperparameter Tuning (Keras Tuner)

Run:

```bash
python src/model/keras_tuner_experimenter.py
```

Results are saved in:

```
keras_tuner_experiments/
```

---

## ⭐ IMPORTANT: Retrain After Tuning

Keras Tuner does NOT produce the final model.

You MUST:

1. Check best hyperparameters from tuning output
2. Apply them manually in:

```
src/model/model_training.py
```

3. Retrain:

```bash
python src/model/model_training.py
```

If you skip this step:

* you are NOT using optimized parameters
* your results will not reflect tuning

---

## 📉 Key Findings

* Dataset imbalance causes model collapse
* Balancing improves performance
* Blendshape features overlap heavily
* "Sad" is hardest to classify
* LSTM captures feature relationships

---

## ⚠️ Limitations

* Weak performance on Sad
* Feature overlap
* Sensitive to lighting

---

## 🚀 Future Work

* Improve minority class detection
* Use multimodal inputs
* Explore different architectures

---

## 🙏 Acknowledgment

This project is a reconstruction of:

**Emotion estimation from video footage with LSTM**
Samer Attrah

Original implementation:
[https://github.com/Samir-atra/Emotion_estimation_from_video_footage_with_LSTM_ML_algorithm/tree/main](https://github.com/Samir-atra/Emotion_estimation_from_video_footage_with_LSTM_ML_algorithm/tree/main)

---

## 👨‍💻 Author

M. Hisyam Al Firdaus

---