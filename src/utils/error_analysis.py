import numpy as np
import keras
import csv
from sklearn.metrics import classification_report, confusion_matrix


# ===== LOAD MODEL =====
model = keras.models.load_model("ckpt/epoch_40-val_loss_0.6317.keras")


# ===== LABEL REMAP =====
def remap_label(label):
    label = int(label)
    if label == 3:
        return 0
    elif label == 4:
        return 2
    else:
        return 1


# ===== LOAD TEST DATA =====
X = []
y_true = []

with open("data/blendshapes_test_indexed.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        X.append(row[:-2])
        y_true.append(remap_label(row[-2]))

X = np.array(X, dtype=np.float32)
X = X.reshape((X.shape[0], X.shape[1], 1))
y_true = np.array(y_true)


# ===== PREDICTIONS =====
y_pred = model.predict(X)
y_pred = np.argmax(y_pred, axis=1)


# ===== ANALYSIS =====
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_true, y_pred))