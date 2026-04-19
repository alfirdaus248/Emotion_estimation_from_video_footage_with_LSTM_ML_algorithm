import csv
import numpy as np
import keras


# ===== LOAD MODEL =====
model = keras.models.load_model("ckpt/epoch_40-val_loss_0.6317.keras")


# ===== LABEL REMAPPING =====
def remap_label(label):
    label = int(label)

    if label == 3:
        return 0   # happy
    elif label == 4:
        return 2   # sad
    else:
        return 1   # unknown


# ===== LOAD TEST DATA =====
X = []
y = []
indices = []

with open("data/blendshapes_test_indexed.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        features = row[:-2]   # all blendshapes
        label = row[-2]       # emotion
        index = row[-1]       # index

        X.append(features)
        y.append(remap_label(label))
        indices.append(index)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# one-hot
y = keras.utils.to_categorical(y, num_classes=3)


# ===== EVALUATION =====
loss, cce, acc, f1 = model.evaluate(X, y, verbose=2)

print("\n=== FINAL TEST RESULTS ===")
print(f"Loss: {loss:.4f}")
print(f"Categorical Crossentropy: {cce:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")