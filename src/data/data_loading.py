import csv
import numpy as np
import tensorflow as tf


def load_csv(path):
    data = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append(row)
    return np.array(data)


def remap_label(label):
    label = int(label)

    # FER2013 original:
    # 0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral
    # Paper's 3 classes:
    # 0 happy, 1 unknown, 2 sad
    if label == 3:
        return 0   # happy
    elif label == 4:
        return 2   # sad
    else:
        return 1   # unknown


def prepare_data(data):
    X = data[:, :-1].astype(np.float32)
    y_raw = data[:, -1]

    y = np.array([remap_label(label) for label in y_raw], dtype=np.int32)

    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = tf.keras.utils.to_categorical(y, num_classes=3)

    return X, y


def load_training_data():
    data = load_csv("data/blendshapes_train.csv")
    np.random.shuffle(data)
    return prepare_data(data)


def load_validation():
    data = load_csv("data/blendshapes_val.csv")
    np.random.shuffle(data)
    return prepare_data(data)