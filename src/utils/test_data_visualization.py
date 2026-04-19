import csv
import matplotlib.pyplot as plt
import numpy as np


def remap_label(label):
    label = int(label)
    if label == 3:
        return 0
    elif label == 4:
        return 2
    else:
        return 1


def visualize_blendshapes(dataset_path):
    data = []

    with open(dataset_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            features = row[:-2]
            label = remap_label(row[-2])
            data.append(features + [label])

    data = np.array(data, dtype=np.float32)

    print(f"Rows: {data.shape[0]}")
    print(f"Cols: {data.shape[1]}")

    # split classes
    class_0 = data[data[:, -1] == 0]
    class_1 = data[data[:, -1] == 1]
    class_2 = data[data[:, -1] == 2]

    print("\nClass distribution:")
    print(f"Happy (0): {len(class_0)}")
    print(f"Unknown (1): {len(class_1)}")
    print(f"Sad (2): {len(class_2)}")

    num_blendshapes = data.shape[1] - 1

    plt.figure(figsize=(20, 20))

    for i in range(num_blendshapes):
        plt.subplot(6, 5, i + 1)

        plt.boxplot([
            class_0[:, i],
            class_1[:, i],
            class_2[:, i]
        ])

        plt.title(f"B{i}")
        plt.xticks([1, 2, 3], ["Happy", "Unknown", "Sad"])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_blendshapes("data/blendshapes_test_indexed.csv")