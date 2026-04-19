"""Module for cleaning image datasets by filtering out images not detectable by MediaPipe.

This module provides functions to:
- Split a full dataset into training, validation, and test sets.
- Clean datasets by removing images where MediaPipe's face detector cannot find blendshapes.
- Write the cleaned dataset splits into new CSV files.
"""

import os
import tensorflow as tf
import mediapipe as mp
from dotenv import load_dotenv
import csv
from mediapipe_tools.visualizing_and_setup import detector
from utils.csv_writer import csv_writer

load_dotenv()


def list_creator(full_data_set):
    """
    Splits a comprehensive dataset into training, validation, and test sets
    based on a 'Usage' field within each data instance.

    Args:
        full_data_set (list): A list of data instances, where each instance is expected
                              to be a list or array-like object. The element at index `[2]`
                              is assumed to contain the usage type (e.g., "Training",
                              "PublicTest", "PrivateTest").

    Returns:
        tuple: A tuple containing three lists:
            - training_set_list (list): Instances marked as "Training".
            - validation_set_list (list): Instances marked as "PublicTest".
            - test_set_list (list): Instances marked as "PrivateTest".
    """
    # Initialize empty lists for the dataset splits
    training_set_list = []
    validation_set_list = []
    test_set_list = []
    
    # Iterate through the full dataset and categorize instances into respective lists
    for i in full_data_set:
        if i[2] == "Training":                # Assign to training split
            training_set_list.append(i)
        elif i[2] == "PublicTest":            # Assign to validation split
            validation_set_list.append(i)
        elif i[2] == "PrivateTest":           # Assign to test split
            test_set_list.append(i)

    return training_set_list, validation_set_list, test_set_list


def sets_cleaner(data_set_path):
    """
    Cleans a dataset by filtering out images that are not recognizable by MediaPipe's
    face detector (i.e., no blendshapes are detected).

    Args:
        data_set_path (str): The environment variable key (e.g., "TRAIN_DATASET")
                             that points to the path of the dataset CSV file.

    Returns:
        list: A list of cleaned data instances.
    """
    data_set = []
    kept = 0
    skipped = 0

    # Read the dataset file using TensorFlow's file I/O
    data = tf.io.read_file(os.getenv(data_set_path))
    f = tf.strings.split(data, sep="\n")

    # Create detector once, not inside the loop
    face_landmarker_detector = detector()

    # Loop through each line in the dataset, skipping header and last empty line
    for lines in f[1:-1]:
        line_str = tf.strings.as_string(lines).numpy().decode("utf-8")
        parts = line_str.split(",")

        # Safety check in case a row is malformed
        if len(parts) < 3:
            skipped += 1
            continue

        image_pixels_str = parts[1].split(" ")

        # Convert pixel strings to tensor, reshape to 48x48x1, then grayscale -> RGB
        image_tensor = tf.convert_to_tensor(image_pixels_str)
        image_tensor = tf.make_tensor_proto(image_tensor, dtype=tf.uint8)
        image_array = tf.make_ndarray(image_tensor).reshape(48, 48, 1)
        image_tensor_uint8 = tf.convert_to_tensor(image_array, dtype=tf.uint8)
        rgb_image = tf.image.grayscale_to_rgb(image_tensor_uint8).numpy()

        # Create MediaPipe Image object
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect face blendshapes using MediaPipe detector
        detection_result = face_landmarker_detector.detect(frame)

        if detection_result.face_blendshapes == []:
            skipped += 1
            continue
        else:
            clean_row = [parts[0].strip(), parts[1].strip(), parts[2].strip()]
            data_set.append(clean_row)
            kept += 1

    print(f"{data_set_path}: kept={kept}, skipped={skipped}, total={kept + skipped}")
    return data_set


def write_files(training_data, validation_data, test_data):
    """
    Writes the cleaned training, validation, and test datasets to separate CSV files.

    Args:
        training_data (list): The cleaned training set instances.
        validation_data (list): The cleaned validation set instances.
        test_data (list): The cleaned test set instances.
    """
    fields = ["emotion", "pixels", "Usage"]
    csv_writer("data/training_set_full.csv", fields, training_data)
    csv_writer("data/validation_set_full.csv", fields, validation_data)
    csv_writer("data/test_set_full.csv", fields, test_data)


if __name__ == "__main__":
    # Build the balanced dataset directly from the original FER2013 CSV
   # Read the original FER2013 CSV directly
    fullset = []
    with open(os.getenv("FER2013_DATASET_PATH"), mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            fullset.append(row)

    # Split into train / val / test based on the Usage column
    training_set, validation_set, test_set = list_creator(fullset)

    # Save split files first so sets_cleaner() can read them
    fields = ["emotion", "pixels", "Usage"]
    csv_writer("data/training_set_full.csv", fields, training_set)
    csv_writer("data/validation_set_full.csv", fields, validation_set)
    csv_writer("data/test_set_full.csv", fields, test_set)

    # Then clean them with MediaPipe
    training_set_hus = sets_cleaner("TRAIN_DATASET")
    validation_set_hus = sets_cleaner("VAL_DATASET")
    test_set_hus = sets_cleaner("TEST_DATASET")

    # Overwrite with cleaned versions
    write_files(training_set_hus, validation_set_hus, test_set_hus)