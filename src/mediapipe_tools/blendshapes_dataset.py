"""Create blendshapes datasets from cleaned FER2013 CSV files.

This script reads the cleaned training, validation, and test CSV files,
converts each 48x48 grayscale image into an RGB image for MediaPipe,
extracts a selected subset of important blendshape scores, and writes
the results into new CSV files for model training.

Expected input CSV columns:
    emotion,pixels,Usage

Output CSV columns:
    blend_1, blend_2, ..., blend_49, emotion
    (only the selected blendshape indices listed below are included)
"""

import csv
import os
from typing import List, Tuple, Optional, Union

import mediapipe as mp
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from mediapipe_tools.visualizing_and_setup import detector

load_dotenv()

# Selected blendshape indices from the author's script / ablation choice
# These correspond to the subset of important blendshapes used to reduce feature count.
SELECTED_BLENDSHAPE_INDICES = [
    1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 25, 34, 35, 38, 44, 45,
    46, 47, 48, 49,
]

# Output filenames
OUTPUT_TRAIN = "data/blendshapes_train.csv"
OUTPUT_VAL = "data/blendshapes_val.csv"
OUTPUT_TEST = "data/blendshapes_test.csv"


def load_pixel_dataset(csv_path: str) -> List[List[str]]:
    """Load a cleaned FER2013-style CSV file.

    Returns rows in the form:
        [emotion, pixels, Usage]
    """
    rows: List[List[str]] = []

    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            if len(row) < 3:
                continue

            emotion = row[0].strip()
            pixels = row[1].strip()
            usage = row[2].strip()

            if not emotion or not pixels or not usage:
                continue

            rows.append([emotion, pixels, usage])

    return rows


def pixels_to_rgb_image(pixel_str: str) -> np.ndarray:
    """Convert FER2013 pixel string into a 48x48 RGB uint8 image."""
    pixel_values = pixel_str.split()

    if len(pixel_values) != 48 * 48:
        raise ValueError(f"Expected 2304 pixels, got {len(pixel_values)}")

    image = np.array(pixel_values, dtype=np.uint8).reshape(48, 48, 1)

    # Convert grayscale to RGB for MediaPipe
    image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    rgb_image = tf.image.grayscale_to_rgb(image_tensor).numpy()

    return rgb_image


def extract_selected_blendshapes(
    rgb_image: np.ndarray,
    face_landmarker_detector,
) -> Optional[List[float]]:
    """Extract selected blendshape scores from an RGB image.

    Returns:
        list of selected blendshape scores if detection succeeds,
        None otherwise.
    """
    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = face_landmarker_detector.detect(frame)

    if not detection_result.face_blendshapes:
        return None

    # Build a lookup from blendshape index -> score
    score_by_index = {}
    for category in detection_result.face_blendshapes[0]:
        score_by_index[category.index] = category.score

    # Preserve fixed feature order
    features = [float(score_by_index.get(idx, 0.0)) for idx in SELECTED_BLENDSHAPE_INDICES]
    return features


def create_blendshape_rows(
    dataset_rows: List[List[str]],
    dataset_name: str,
) -> Tuple[List[List[Union[float, int]]], int, int]:
    """Convert cleaned FER2013 rows into blendshape feature rows.

    Returns:
        feature_rows, kept_count, skipped_count
    """
    feature_rows: List[List[Union[float, int]]] = []
    kept = 0
    skipped = 0

    face_landmarker_detector = detector()

    for row in dataset_rows:
        try:
            emotion = int(row[0].strip())
            pixel_str = row[1].strip()

            rgb_image = pixels_to_rgb_image(pixel_str)
            features = extract_selected_blendshapes(rgb_image, face_landmarker_detector)

            if features is None:
                skipped += 1
                continue

            feature_rows.append(features + [emotion])
            kept += 1

        except Exception:
            skipped += 1
            continue

    print(f"{dataset_name}: kept={kept}, skipped={skipped}, total={kept + skipped}")
    return feature_rows, kept, skipped


def write_blendshape_csv(output_path: str, rows: List[List[Union[float, int]]]) -> None:
    """Write extracted blendshape features to CSV."""
    header = [f"blend_{idx}" for idx in SELECTED_BLENDSHAPE_INDICES] + ["emotion"]

    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def process_one_dataset(env_key: str, output_path: str, dataset_name: str) -> None:
    """Load, convert, and save one dataset split."""
    input_path = os.getenv(env_key)
    if not input_path:
        raise ValueError(f"Environment variable '{env_key}' is not set.")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"\nProcessing {dataset_name} from: {input_path}")
    rows = load_pixel_dataset(input_path)
    print(f"{dataset_name}: loaded {len(rows)} cleaned rows")

    blendshape_rows, kept, skipped = create_blendshape_rows(rows, dataset_name)
    write_blendshape_csv(output_path, blendshape_rows)

    print(f"{dataset_name}: saved to {output_path}")
    print(f"{dataset_name}: final rows written = {kept}")


if __name__ == "__main__":
    process_one_dataset("TRAIN_DATASET", OUTPUT_TRAIN, "TRAIN")
    process_one_dataset("VAL_DATASET", OUTPUT_VAL, "VAL")
    process_one_dataset("TEST_DATASET", OUTPUT_TEST, "TEST")

    print("\nDone creating blendshapes datasets.")