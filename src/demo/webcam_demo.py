import os
import sys
import time
import cv2
import numpy as np
import keras
import mediapipe as mp

# Allow imports from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mediapipe_tools.visualizing_and_setup import detector

MODEL_PATH = "ckpt/epoch_40-val_loss_0.6317.keras"

# Same 27 selected blendshape indices used in your dataset pipeline
SELECTED_BLENDSHAPE_INDICES = [
    1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 25, 34, 35, 38, 44, 45,
    46, 47, 48, 49,
]

LABELS = {
    0: "Happy",
    1: "Unknown",
    2: "Sad",
}


def extract_selected_blendshapes_from_bgr(frame_bgr, face_landmarker_detector):
    """Extract the 27 selected blendshapes from a webcam frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    detection_result = face_landmarker_detector.detect(mp_image)

    if not detection_result.face_blendshapes:
        return None

    score_by_index = {}
    for category in detection_result.face_blendshapes[0]:
        score_by_index[category.index] = category.score

    features = [float(score_by_index.get(idx, 0.0)) for idx in SELECTED_BLENDSHAPE_INDICES]
    return np.array(features, dtype=np.float32)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    model = keras.models.load_model(MODEL_PATH)
    face_landmarker_detector = detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    prev_time = time.time()
    fps = 0.0
    last_prediction = "No face"
    last_confidence = 0.0

    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        features = extract_selected_blendshapes_from_bgr(frame, face_landmarker_detector)

        if features is not None:
            x = features.reshape(1, len(SELECTED_BLENDSHAPE_INDICES), 1)
            y_pred = model.predict(x, verbose=0)[0]
            pred_class = int(np.argmax(y_pred))
            confidence = float(np.max(y_pred))

            last_prediction = LABELS[pred_class]
            last_confidence = confidence
        else:
            last_prediction = "No face"
            last_confidence = 0.0

        current_time = time.time()
        delta = current_time - prev_time
        prev_time = current_time
        if delta > 0:
            fps = 1.0 / delta

        cv2.putText(
            frame,
            f"Prediction: {last_prediction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Confidence: {last_confidence:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("BlendFER-Lite Webcam Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()