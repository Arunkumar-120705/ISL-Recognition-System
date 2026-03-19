import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
from collections import deque

# ================= CONFIG =================
PROJECT_ROOT = r"D:\ISL_PROJECT_FINAL"

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "alphabets_landmark_2hand_best.h5")
LABELS_PATH = os.path.join(PROJECT_ROOT, "landmarks", "alphabet_class_labels.json")

MEAN_PATH = os.path.join(PROJECT_ROOT, "models", "alphabets_2hand_mean.npy")
STD_PATH  = os.path.join(PROJECT_ROOT, "models", "alphabets_2hand_std.npy")

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 7
# =========================================

print("Loading alphabet landmark model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

labels = {v: k for k, v in label_map.items()}
print("Classes:", labels)

mean = np.load(MEAN_PATH)
std  = np.load(STD_PATH)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

pred_queue = deque(maxlen=SMOOTHING_WINDOW)

cap = cv2.VideoCapture(0)
print("\nPress 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    display_text = "No hand detected"

    if result.multi_hand_landmarks:
        landmarks_all = []

        # sort hands left → right for consistency
        hands_sorted = sorted(
            result.multi_hand_landmarks,
            key=lambda h: np.mean([lm.x for lm in h.landmark])
        )

        for hand_lms in hands_sorted[:2]:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            row = []
            for lm in hand_lms.landmark:
                row.extend([lm.x, lm.y, lm.z])
            landmarks_all.append(row)

        # pad if only one hand (C / L)
        if len(landmarks_all) == 1:
            landmarks_all.append([0.0] * 63)

        X = np.array(landmarks_all, dtype=np.float32).reshape(1, -1)

        # normalize
        X = (X - mean) / std

        preds = model.predict(X, verbose=0)[0]
        best_idx = np.argmax(preds)
        confidence = preds[best_idx]

        pred_queue.append(best_idx)

        if len(pred_queue) == SMOOTHING_WINDOW:
            final_idx = max(set(pred_queue), key=pred_queue.count)

            if confidence >= CONFIDENCE_THRESHOLD:
                display_text = f"{labels[final_idx]} ({confidence:.2f})"
            else:
                display_text = "Uncertain"

    cv2.rectangle(frame, (10, 10), (380, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        display_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ISL Alphabets – Two Hand Landmark Model", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
