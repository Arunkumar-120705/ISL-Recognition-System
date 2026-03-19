import os
import cv2
import json
import numpy as np
import mediapipe as mp

# ================= CONFIG =================
PROJECT_ROOT = r"D:\ISL_PROJECT_FINAL"
IMAGE_DIR = os.path.join(PROJECT_ROOT, "static_numbers")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "landmarks")

IMG_SIZE = 256
MAX_HANDS = 1
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Initializing MediaPipe Hands...")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.7
)

X = []
y = []

# Read class folders
classes = sorted([
    d for d in os.listdir(IMAGE_DIR)
    if os.path.isdir(os.path.join(IMAGE_DIR, d))
])

label_map = {label: idx for idx, label in enumerate(classes)}

print("Classes found:", classes)

for label in classes:
    class_path = os.path.join(IMAGE_DIR, label)
    images = os.listdir(class_path)

    print(f"Processing '{label}' | Images: {len(images)}")

    for img_name in images:
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        row = []
        for lm in hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])

        X.append(row)
        y.append(label_map[label])

hands.close()

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

np.save(os.path.join(OUTPUT_DIR, "X_landmarks.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)

with open(os.path.join(OUTPUT_DIR, "class_labels.json"), "w") as f:
    json.dump(label_map, f, indent=4)

print("\nLandmark extraction completed")
print("X shape:", X.shape)   # (samples, 63)
print("y shape:", y.shape)
print("Saved to:", OUTPUT_DIR)
