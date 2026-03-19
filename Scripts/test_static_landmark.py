import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
import sys
import os

# ================= CONFIG =================
PROJECT_ROOT = r"D:\ISL_PROJECT_FINAL"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "numbers_landmark_best.h5")
LABELS_PATH = os.path.join(PROJECT_ROOT, "landmarks", "class_labels.json")
IMG_SIZE = 256

CONFIDENCE_THRESHOLD = 0.85
AMBIGUITY_GAP = 0.15
# =========================================

# -------- Load model ----------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# -------- Load labels ----------
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

labels = {v: k for k, v in label_map.items()}
print("Class order:", labels)

# -------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -------- Image path ----------
if len(sys.argv) < 2:
    print("Usage: python test_static_landmark.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# -------- Read image ----------
img = cv2.imread(image_path)
if img is None:
    print("Image not found")
    sys.exit(1)

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------- Extract landmarks ----------
result = hands.process(img_rgb)

if not result.multi_hand_landmarks:
    print("No hand detected")
    sys.exit(1)

landmarks = result.multi_hand_landmarks[0]

row = []
for lm in landmarks.landmark:
    row.extend([lm.x, lm.y, lm.z])

X = np.array(row, dtype=np.float32).reshape(1, -1)

# -------- Prediction ----------
preds = model.predict(X)[0]

top5 = np.argsort(preds)[-5:][::-1]

print("\nTop-5 predictions:")
for i in top5:
    print(f"{labels[i]} : {preds[i]:.4f}")

# -------- Decision Logic ----------
top1 = top5[0]
top2 = top5[1]

confidence = preds[top1]
gap = preds[top1] - preds[top2]

print("\nDecision:")

if confidence < CONFIDENCE_THRESHOLD:
    print("Result: UNCERTAIN")
    print("Reason: Low confidence")
    print("Top guess:", labels[top1], f"{confidence:.3f}")

elif gap < AMBIGUITY_GAP:
    print("Result: AMBIGUOUS")
    print("Top-2 close predictions:")
    print(f"{labels[top1]} : {preds[top1]:.3f}")
    print(f"{labels[top2]} : {preds[top2]:.3f}")

else:
    print("Result: ACCEPTED")
    print("Predicted:", labels[top1])
    print("Confidence:", f"{confidence:.3f}")
