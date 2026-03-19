import os
import cv2
import json
import numpy as np
import mediapipe as mp

# ================= CONFIG =================
PROJECT_ROOT = r"D:\ISL_PROJECT_FINAL"
IMAGE_DIR = os.path.join(PROJECT_ROOT, "alphabet_images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "landmarks")

IMG_SIZE = 256
MAX_HANDS = 2
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.6
)

X, y = [], []

classes = sorted(os.listdir(IMAGE_DIR))
label_map = {label: idx for idx, label in enumerate(classes)}

print("====================================")
print("Alphabet Classes:", classes)
print("Total Classes:", len(classes))
print("====================================")

total_images = 0
processed_images = 0
skipped_images = 0

for label in classes:
    folder = os.path.join(IMAGE_DIR, label)
    images = os.listdir(folder)

    print(f"\nProcessing class '{label}' | Images: {len(images)}")

    for idx, img_name in enumerate(images):
        total_images += 1
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            skipped_images += 1
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            skipped_images += 1
            continue

        # -------- Extract landmarks ----------
        row = []

        detected_hands = result.multi_hand_landmarks

        for h in range(MAX_HANDS):
            if h < len(detected_hands):
                for lm in detected_hands[h].landmark:
                    row.extend([lm.x, lm.y, lm.z])
            else:
                # Pad missing hand with zeros (63 values)
                row.extend([0.0] * 63)

        # Safety check
        if len(row) != 126:
            skipped_images += 1
            continue

        X.append(row)
        y.append(label_map[label])
        processed_images += 1

        # ---- Progress print every 50 images ----
        if idx % 50 == 0:
            print(
                f"  [{label}] Processed {idx}/{len(images)} | "
                f"Total OK: {processed_images}"
            )

hands.close()

# -------- Convert & Save ----------
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

np.save(os.path.join(OUTPUT_DIR, "X_alphabets_2hand.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_alphabets_2hand.npy"), y)

with open(os.path.join(OUTPUT_DIR, "alphabet_class_labels_2hand.json"), "w") as f:
    json.dump(label_map, f, indent=4)

print("\n====================================")
print("EXTRACTION COMPLETE")
print("Total images seen     :", total_images)
print("Successfully processed:", processed_images)
print("Skipped images        :", skipped_images)
print("Final X shape         :", X.shape)
print("Final y shape         :", y.shape)
print("Saved as:")
print("  X_alphabets_2hand.npy")
print("  y_alphabets_2hand.npy")
print("  alphabet_class_labels_2hand.json")
print("====================================")
