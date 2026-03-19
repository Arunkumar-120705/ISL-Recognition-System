import os
import cv2
import json
import numpy as np
import mediapipe as mp

# ================= CONFIG =================
PROJECT_ROOT = r"D:\ISL_PROJECT_FINAL"
VIDEO_DIR = os.path.join(PROJECT_ROOT, "Dynamic")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "landmarks")

FRAMES_PER_VIDEO = 30
IMG_SIZE = 256
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nInitializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

X, y = [], []

# -------- Read class folders --------
classes = sorted(os.listdir(VIDEO_DIR))
label_map = {label: idx for idx, label in enumerate(classes)}

print("\nDynamic gesture classes found:")
for k, v in label_map.items():
    print(f"{v} -> {k}")

# -------- Process each class --------
for label in classes:
    class_path = os.path.join(VIDEO_DIR, label)
    videos = os.listdir(class_path)

    print(f"\nProcessing class '{label}' | Videos: {len(videos)}")

    for vid_name in videos:
        vid_path = os.path.join(class_path, vid_name)
        cap = cv2.VideoCapture(vid_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)

        cap.release()

        if len(frames) < FRAMES_PER_VIDEO:
            print(f"  Skipped {vid_name} (too short)")
            continue

        # Uniform frame sampling
        idxs = np.linspace(0, len(frames) - 1, FRAMES_PER_VIDEO, dtype=int)

        sequence = []

        for i in idxs:
            img_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            frame_landmarks = np.zeros(126, dtype=np.float32)

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_idx, hand_lms in enumerate(result.multi_hand_landmarks):
                    handedness = result.multi_handedness[hand_idx].classification[0].label

                    offset = 0 if handedness == "Left" else 63

                    for j, lm in enumerate(hand_lms.landmark):
                        frame_landmarks[offset + j*3 + 0] = lm.x
                        frame_landmarks[offset + j*3 + 1] = lm.y
                        frame_landmarks[offset + j*3 + 2] = lm.z

            sequence.append(frame_landmarks)

        X.append(sequence)
        y.append(label_map[label])

        print(f"  Processed: {vid_name}")

hands.close()

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

np.save(os.path.join(OUTPUT_DIR, "X_dynamic_words.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_dynamic_words.npy"), y)

with open(os.path.join(OUTPUT_DIR, "dynamic_word_labels.json"), "w") as f:
    json.dump(label_map, f, indent=4)

print("\n================ DONE ================")
print("Total samples:", X.shape[0])
print("X shape:", X.shape)   # (N, 30, 126)
print("y shape:", y.shape)
print("Saved to:", OUTPUT_DIR)
