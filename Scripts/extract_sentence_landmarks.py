import os
import cv2
import json
import numpy as np
import mediapipe as mp

# ================= CONFIG =================
PROJECT_ROOT = r"D:\ISL_PROJECT_FINAL"
VIDEO_DIR = os.path.join(PROJECT_ROOT, "Videos_Sentence_Level")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "landmarks")

FIXED_FRAMES = 30          # frames per video
LANDMARK_DIM = 126         # 2 hands × 21 × 3
IMG_SIZE = 256
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

X, y = [], []

# -------- Sentence labels ----------
sentences = sorted(os.listdir(VIDEO_DIR))
label_map = {name: idx for idx, name in enumerate(sentences)}

print("\nSentence Classes:")
for k, v in label_map.items():
    print(f"{v} -> {k}")

# -------- Helper functions ----------
def extract_frame_landmarks(results):
    frame_vec = np.zeros(LANDMARK_DIM, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return frame_vec

    idx = 0
    for hand in results.multi_hand_landmarks[:2]:
        for lm in hand.landmark:
            frame_vec[idx:idx+3] = [lm.x, lm.y, lm.z]
            idx += 3

    return frame_vec


def uniform_sample(frames, target_len):
    if len(frames) >= target_len:
        idxs = np.linspace(0, len(frames)-1, target_len).astype(int)
        return [frames[i] for i in idxs]
    else:
        while len(frames) < target_len:
            frames.append(frames[-1])
        return frames

# -------- Main extraction ----------
total_videos = 0

for sentence in sentences:
    folder = os.path.join(VIDEO_DIR, sentence)
    videos = [v for v in os.listdir(folder) if v.endswith(".mp4")]

    print(f"\nProcessing sentence: '{sentence}' ({len(videos)} videos)")

    for video_name in videos:
        video_path = os.path.join(folder, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_name}")
            continue

        frames = []
        frame_count = 0

        print(f"  ▶ Reading video: {video_name}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)
            lm_vec = extract_frame_landmarks(result)
            frames.append(lm_vec)

        cap.release()

        if len(frames) == 0:
            print("    ⚠ No landmarks detected, skipping")
            continue

        frames = uniform_sample(frames, FIXED_FRAMES)

        X.append(frames)
        y.append(label_map[sentence])
        total_videos += 1

        print(f"    ✔ Frames extracted: {frame_count} → used: {FIXED_FRAMES}")

hands.close()

# -------- Save outputs ----------
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

np.save(os.path.join(OUTPUT_DIR, "X_sentences.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_sentences.npy"), y)

with open(os.path.join(OUTPUT_DIR, "sentence_class_labels.json"), "w") as f:
    json.dump(label_map, f, indent=4)

print("\n================ DONE ================")
print("Total sentence videos processed:", total_videos)
print("X shape:", X.shape)   # (samples, 30, 126)
print("y shape:", y.shape)
print("Saved to:", OUTPUT_DIR)
