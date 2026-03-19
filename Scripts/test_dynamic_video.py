import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers

# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "dynamic_transformer_best.h5")
LABEL_PATH = os.path.join(PROJECT_ROOT, "landmarks", "dynamic_word_labels.json")

SEQ_LEN = 30
FEATURE_DIM = 126   # 21 landmarks × 3 × 2 hands (max)

# =========================
# TRANSFORMER BLOCK (MUST MATCH TRAINING)
# =========================

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "rate": self.dropout1.rate,
        })
        return config

# =========================
# LOAD LABEL MAP
# =========================

with open(LABEL_PATH, "r") as f:
    label_map = json.load(f)

inv_label_map = {int(v): k for k, v in label_map.items()}

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"TransformerBlock": TransformerBlock},
    compile=False
)

print("✅ Model loaded successfully")

# =========================
# MEDIAPIPE INITIALIZATION
# =========================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# LANDMARK EXTRACTION
# =========================

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        frame_landmarks = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks[:2]:
                for lm in hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])

        # Pad if only one hand or no hand
        while len(frame_landmarks) < FEATURE_DIM:
            frame_landmarks.append(0.0)

        frames.append(frame_landmarks)

    cap.release()

    # Normalize sequence length
    if len(frames) >= SEQ_LEN:
        frames = frames[:SEQ_LEN]
    else:
        while len(frames) < SEQ_LEN:
            frames.append([0.0] * FEATURE_DIM)

    return np.array(frames)

# =========================
# MAIN TEST FUNCTION
# =========================

def test_video(video_path):
    print(f"\n🎥 Testing video: {video_path}")

    X = extract_landmarks_from_video(video_path)
    X = np.expand_dims(X, axis=0)  # (1, 30, 126)

    predictions = model.predict(X)[0]
    class_id = np.argmax(predictions)
    confidence = predictions[class_id]

    label = inv_label_map[class_id]

    print(f"🟢 Predicted Word : {label}")
    print(f"📊 Confidence     : {confidence:.4f}")

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("python scripts/test_dynamic_video.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print("❌ Video file not found")
        sys.exit(1)

    test_video(video_path)