import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
from collections import deque
import os

# ================= CONFIG =================
SEQ_LEN = 30
FEATURE_DIM = 126

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "dynamic_transformer_best.h5")
LABELS_PATH = os.path.join(PROJECT_ROOT, "landmarks", "dynamic_word_labels.json")
# ==========================================


# ---------- Custom Transformer Block ----------
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

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
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config


# ---------- Load Model ----------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"TransformerBlock": TransformerBlock}
)
print("✅ Model loaded successfully")


# ---------- Load Labels ----------
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

id_to_label = {v: k for k, v in label_map.items()}


# ---------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------- Sequence Buffer ----------
sequence_buffer = deque(maxlen=SEQ_LEN)

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)
print("🎥 Webcam started. Press Q to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    if result.multi_hand_landmarks:
        all_landmarks = []
        for hand in result.multi_hand_landmarks[:2]:
            for lm in hand.landmark:
                all_landmarks.extend([lm.x, lm.y, lm.z])

        features[:len(all_landmarks)] = all_landmarks[:FEATURE_DIM]

    sequence_buffer.append(features)

    if len(sequence_buffer) == SEQ_LEN:
        X = np.expand_dims(np.array(sequence_buffer), axis=0)
        preds = model.predict(X, verbose=0)

        pred_id = int(np.argmax(preds))
        confidence = float(np.max(preds))

        if confidence >= 0.7:
            label = id_to_label[pred_id]
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

    cv2.imshow("ISL Dynamic Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()