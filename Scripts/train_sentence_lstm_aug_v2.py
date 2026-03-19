import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ================= CONFIG =================
PROJECT_ROOT = r"D:\ISL_PROJECT_FINAL"
DATA_DIR = os.path.join(PROJECT_ROOT, "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

X_PATH = os.path.join(DATA_DIR, "X_sentences.npy")
Y_PATH = os.path.join(DATA_DIR, "y_sentences.npy")

EPOCHS = 120
BATCH_SIZE = 8
TIMESTEPS = 30
FEATURES = 126

TEMPORAL_AUGMENTS = 3
TEMPORAL_JITTER = 2
# =========================================


# ---------- Temporal Augmentation ----------
def temporal_augment(X, y, augments=2, jitter=2):
    X_aug = [X]
    y_aug = [y]

    for _ in range(augments):
        X_shifted = np.zeros_like(X)

        for i in range(len(X)):
            shift = np.random.randint(-jitter, jitter + 1)
            X_shifted[i] = np.roll(X[i], shift, axis=0)

        X_aug.append(X_shifted)
        y_aug.append(y)

    return np.vstack(X_aug), np.hstack(y_aug)


# ---------- Load Data ----------
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


# ---------- FIX LABEL INDEXING (CRITICAL) ----------
unique_labels = sorted(np.unique(y))
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_map[v] for v in y], dtype=np.int64)

print("\nLabel remapping done")
print("Min label:", y.min())
print("Max label:", y.max())
print("Total classes:", len(unique_labels))


# ---------- Temporal Augmentation ----------
X, y = temporal_augment(
    X, y,
    augments=TEMPORAL_AUGMENTS,
    jitter=TEMPORAL_JITTER
)

print("\nAfter temporal augmentation:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ---------- Normalize ----------
mean = X.mean(axis=(0, 1), keepdims=True)
std = X.std(axis=(0, 1), keepdims=True) + 1e-8
X = (X - mean) / std


# ---------- Train / Validation Split ----------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

num_classes = len(np.unique(y))
print("\nFinal number of classes:", num_classes)


# ---------- Model ----------
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(TIMESTEPS, FEATURES)),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(128),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ---------- Callbacks ----------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "sentence_lstm_aug_best.h5"),
        monitor="val_loss",
        save_best_only=True
    )
]


# ---------- Train ----------
print("\nStarting final sentence training...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)


# ---------- Save Final ----------
final_path = os.path.join(MODEL_DIR, "sentence_lstm_aug_final.h5")
model.save(final_path)

print("\n================ DONE ================")
print("Best model: sentence_lstm_aug_best.h5")
print("Final model:", final_path)
