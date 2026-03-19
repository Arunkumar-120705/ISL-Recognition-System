import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ================= CONFIG =================
PROJECT_ROOT = r"D:\ISL_PROJECT_FINAL"

DATA_DIR  = os.path.join(PROJECT_ROOT, "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS = 60
BATCH_SIZE = 32

# augmentation
NUM_AUGMENTS = 2
NOISE_STD = 0.006
# =========================================


# -------- Landmark Noise Augmentation --------
def augment_landmarks(X, y, num_augments=2, noise_std=0.005):
    X_aug = [X]
    y_aug = [y]

    for _ in range(num_augments):
        noise = np.random.normal(0.0, noise_std, X.shape)

        # reduce noise on palm joints (both hands)
        noise[:, :15] *= 0.3
        noise[:, 63:78] *= 0.3

        X_aug.append(X + noise)
        y_aug.append(y)

    return np.vstack(X_aug), np.hstack(y_aug)


# -------- Load Data --------
X = np.load(os.path.join(DATA_DIR, "X_alphabets_2hand.npy"))
y = np.load(os.path.join(DATA_DIR, "y_alphabets_2hand.npy"))

print("Original data shape:", X.shape)

# -------- Augmentation --------
X, y = augment_landmarks(X, y, NUM_AUGMENTS, NOISE_STD)
print("After augmentation:", X.shape)

# -------- Normalization (CRITICAL) --------
mean = X.mean(axis=0)
std  = X.std(axis=0) + 1e-8

X = (X - mean) / std

# save normalization for webcam use
np.save(os.path.join(MODEL_DIR, "alphabets_2hand_mean.npy"), mean)
np.save(os.path.join(MODEL_DIR, "alphabets_2hand_std.npy"), std)

num_classes = len(np.unique(y))
print("Number of classes:", num_classes)

# -------- Split --------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------- Model --------
model = Sequential([
    Dense(512, activation="relu", input_shape=(126,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.2),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------- Callbacks --------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "alphabets_landmark_2hand_best.h5"),
        monitor="val_loss",
        save_best_only=True
    )
]

# -------- Train --------
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# -------- Save Final --------
final_path = os.path.join(MODEL_DIR, "alphabets_landmark_2hand_final.h5")
model.save(final_path)

print("\n===================================")
print("TRAINING COMPLETED")
print("Best model : alphabets_landmark_2hand_best.h5")
print("Final model:", final_path)
print("===================================")
