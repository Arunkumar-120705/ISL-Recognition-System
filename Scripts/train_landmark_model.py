import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ================= CONFIG =================
PROJECT_ROOT = r"D:\ISL_PROJECT_FINAL"
DATA_DIR = os.path.join(PROJECT_ROOT, "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 32

# augmentation
NUM_AUGMENTS = 3
NOISE_STD = 0.008
# =========================================


# ---------- Landmark Noise Augmentation ----------
def augment_landmarks(X, y, num_augments=3, noise_std=0.008):
    X_aug = [X]
    y_aug = [y]

    for _ in range(num_augments):
        noise = np.random.normal(0.0, noise_std, X.shape)

        # Palm landmarks (wrist + base joints)
        noise[:, :15] *= 0.3

        # Reduce Z-noise (depth is unstable)
        noise[:, 2::3] *= 0.5

        X_aug.append(X + noise)
        y_aug.append(y)

    return np.vstack(X_aug).astype(np.float32), np.hstack(y_aug)


# ---------- Load Data ----------
X = np.load(os.path.join(DATA_DIR, "X_landmarks.npy"))
y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))

print("Original data shape:", X.shape)

# ---------- Augment ----------
X, y = augment_landmarks(X, y, NUM_AUGMENTS, NOISE_STD)
print("After augmentation:", X.shape)

# ---------- Normalize (SAVE THIS!) ----------
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8

np.save(os.path.join(MODEL_DIR, "numbers_landmark_mean.npy"), mean)
np.save(os.path.join(MODEL_DIR, "numbers_landmark_std.npy"), std)

X = (X - mean) / std

num_classes = len(np.unique(y))
print("Number of classes:", num_classes)

# ---------- Split ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ---------- Model ----------
model = Sequential([
    Dense(256, activation="relu", input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------- Callbacks ----------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "numbers_landmark_aug_best.h5"),
        monitor="val_loss",
        save_best_only=True
    )
]

# ---------- Train ----------
print("Starting training...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ---------- Save Final ----------
final_path = os.path.join(MODEL_DIR, "numbers_landmark_aug_final.h5")
model.save(final_path)

print("\nTraining completed")
print("Best model: numbers_landmark_aug_best.h5")
print("Final model:", final_path)
