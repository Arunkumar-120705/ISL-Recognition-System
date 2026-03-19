import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ================= CONFIG =================
PROJECT_ROOT = r"D:\ISL_PROJECT_FINAL"

X_PATH = os.path.join(PROJECT_ROOT, "landmarks", "X_dynamic_words.npy")
Y_PATH = os.path.join(PROJECT_ROOT, "landmarks", "y_dynamic_words.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "landmarks", "dynamic_word_labels.json")

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "dynamic_transformer_final.h5")
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "dynamic_transformer_best.h5")

SEQ_LEN = 30
FEATURE_DIM = 126   # ✅ IMPORTANT FIX
EPOCHS = 80
BATCH_SIZE = 8
LR = 1e-4

# ==========================================

print("Loading data...")
X = np.load(X_PATH)
y = np.load(Y_PATH)

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

NUM_CLASSES = len(label_map)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", NUM_CLASSES)

# ✅ Correct assertion
assert X.shape[1:] == (SEQ_LEN, FEATURE_DIM), "Feature shape mismatch!"

# ========== Train / Val split ==========
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== Positional Encoding ==========
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], tf.float32)

# ========== Transformer Block ==========
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.3):
        super().__init__()
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
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(rate)
        self.drop2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.drop1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.drop2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


# ========== Build Model ==========
inputs = tf.keras.Input(shape=(SEQ_LEN, FEATURE_DIM))

x = tf.keras.layers.Dense(64)(inputs)
x += positional_encoding(SEQ_LEN, 64)

x = TransformerBlock(d_model=64, num_heads=8, ff_dim=128)(x)
x = TransformerBlock(d_model=64, num_heads=8, ff_dim=128)(x)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ========== Callbacks ==========
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
]

# ========== Train ==========
print("\nStarting Dynamic Transformer Training...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

model.save(MODEL_SAVE_PATH)

print("\n================ DONE ================")
print("Best model :", os.path.basename(BEST_MODEL_PATH))
print("Final model:", MODEL_SAVE_PATH)
