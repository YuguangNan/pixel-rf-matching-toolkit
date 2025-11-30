import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks, regularizers
import matplotlib.pyplot as plt

# =========================
# Import utilities
# =========================
from src.utils.model_utils import (
    se_block,
    conv_bn_swish,
    multi_scale_res_block_se,
    build_forward_model
)
from src.utils.sparam_utils import (
    flat_to_complex,
    db20
)

# =========================
# Environment and basic setup
# =========================
os.makedirs("checkpoints", exist_ok=True)

# Optional: mixed precision (for RTX30/40)
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    MIXED_PRECISION = True
except Exception:
    MIXED_PRECISION = False
    pass


# =========================
# Load dataset
# =========================
data = np.load("sparam_dataset.npz")
X = data["X"]
y = data["y"]

# Ensure input shape is (N, 15, 15, 1)
if X.ndim == 3:
    X = X[..., np.newaxis]

X = X.astype(np.float32)
y = y.astype(np.float32)

print("Loaded dataset:", X.shape, y.shape)


# =========================
# Build model
# =========================
model = build_forward_model(
    input_shape=(15, 15, 1),
    output_dim=y.shape[-1]
)

# Optimizer: AdamW preferred, fallback to Adam
try:
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4
    )
except Exception:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


# =========================
# Custom loss
# =========================
def hybrid_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.7 * mse + 0.3 * mae


model.compile(
    optimizer=optimizer,
    loss=hybrid_loss,
    metrics=["mae"]
)

model.summary()


# =========================
# Callbacks
# =========================
checkpoint_cb = callbacks.ModelCheckpoint(
    "checkpoints/dnn_model_20GHz_best.h5",
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)

earlystop_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

csvlog_cb = callbacks.CSVLogger(
    "checkpoints/train_log.csv",
    append=False
)


# =========================
# Train (original method preserved)
# =========================
history = model.fit(
    X, y,
    batch_size=128,
    epochs=120,
    validation_split=0.1,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb, csvlog_cb],
    verbose=1
)

model.save("checkpoints/dnn_20GHz.h5")


# =========================
# SE channel attention heatmap
# =========================
def visualize_se_heatmap(model, stage_names, figsize=(10, 5)):
    se_weights = []

    for stage in stage_names:
        try:
            fc2_layer = model.get_layer(stage + "_fc2")
        except ValueError:
            print(f"[WARN] Layer not found: {stage}_fc2")
            continue

        weights, _ = fc2_layer.get_weights()
        avg_weight = np.mean(weights, axis=0)  # shape (channels,)
        se_weights.append(avg_weight.astype(np.float32))

    if len(se_weights) == 0:
        print("❌ No SE weights collected. Check stage names.")
        return

    # Make into a matrix (pad if channels differ)
    max_len = max(len(w) for w in se_weights)
    mat = np.zeros((len(se_weights), max_len), dtype=np.float32)

    for i, w in enumerate(se_weights):
        mat[i, :len(w)] = w

    # Normalize
    mat = np.log1p(mat - np.min(mat) + 1e-8)
    mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-8)

    # =========================
    # Training loss curves
    # =========================
    def plot_training_curves(history, save_path="loss_curve.png"):
        train_loss = history.history["loss"]
        val_loss   = history.history["val_loss"]
        epochs = np.arange(1, len(train_loss) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Training Loss", linewidth=2)
        plt.plot(epochs, val_loss,   label="Validation Loss", linewidth=2)

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training & Validation Loss", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

    plot_training_curves(history, save_path="checkpoints/loss_curve.png")

    # =========================
    # SE heatmap
    # =========================
    plt.figure(figsize=figsize)
    im = plt.imshow(mat, aspect="auto", cmap="plasma")
    plt.colorbar(im, label="Normalized SE Channel Weight")
    plt.xticks(np.arange(mat.shape[1]), labels=np.arange(mat.shape[1]), rotation=90)
    plt.yticks(np.arange(len(stage_names)), labels=stage_names)
    plt.title("SE Channel Attention Heatmap")
    plt.xlabel("Channel Index")
    plt.ylabel("Stage")
    plt.tight_layout()
    plt.show()


# Call visualization
visualize_se_heatmap(model, [
    "stage1_block2_se",
    "stage2_block2_se",
    "stage3_block2_se",
    "stage4_block2_se"
])
