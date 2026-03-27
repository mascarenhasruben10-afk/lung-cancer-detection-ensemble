import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json

from tensorflow.keras.applications import Xception, InceptionResNetV2, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

# ======================
# CONFIG
# ======================

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 60

DATA_DIR = "data"

# ======================
# DATASET LOADING
# ======================

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/val",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("Classes:", class_names)

train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))

# ======================
# MODEL BUILDER
# ======================

def build_model(base_model, name):

    base = base_model(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)

    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(base.input, outputs, name=name)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ======================
# CREATE MODELS
# ======================

models = {
"xception": build_model(Xception, "xception"),
"inceptionresnet": build_model(InceptionResNetV2, "inceptionresnet"),
"mobilenet": build_model(MobileNetV2, "mobilenet")
}

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

histories = {}

# ======================
# TRAIN MODELS
# ======================

for name, model in models.items():

    print("\nTraining", name)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save(f"models/{name}.keras")

    histories[name] = history.history

    # accuracy graph
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.savefig(f"results/{name}_accuracy.png")

    # loss graph
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.savefig(f"results/{name}_loss.png")

# save history
with open("results/training_history.json", "w") as f:
    json.dump(histories, f)

print("Training complete")
