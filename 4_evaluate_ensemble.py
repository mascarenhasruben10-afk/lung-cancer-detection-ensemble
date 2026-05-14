import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.metrics import confusion_matrix, log_loss
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 256
BATCH_SIZE = 32
TEST_DIR = "data/test"

model_classes = ["Benign","Malignant","Normal"]

# ✅ FIXED PATHS
models = [
    tf.keras.models.load_model("models/xception.h5", compile=False),
    tf.keras.models.load_model("models/inceptionresnet.h5", compile=False),
    tf.keras.models.load_model("models/mobilenet.h5", compile=False)
]

datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

y_true = generator.classes
y_true_cat = label_binarize(y_true, classes=[0,1,2])

preds = [model.predict(generator) for model in models]

avg_pred = np.mean(preds, axis=0)
y_pred = np.argmax(avg_pred, axis=1)

accuracy = np.mean(y_pred == y_true)
loss = log_loss(y_true_cat, avg_pred)

print("Accuracy:", accuracy)
print("Loss:", loss)

cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
