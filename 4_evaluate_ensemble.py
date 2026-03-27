import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    log_loss
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# CONFIG
# =========================

IMG_SIZE = 256
BATCH_SIZE = 32
TEST_DIR = "data/test"

model_classes = ["Benign","Malignant","Normal"]

os.makedirs("results", exist_ok=True)

# =========================
# LOAD MODELS
# =========================

models = [
    tf.keras.models.load_model("models/xception.keras"),
    tf.keras.models.load_model("models/inceptionresnet.keras"),
    tf.keras.models.load_model("models/mobilenet.keras")
]

print("✅ Models loaded")

# =========================
# LOAD DATA
# =========================

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

# =========================
# PREDICTIONS
# =========================

preds = []

for model in models:
    preds.append(model.predict(generator, verbose=1))

avg_pred = np.mean(preds, axis=0)
y_pred = np.argmax(avg_pred, axis=1)

# =========================
# ENSEMBLE ACCURACY & LOSS
# =========================

accuracy = np.mean(y_pred == y_true)
loss = log_loss(y_true_cat, avg_pred)

print(f"\n🔥 Ensemble Accuracy: {accuracy*100:.2f}%")
print(f"🔥 Ensemble Loss: {loss:.4f}")

# =========================
# ENSEMBLE PERFORMANCE GRAPH
# =========================

plt.figure()
metrics = ["Accuracy", "Loss"]
values = [accuracy, loss]

plt.bar(metrics, values)
plt.title("Ensemble Model Performance")
plt.ylabel("Value")

plt.savefig("results/ensemble_performance.png")
plt.close()

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=model_classes))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model_classes,
            yticklabels=model_classes)

plt.title("Confusion Matrix - Ensemble")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.savefig("results/confusion_matrix.png")
plt.close()

# =========================
# ROC CURVES
# =========================

plt.figure()

for i, model_pred in enumerate(preds):
    for j in range(3):
        fpr, tpr, _ = roc_curve(y_true_cat[:,j], model_pred[:,j])
        plt.plot(fpr, tpr, label=f"Model {i+1} - {model_classes[j]}")

# Ensemble ROC
for j in range(3):
    fpr, tpr, _ = roc_curve(y_true_cat[:,j], avg_pred[:,j])
    plt.plot(fpr, tpr, '--', label=f"Ensemble - {model_classes[j]}")

plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC Comparison")
plt.xlabel("FPR")
plt.ylabel("TPR")

plt.savefig("results/roc_comparison.png")
plt.close()

# =========================
# PROBABILITY DISTRIBUTION
# =========================

plt.figure(figsize=(10,5))

for i in range(3):
    plt.hist(avg_pred[:,i], bins=30, alpha=0.5, label=model_classes[i])

plt.legend()
plt.title("Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")

plt.savefig("results/prob_distribution.png")
plt.close()

# =========================
# CONFIDENCE HISTOGRAM
# =========================

confidence = np.max(avg_pred, axis=1)

plt.figure()
plt.hist(confidence, bins=30, color='green')

plt.title("Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Count")

plt.savefig("results/confidence_hist.png")
plt.close()

# =========================
# ERROR ANALYSIS
# =========================

print("\n⚠️ Error Analysis:")

for i in range(3):
    for j in range(3):
        if i != j and cm[i][j] > 0:
            print(f"{model_classes[i]} → {model_classes[j]}: {cm[i][j]} cases")

print("\n✅ All graphs saved in 'results/' folder")
