import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import label_binarize

# ======================
# CONFIG
# ======================

IMG_SIZE = 256
BATCH_SIZE = 32

DATA_DIR = "data"

MODEL_PATHS = [
"models/xception.keras",
"models/inceptionresnet.keras",
"models/mobilenet.keras"
]

MODEL_NAMES = [
"Xception",
"InceptionResNetV2",
"MobileNetV2"
]

os.makedirs("results", exist_ok=True)

# ======================
# LOAD DATA
# ======================

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

test_ds = test_ds.map(lambda x, y: (x/255.0, y))

X_test = np.concatenate([x for x,_ in test_ds], axis=0)
y_true = np.concatenate([y for _,y in test_ds], axis=0)

print("Test samples:", len(X_test))

# ======================
# LOAD MODELS
# ======================

models = []

for path in MODEL_PATHS:

    print("Loading", path)

    model = tf.keras.models.load_model(path)

    models.append(model)

# ======================
# MODEL PREDICTIONS
# ======================

probs_list = []

for model in models:

    probs = model.predict(X_test, verbose=0)

    probs_list.append(probs)

# ======================
# ENSEMBLE
# ======================

avg_probs = np.mean(probs_list, axis=0)

ensemble_pred = np.argmax(avg_probs, axis=1)

# ======================
# CONFUSION MATRIX
# ======================

cm = confusion_matrix(y_true, ensemble_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix - Ensemble")

plt.savefig("results/confusion_matrix.png")

plt.show()

# ======================
# ROC CURVES
# ======================

y_bin = label_binarize(y_true, classes=range(len(class_names)))

plt.figure()

for i in range(len(class_names)):

    fpr, tpr, _ = roc_curve(y_bin[:,i], avg_probs[:,i])

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{class_names[i]} AUC={roc_auc:.3f}")

plt.plot([0,1],[0,1],"k--")

plt.title("ROC Curve")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.savefig("results/roc_curves.png")

plt.show()

# ======================
# MODEL COMPARISON
# ======================

accuracies = []

for probs in probs_list:

    pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, pred)

    accuracies.append(acc)

ensemble_acc = accuracy_score(y_true, ensemble_pred)

labels = MODEL_NAMES + ["Ensemble"]

values = accuracies + [ensemble_acc]

plt.figure()

plt.bar(labels, values)

plt.title("Overall Accuracy")

plt.ylabel("Accuracy")

plt.savefig("results/model_comparison.png")

plt.show()

# ======================
# PRECISION / RECALL
# ======================

precisions = []
recalls = []

for probs in probs_list:

    pred = np.argmax(probs, axis=1)

    precisions.append(precision_score(y_true, pred, average="macro"))

    recalls.append(recall_score(y_true, pred, average="macro"))

precisions.append(precision_score(y_true, ensemble_pred, average="macro"))

recalls.append(recall_score(y_true, ensemble_pred, average="macro"))

# precision graph

plt.figure()

plt.bar(labels, precisions)

plt.title("Overall Precision")

plt.savefig("results/precision_comparison.png")

plt.show()

# recall graph

plt.figure()

plt.bar(labels, recalls)

plt.title("Overall Recall")

plt.savefig("results/recall_comparison.png")

plt.show()

# ======================
# REPORT
# ======================

print("\nClassification Report\n")

print(classification_report(y_true, ensemble_pred, target_names=class_names))