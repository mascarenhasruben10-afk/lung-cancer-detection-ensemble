import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG
# =========================

model_classes = ["Benign","Malignant","Normal"]
ENSEMBLE_ACCURACY = "96.5%"

# =========================
# LOAD MODELS
# =========================

model_paths = [
    "models/xception.keras",
    "models/inceptionresnet.keras",
    "models/mobilenet.keras"
]

print("Models Folder:", os.listdir("models"))

models = [tf.keras.models.load_model(p) for p in model_paths]

print("✅ Models loaded successfully")

# =========================
# PREPROCESS
# =========================

def preprocess(image):
    image = image.resize((256,256))
    img = np.array(image)/255.0

    if len(img.shape)==2:
        img = np.stack((img,)*3,axis=-1)

    img = np.expand_dims(img,axis=0)
    return img

# =========================
# PREDICTION
# =========================

def predict(image):

    img = preprocess(image)

    preds = []
    for model in models:
        preds.append(model.predict(img,verbose=0)[0])

    avg_probs = np.mean(preds,axis=0)

    pred_index = np.argmax(avg_probs)
    pred_class = model_classes[pred_index]
    confidence = avg_probs[pred_index]*100

    # =========================
    # GRAPH
    # =========================

    plt.figure(figsize=(7,4))
    bars = plt.bar(model_classes,avg_probs*100)

    bars[pred_index].set_color("red")

    plt.ylabel("Probability (%)")
    plt.title("Cancer Type Probability")

    plt.tight_layout()
    graph_path = "prediction_graph.png"
    plt.savefig(graph_path)
    plt.close()

    # =========================
    # OUTPUT
    # =========================

    result = f"""
### Prediction: {pred_class}

**Confidence: {confidence:.2f}%**

### Individual Model Predictions:
- Xception → {model_classes[np.argmax(preds[0])]}
- InceptionResNet → {model_classes[np.argmax(preds[1])]}
- MobileNet → {model_classes[np.argmax(preds[2])]}
"""

    probs = {
        model_classes[i]: round(float(avg_probs[i]),3)
        for i in range(3)
    }

    return result, probs, graph_path

# =========================
# ABOUT TEXT
# =========================

about_text = """
## 🫁 Lung Cancer Detection using Ensemble Deep Learning

This system classifies lung CT scans into:

- Benign
- Malignant
- Normal

### 🧠 Models Used
- Xception
- InceptionResNetV2
- MobileNetV2

### ⚙️ How It Works
Each model predicts probabilities independently.  
Final prediction is obtained using **ensemble averaging**.

### 🎯 Why Ensemble?
- Reduces individual model errors
- Improves accuracy
- More reliable predictions

### 🏆 Results
- Ensemble Accuracy: **96.5%**
- MobileNetV2: Best individual model
- Ensemble: Most stable model
"""

# =========================
# PERFORMANCE TEXT
# =========================

performance_text = f"""
## 📊 Model Performance

### Ensemble Accuracy
**{ENSEMBLE_ACCURACY}**

### Key Insights

• MobileNetV2 achieved highest individual accuracy  
• Ensemble model improves robustness  
• AUC scores close to 1  

### Why Ensemble Works?

Combining multiple models reduces variance and improves generalization.
"""

# =========================
# UI
# =========================

with gr.Blocks(title="Lung Cancer Detection System") as app:

    gr.Markdown("# 🫁 Lung Cancer Detection System")

    with gr.Tabs():

        # ======================
        # PREDICTION TAB
        # ======================
        with gr.Tab("Prediction"):

            gr.Markdown(f"### Ensemble Model Accuracy: {ENSEMBLE_ACCURACY}")

            image = gr.Image(type="pil", label="Upload Lung CT Scan")

            output_text = gr.Markdown()
            output_probs = gr.Label()
            output_graph = gr.Image()

            btn = gr.Button("Predict")

            btn.click(
                predict,
                inputs=image,
                outputs=[output_text, output_probs, output_graph]
            )

        # ======================
        # MODEL PERFORMANCE TAB
        # ======================
        with gr.Tab("Model Performance"):

            gr.Markdown(performance_text)

            # MODEL COMPARISON
            gr.Markdown("## 📊 Model Comparison")
            gr.Image("results/model_comparison.png", label="Accuracy Comparison")
            gr.Image("results/precision_comparison.png", label="Precision Comparison")
            gr.Image("results/recall_comparison.png", label="Recall Comparison")

            # CONFUSION MATRIX
            gr.Markdown("## 🔍 Confusion Matrix")
            gr.Image("results/confusion_matrix.png")

            # ROC
            gr.Markdown("## 📈 ROC Curves")
            gr.Image("results/roc_curves.png")
            gr.Image("results/roc_comparison.png")

            # DISTRIBUTION
            gr.Markdown("## 📊 Prediction Distribution")
            gr.Image("results/prob_distribution.png")
            gr.Image("results/confidence_hist.png")

            # TRAINING GRAPHS
            gr.Markdown("## 🧠 Training Performance")

            gr.Markdown("### Xception")
            gr.Image("results/xception_accuracy.png")
            gr.Image("results/xception_loss.png")

            gr.Markdown("### InceptionResNetV2")
            gr.Image("results/inceptionresnet_accuracy.png")
            gr.Image("results/inceptionresnet_loss.png")

            gr.Markdown("### MobileNetV2")
            gr.Image("results/mobilenet_accuracy.png")
            gr.Image("results/mobilenet_loss.png")

        # ======================
        # ABOUT TAB
        # ======================
        with gr.Tab("About Project"):
            gr.Markdown(about_text)

# =========================
# RUN
# =========================

app.launch()
