import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import os

# =========================
# CLASS ORDER
# =========================

model_classes = ["Benign","Malignant","Normal"]

# =========================
# LOAD MODELS
# =========================

model_paths = [
    "models/xception.keras",
    "models/inceptionresnet.keras",
    "models/mobilenet.keras"
]

models = []

for p in model_paths:
    model = tf.keras.models.load_model(p)
    models.append(model)

print("Models loaded successfully")

# =========================
# IMAGE PREPROCESSING
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

    probs_list = []

    for model in models:

        p = model.predict(img,verbose=0)[0]

        probs_list.append(p)

    avg_probs = np.mean(probs_list,axis=0)

    pred_index = np.argmax(avg_probs)

    pred_class = model_classes[pred_index]

    confidence = avg_probs[pred_index]*100

    # probability graph
    plt.figure(figsize=(7,4))

    bars = plt.bar(model_classes,avg_probs*100)

    bars[pred_index].set_color("red")

    plt.ylabel("Probability (%)")

    plt.title("Cancer Type Probability")

    plt.tight_layout()

    graph="prediction_graph.png"

    plt.savefig(graph)

    plt.close()

    result=f"""
Prediction: **{pred_class}**

Confidence: **{confidence:.2f}%**
"""

    probs={model_classes[i]:float(avg_probs[i]*100) for i in range(3)}

    return result,probs,graph

# =========================
# ABOUT TEXT
# =========================

about_text = """
## Lung Cancer Detection using Deep Learning Ensemble

This project presents an AI-powered system for automated lung cancer detection from CT scan images.

### Objective
The goal of this system is to assist medical professionals in early detection of lung cancer.

### Dataset
The system uses the **IQ-OTH/NCCD Lung Cancer Dataset** containing CT scan images categorized into:

• Benign  
• Malignant  
• Normal  

### Deep Learning Models Used
The system uses three CNN architectures:

• Xception  
• InceptionResNetV2  
• MobileNetV2  

### Ensemble Learning
Predictions from all three models are combined using ensemble averaging to improve accuracy.

### Technology Stack

• Python  
• TensorFlow / Keras  
• NumPy / Scikit-learn  
• Matplotlib  
• Gradio

### Application
This system can assist radiologists in automated lung cancer screening using CT images.
"""

# =========================
# PERFORMANCE TEXT
# =========================

performance_text = """
## Model Performance Evaluation

The performance of the models was evaluated using several metrics.

### Confusion Matrix
Shows classification accuracy across Benign, Malignant and Normal classes.

### ROC Curve
Shows model ability to distinguish between classes with AUC scores close to 1.

### Model Comparison
Accuracy, precision and recall comparison between Xception, InceptionResNetV2, MobileNetV2 and the ensemble.

### Observations
• MobileNetV2 achieved highest individual accuracy  
• Ensemble model improves robustness  
• All models achieve AUC greater than 0.99
"""

# =========================
# GRADIO APP
# =========================

with gr.Blocks(title="Lung Cancer Detection System") as app:

    gr.Markdown("# 🫁 Lung Cancer Detection System")

    with gr.Tabs():

        # ======================
        # Prediction Tab
        # ======================

        with gr.Tab("Prediction"):

            image = gr.Image(type="pil",label="Upload Lung CT Scan")

            output_text = gr.Markdown()

            output_probs = gr.Label()

            output_graph = gr.Image()

            btn = gr.Button("Predict")

            btn.click(
                predict,
                inputs=image,
                outputs=[output_text,output_probs,output_graph]
            )

        # ======================
        # Model Performance
        # ======================

        with gr.Tab("Model Performance"):

            gr.Markdown(performance_text)

            gr.Image("results/confusion_matrix.png",label="Confusion Matrix")

            gr.Image("results/roc_curves.png",label="ROC Curve")

            gr.Image("results/model_comparison.png",label="Accuracy Comparison")

            gr.Image("results/precision_comparison.png",label="Precision Comparison")

            gr.Image("results/recall_comparison.png",label="Recall Comparison")

            gr.Image("results/xception_accuracy.png",label="Xception Accuracy")

            gr.Image("results/xception_loss.png",label="Xception Loss")

            gr.Image("results/inceptionresnet_accuracy.png",label="InceptionResNetV2 Accuracy")

            gr.Image("results/inceptionresnet_loss.png",label="InceptionResNetV2 Loss")

            gr.Image("results/mobilenet_accuracy.png",label="MobileNetV2 Accuracy")

            gr.Image("results/mobilenet_loss.png",label="MobileNetV2 Loss")

        # ======================
        # About Tab
        # ======================

        with gr.Tab("About Project"):

            gr.Markdown(about_text)

app.launch()