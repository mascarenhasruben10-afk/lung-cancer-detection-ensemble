import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import os

# ========================= CONFIG =========================
MODEL_CLASSES = ["Benign", "Malignant", "Normal"]
ENSEMBLE_ACCURACY = "96.5%"

# ========================= LOAD MODELS =========================
model_paths = [
    "models/xception.h5",
    "models/inceptionresnet.h5",
    "models/mobilenet.h5"
]

print("Loading models...")
models = [tf.keras.models.load_model(p, compile=False) for p in model_paths]
print("✅ All 3 models loaded successfully!")

# ========================= PREPROCESS =========================
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    return np.expand_dims(img_array, axis=0)

# ========================= PREDICTION =========================
def predict(image: Image.Image):
    if image is None:
        return "⚠️ Please upload a Lung CT Scan image.", {}, None

    img = preprocess(image)
    preds = [model.predict(img, verbose=0)[0] for model in models]
    avg_probs = np.mean(preds, axis=0)
    pred_index = int(np.argmax(avg_probs))
    pred_class = MODEL_CLASSES[pred_index]
    confidence = float(avg_probs[pred_index] * 100)

    # Probability Chart
    plt.figure(figsize=(8, 4.5))
    colors = ["#e74c3c" if i == pred_index else "#3498db" for i in range(3)]
    bars = plt.bar(MODEL_CLASSES, avg_probs * 100, color=colors)
    plt.ylabel("Probability (%)")
    plt.title("Ensemble Prediction")
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1.5, f'{height:.1f}%', ha='center')
    plt.tight_layout()
    graph_path = "prediction_graph.png"
    plt.savefig(graph_path, dpi=220, bbox_inches='tight')
    plt.close()

    result = f"""
### 🫁 **Prediction: {pred_class}**

**Confidence:** `{confidence:.2f}%`

**Individual Model Predictions:**
• Xception → **{MODEL_CLASSES[np.argmax(preds[0])]}**  
• InceptionResNetV2 → **{MODEL_CLASSES[np.argmax(preds[1])]}**  
• MobileNetV2 → **{MODEL_CLASSES[np.argmax(preds[2])]}**
"""

    probs_dict = {cls: f"{prob*100:.2f}%" for cls, prob in zip(MODEL_CLASSES, avg_probs)}
    return result, probs_dict, graph_path

# ========================= ABOUT TEXT =========================
about_text = """
# 🫁 Lung Cancer Detection using Ensemble Deep Learning

This system classifies lung CT scans into:

- **Benign**
- **Malignant**
- **Normal**

### Models Used
- Xception
- InceptionResNetV2
- MobileNetV2

### How It Works
Each model predicts probabilities independently.  
Final prediction is obtained using **ensemble averaging**.

### Why Ensemble?
- Reduces individual model errors
- Improves accuracy  
- More reliable predictions

### Results
- **Ensemble Accuracy: 96.5%**
- MobileNetV2: Best individual model
- Ensemble: Most stable overall
"""

# ========================= GRADIO APP =========================
with gr.Blocks(
    title="Lung Cancer Detection System",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown("# 🫁 Lung Cancer Detection System")

    with gr.Tabs():
        # Prediction Tab
        with gr.Tab("🔬 Prediction"):
            gr.Markdown("### Upload a Lung CT Scan to get prediction")
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil", 
                        label="Upload Lung CT Scan",
                        height=420
                    )
                    with gr.Row():
                        predict_btn = gr.Button("🔍 Predict", variant="primary", size="large")
                        clear_btn = gr.Button("🗑️ Clear", size="large")

                with gr.Column(scale=1):
                    output_text = gr.Markdown(label="Prediction Result")
                    output_probs = gr.JSON(label="Class Probabilities")
                    output_graph = gr.Image(label="Probability Distribution Chart")

            predict_btn.click(
                predict, 
                inputs=image_input, 
                outputs=[output_text, output_probs, output_graph]
            )
            
            clear_btn.click(
                lambda: (None, "", {}, None),
                inputs=None,
                outputs=[image_input, output_text, output_probs, output_graph]
            )

        # Model Performance Tab
        with gr.Tab("📊 Model Performance"):
            gr.Markdown("## Model Performance & Evaluation Results")
            
            with gr.Row():
                gr.Image("results/confusion_matrix.png", label="Confusion Matrix", height=500)
            
            with gr.Row():
                with gr.Column():
                    gr.Image("results/xception_accuracy.png", label="Xception - Accuracy")
                    gr.Image("results/xception_loss.png", label="Xception - Loss")
                with gr.Column():
                    gr.Image("results/inceptionresnet_accuracy.png", label="InceptionResNetV2 - Accuracy")
                    gr.Image("results/inceptionresnet_loss.png", label="InceptionResNetV2 - Loss")
            
            with gr.Row():
                gr.Image("results/ensemble_performance.png", label="Ensemble Overall Performance")
            
            with gr.Row():
                gr.Image("results/model_comparison.png", label="Overall Accuracy Comparison")
                gr.Image("results/precision_comparison.png", label="Precision Comparison")
                gr.Image("results/recall_comparison.png", label="Recall Comparison")
            
            with gr.Row():
                gr.Image("results/roc_curves.png", label="ROC Curves")
                gr.Image("results/roc_comparison.png", label="ROC Comparison")

        # About Tab
        with gr.Tab("ℹ️ About Project"):
            gr.Markdown(about_text)

# ========================= LAUNCH =========================
if __name__ == "__main__":
    print("🚀 Starting Lung Cancer Detection Web App...")
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
