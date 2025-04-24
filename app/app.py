import os
import numpy as np
import tensorflow as tf
import requests
import streamlit as st
from PIL import Image
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="OSTEO VISION",
    page_icon="🦵",
    layout="wide"
)

MODEL_URL = "https://osteovision-models.s3.us-west-2.amazonaws.com/OSTEO_VISION_MODEL_6_+Knee+Osteoarthritis+Detection+with+Fine-Tuned+ResNet152V2+on+dataset+3+MODEL_6.h5"
MODEL_PATH = "src\models\OSTEO_VISION_MODEL_6_ Knee Osteoarthritis Detection with Fine-Tuned ResNet152V2 on dataset 3 MODEL_6.h5"

def download_model_from_url(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    else:
        raise Exception(f"Failed to download model. HTTP Status code: {response.status_code}")

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model. Please wait...")
    download_model_from_url(MODEL_URL, MODEL_PATH)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

try:
    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[model.get_layer("global_average_pooling2d").input, model.output]
    )
except Exception as e:
    st.error(f"Failed to configure Grad-CAM: {e}")
    st.stop()

try:
    st.sidebar.image("app/img/jssate .png", caption="JSSATE-B", width=220)
    st.sidebar.image("app/img/jssaher.jpg", caption="JSSAHER", width=220)
except FileNotFoundError:
    st.warning("Sidebar images not found. Ensure the paths are correct.")

uploaded_file = st.sidebar.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

st.title("OSTEO VISION")

target_size = (224, 224)
class_names = ["KL-GRADE 0", "KL-GRADE 1", "KL-GRADE 2", "KL-GRADE 3", "KL-GRADE 4"]

def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

if uploaded_file:
    try:
        img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.xception.preprocess_input(img_array)

        with st.spinner("Analyzing the image..."):
            predictions = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(predictions)]
            prediction_probabilities = 100 * predictions

        st.subheader("Prediction Result ✅")
        st.metric(
            label="Predicted Severity Level",
            value=predicted_class,
            delta=f"Confidence: {np.max(prediction_probabilities):.2f}%"
        )

        heatmap = make_gradcam_heatmap(grad_model, img_array)
        heatmap_overlay = overlay_heatmap_on_image(
            tf.keras.preprocessing.image.img_to_array(img),
            heatmap
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Input Image")
            st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
        with col2:
            st.subheader("📊 Explainability with Grad-CAM")
            st.image(heatmap_overlay, caption="Grad-CAM Heatmap", use_column_width=True)

        st.subheader("📈 Prediction Confidence Levels")
        fig, ax = plt.subplots(figsize=(5, 2))

        bars = ax.barh(class_names, prediction_probabilities, color='skyblue')
        ax.set_xlim([0, 100])
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Prediction Confidence Levels")

        for bar, prob in zip(bars, prediction_probabilities):
            ax.text(
                prob + 2,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.2f}%",
                va='center',
                ha='left',
                fontsize=10
            )

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing the image: {e}")

else:
    st.info("Please upload an X-ray image to begin analysis.")