import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Brain Tumor Detection System 🧠")

# Load models safely
try:
    vgg_model = tf.keras.models.load_model(
        "brain_tumor_vgg16.h5",
        compile=False
    )

    st.success("VGG16 model loaded successfully ✅")

except Exception as e:
    st.error(f"Error loading VGG model: {e}")

try:
    resnet_model = tf.keras.models.load_model(
        "brain_tumor_resnet.keras",
        compile=False
    )

    st.success("ResNet model loaded successfully ✅")

except Exception as e:
    st.error(f"Error loading ResNet model: {e}")

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

model_choice = st.selectbox(
    "Choose Model",
    ["VGG16", "ResNet50"]
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "png", "jpeg"]
)

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, use_container_width=True)

    img = preprocess(image)

    if model_choice == "VGG16":
        pred = vgg_model.predict(img)
    else:
        pred = resnet_model.predict(img)

    result = classes[np.argmax(pred)]

    confidence = np.max(pred) * 100

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}%")