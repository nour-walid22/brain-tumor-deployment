import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load models
resnet_model = tf.keras.models.load_model("brain_tumor_resnet.h5")
vgg_model = tf.keras.models.load_model("brain_tumor_vgg16.h5")

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("Brain Tumor Detection")

model_choice = st.selectbox("Choose Model", ["ResNet50", "VGG16"])

uploaded_file = st.file_uploader("Upload MRI Image")

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    processed = preprocess(image)

    if model_choice == "ResNet50":
        pred = resnet_model.predict(processed)
    else:
        pred = vgg_model.predict(processed)

    classes_list = ['glioma', 'meningioma', 'notumor', 'pituitary']
    st.success("Prediction: " + classes_list[np.argmax(pred)])
