import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = 'models/waste_model.h5'
IMAGE_SIZE = (150, 150)

# Define the class labels in the same order as TensorFlow trained them
CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Function to load and preprocess the image
def preprocess_image(img_file):
    # Load the image and resize it to the expected model input size (150x150)
    img = image.load_img(img_file, target_size=IMAGE_SIZE)
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand dimensions to create a batch of 1 (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale the image (MANDATORY: must match training preprocessing)
    img_array /= 255.0
    return img_array

# --- Load the Model ---
@st.cache_resource
def load_trained_model():
    """Loads the model and caches it so it's not reloaded every time."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model. Ensure '{MODEL_PATH}' exists. Details: {e}")
        return None

model = load_trained_model()

# --- Streamlit App Layout ---
st.title("♻️ EcoSort AI: Waste Classification")
st.markdown("Upload an image of waste material to classify it into one of six categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # 1. Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # 2. Preprocess and predict
    if st.button("Classify Waste"):
        with st.spinner('Analyzing image...'):
            # Preprocess the image
            processed_img = preprocess_image(uploaded_file)
            
            # Get prediction probabilities
            predictions = model.predict(processed_img)
            
            # Get the index of the highest probability
            predicted_class_index = np.argmax(predictions)
            
            # Get the class label
            predicted_class = CLASS_LABELS[predicted_class_index]
            
            # Get the confidence level
            confidence = predictions[0][predicted_class_index] * 100

        # 3. Display Results
        st.success("Analysis Complete!")
        st.subheader(f"Classification Result:")
        st.markdown(f"The model predicts this is **{predicted_class.upper()}** with **{confidence:.2f}%** confidence.")

        # 4. Optional: Display all probabilities
        st.subheader("Confidence Scores:")
        
        # Create a dictionary of results
        results = {label: f"{prob * 100:.2f}%" for label, prob in zip(CLASS_LABELS, predictions[0])}
        
        # Display the results as a DataFrame (or simple table)
        st.table(results)

# Footer/Instructions
st.markdown("---")
st.markdown("Ensure you have run `python train_model.py` successfully to generate the `waste_model.h5` file.")