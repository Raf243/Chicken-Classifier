import streamlit as st
from PIL import Image
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)
from predict import predict


st.set_page_config(page_title="Chicken Breed Classifier", page_icon="🐔")

st.title("🐔 Chicken Breed Classifier")
st.write("Upload a photo of a chicken to classify it as **Local Breed** or **Broiler**.")

uploaded_file = st.file_uploader("Upload Chicken Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded file temporarily since predict() expects a file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_file_path = os.path.join(current_dir, "temp_upload.jpg")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Classifying..."):
        result = predict(temp_file_path)

    label = result["Label"]
    confidence = result["Confidence"]
    probability = result["Probability"]

    st.subheader("Prediction")
    st.success(f"**{label}**")

    st.subheader("Confidence")
    st.progress(int(confidence))
    st.write(f"{confidence:.2f}%")

    st.subheader("Detailed Summary")
    st.write(f"""
    - **Predicted Breed:** {label}
    - **Confidence:** {confidence:.2f}%
    - **Raw Probability:** {probability:.4f}
    """)
else:
    st.info("Please upload an image to get a prediction.")