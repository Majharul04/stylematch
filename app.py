import streamlit as st
from PIL import Image
import numpy as np
import face_recognition
import cv2

st.set_page_config(page_title="StyleMatch AI", layout="centered")

st.title("👤 StyleMatch: AI Grooming Suggestion")
st.write("Upload your photo and get hairstyle and beard suggestions based on your face shape.")

uploaded_file = st.file_uploader("Upload a front-facing photo", type=["jpg", "jpeg", "png"])

def classify_face_shape(landmarks):
    jaw = landmarks['chin']
    left_jaw = np.linalg.norm(np.array(jaw[0]) - np.array(jaw[4]))
    right_jaw = np.linalg.norm(np.array(jaw[12]) - np.array(jaw[16]))
    jaw_width = (left_jaw + right_jaw) / 2
    face_length = np.linalg.norm(np.array(jaw[8]) - np.array(landmarks['nose_tip'][0]))

    if face_length > jaw_width * 1.5:
        return "Oval"
    elif abs(face_length - jaw_width) < 20:
        return "Round"
    elif jaw_width > face_length:
        return "Square"
    else:
        return "Heart"

def recommend_styles(face_shape):
    styles = {
        "Oval": ("Medium quiff, Pompadour", "Short boxed beard"),
        "Round": ("Faux hawk, Spiky top", "Full beard with sharp edges"),
        "Square": ("Side part, Crew cut", "Light stubble"),
        "Heart": ("Fringe, Textured crop", "Goatee, Balbo"),
    }
    return styles.get(face_shape, ("Classic Cut", "Clean Shave"))

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Photo", use_column_width=True)

    img_array = np.array(image.convert("RGB"))
    face_landmarks_list = face_recognition.face_landmarks(img_array)

    if face_landmarks_list:
        face_shape = classify_face_shape(face_landmarks_list[0])
        hairstyle, beard = recommend_styles(face_shape)

        st.subheader("🧠 AI Analysis Results")
        st.success(f"✅ Detected Face Shape: {face_shape}")
        st.success(f"✅ Recommended Hairstyle: {hairstyle}")
        st.success(f"✅ Recommended Beard: {beard}")
    else:
        st.error("😕 Could not detect a face. Please upload a clear front-facing image.")
else:
    st.info("Please upload your image to get suggestions.")
