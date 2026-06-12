import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

st.title("Emotion Detection with DeepFace ✅")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image with PIL
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to NumPy array
    img_array = np.array(image)

    # Convert RGB to BGR (DeepFace uses OpenCV format)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    try:
        with st.spinner("Analyzing emotions..."):
            result = DeepFace.analyze(
                img_path=img_bgr,
                actions=['emotion'],
                detector_backend='retinaface',  # or 'mtcnn'
                enforce_detection=False
            )

        # Handle multiple faces
        if isinstance(result, list):
            for i, face_res in enumerate(result):
                st.write(f"Face {i+1} emotions:")
                st.json(face_res['emotion'])
        else:
            st.write("Detected emotions:")
            st.json(result['emotion'])

        st.success("✅ Emotion detection completed!")

    except Exception as e:
        st.error(f"Error analyzing emotions: {e}")






