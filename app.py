import os
import streamlit as st
from predict import predict_video

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 Human Activity Recognition")
st.write("Upload a video to predict the human activity.")

uploaded_file = st.file_uploader(
    "Choose a video",
    type=["avi", "mp4", "mov"]
)

if uploaded_file is not None:

    os.makedirs("dataset/uploads", exist_ok=True)

    video_path = os.path.join("dataset/uploads", uploaded_file.name)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    if st.button("Predict Activity"):

        with st.spinner("Predicting..."):

            activity, confidence = predict_video(video_path)

        st.success("Prediction Complete")

        st.subheader(f"Activity: {activity}")
        st.subheader(f"Confidence: {confidence:.2%}")