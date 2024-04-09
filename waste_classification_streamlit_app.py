# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:08:43 2024

@author: vsingh1
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import fastai
from fastai.vision.all import *
from PIL import Image
import os
import time
import csv

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

@st.cache_resource
def load_model():
    path = Path('model_fastai.pkl')
    learn = load_learner(path)
    return learn

learn = load_model()

class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        super().__init__()
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame = frame.to_image()
        return av.VideoFrame.from_image(self.frame)

def save_image(image, label, feedback_dir, feedback, pred_prob=None):
    timestamp = int(time.time())
    filename = f"{label.replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join(feedback_dir, filename)
    image.save(filepath)
    
    # Write feedback data to CSV
    csv_filename = os.path.join(feedback_dir, 'feedback.csv')
    new_entry = [filename, label, str(pred_prob) if pred_prob else "N/A", feedback]
    
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_entry)

    st.write(f"Feedback saved. Image stored as: {filename}")

def main():
    st.title("BeClean")
    st.subheader("Play. Sort. Reward. Repeat.")

    categories = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
                  'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

    feedback_dir = 'feedback_images'
    if not os.path.exists(feedback_dir):
        os.makedirs(feedback_dir)

    # Initialize the CSV file with headers if it doesn't exist
    csv_filename = os.path.join(feedback_dir, 'feedback.csv')
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image ID', 'Predicted Label', 'Predicted Probability', 'Feedback'])

    tab1, tab2 = st.tabs(["Camera", "Result"])

    with tab1:
        webrtc_ctx = webrtc_streamer(key="example",
                                     video_transformer_factory=VideoTransformer,
                                     rtc_configuration=RTC_CONFIGURATION,
                                     media_stream_constraints={"video": True, "audio": False})

        capture_button = st.button("Capture")

    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None
        st.session_state.prediction = None
        st.session_state.pred = None
        st.session_state.pred_prob = None

    if capture_button and webrtc_ctx.state.playing:
        if webrtc_ctx.video_transformer.frame is not None:
            st.session_state.captured_image = webrtc_ctx.video_transformer.frame
            image = PILImage(st.session_state.captured_image)
            pred, pred_idx, probs = learn.predict(image)
            st.session_state.prediction = f'Prediction: {pred}; Probability: {probs[pred_idx].item() * 100:.2f}%'
            st.session_state.pred = pred
            st.session_state.pred_prob = probs[pred_idx].item()
            with tab1:
                st.write("Image captured. Check the 'Result' tab for the classification.")

    with tab2:
        if st.session_state.captured_image is not None:
            st.image(st.session_state.captured_image, caption=st.session_state.prediction if st.session_state.prediction else "No prediction made.")

            feedback = st.radio("Is the prediction correct?", ('Correct', 'Incorrect'), index=1)
            correct_label = st.selectbox("Select the correct label:", categories) if feedback == 'Incorrect' else st.session_state.pred
            if st.button('Submit Feedback'):
                save_image(st.session_state.captured_image, correct_label, feedback_dir, feedback, st.session_state.pred_prob)

if __name__ == "__main__":
    main()
