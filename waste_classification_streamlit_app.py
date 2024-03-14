# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 03:25:14 2024

@author: vsingh1
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import numpy as np
import tensorflow as tf
from PIL import Image

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Load the model
@st.cache_resource
def load_model():
    model_path = r'C:\Users\vsingh1\Desktop\BeClean\waste_classification_modeling\waste_classifier_model.h5'
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Define the labels
labels = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame = frame
        return frame  # Just return the original frame

def main():
    st.title("Real-time Waste Classification App")

    tab1, tab2 = st.tabs(["Camera", "Results"])

    with tab1:
        webrtc_ctx = webrtc_streamer(key="example",
                                     rtc_configuration=RTC_CONFIGURATION,
                                     video_transformer_factory=VideoTransformer,
                                     media_stream_constraints={"video": True, "audio": False})

        capture_button = st.button("Capture")

    if capture_button:
        with tab2:
            if webrtc_ctx.state.playing:
                # Capture the frame to be processed
                frame = webrtc_ctx.video_transformer.frame
                if frame is not None:
                    # Convert the frame to PIL Image for processing
                    image = frame.to_image().convert('RGB')
                    st.image(image, caption='Captured Image', use_column_width=True)

                    # Process the image for classification
                    img = image.resize((180, 180))
                    img_array = np.expand_dims(np.array(img), axis=0)
                    predictions = model.predict(img_array)
                    predicted_class = labels[np.argmax(predictions[0])]
                    #confidence = np.max(predictions[0]) * 100

                    # Display the classification result
                    st.write(f"Prediction: {predicted_class}")
                else:
                    st.warning("No frame captured. Please try again.")

if __name__ == "__main__":
    main()
